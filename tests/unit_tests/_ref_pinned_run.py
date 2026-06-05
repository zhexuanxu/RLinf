# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reference arm of the R36 PINNED same-input residual experiment.

Run in the reference ``openpi-comet-pytorch-mixed`` py-3.11 venv on a GPU. Builds the
reference loader's rank-0 FANOUT (the production batch order: pull ``world_size``
successive micro-batches of 32 per step -> global batch 256), runs the real
``models_pytorch_new.Pi0`` for N steps at effective batch 256 via gradient accumulation
(8 chunks x 32; the shared fp32-weights + autocast-bf16 AdamW+clip+warmup-LR loop) on
those batches + a SHARED fixed noise/time (numpy, so the RLinf arm reproduces them), and
dumps: the batches (``ref_pinned_batches.npz``, 50 x 256), the noise/time
(``ref_pinned_noise_time.npz``), per-step loss, and per-step batch/noise/time sha256
hashes + a provenance block. The RLinf arm (``tools/sft_pinned_residual_probe.py``)
loads the IDENTICAL batches + noise/time and runs RLinf with the same loop; if the
per-step losses match 50/50, the production first-50 residual is the per-step INPUT.

    .venv/bin/python tests/unit_tests/_ref_pinned_run.py <out_dir> <n_steps> <world_size>
"""

import dataclasses
import hashlib
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_MICRO = 32
_SEED = 1234
_BETAS, _EPS, _WD, _CLIP = (0.9, 0.95), 1e-8, 1e-10, 1.0
_PEAK_LR, _WARMUP = 2.5e-5, 1000
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _lr_at(step):
    init = _PEAK_LR / (_WARMUP + 1)
    return init + (_PEAK_LR - init) * step / _WARMUP


def _hash(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _git_rev(repo):
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def main(out_dir, n_steps, world_size):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import openpi.models_pytorch_new.pi0 as pi0_new
        import openpi.models_pytorch_new.pi0_config as pi0_config_new
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        import safetensors.torch
        import torch

        device = "cuda"
        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        # Per-rank micro-batch 32; the rank-0 fanout pulls `world_size` of them per global step.
        run_cfg = dataclasses.replace(cfg, data=data, batch_size=_MICRO, num_workers=0)
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))

        model = pi0_new.Pi0(pi0_config_new.Pi0Config(**cfg.model.__dict__)).to(device)
        model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
        model = model.train()  # fp32 weights; autocast bf16 in the loop

        def _mv(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _mv(v) for k, v in x.items()}
            return x

        rs = np.random.RandomState(_SEED)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=_PEAK_LR,
            betas=_BETAS,
            eps=_EPS,
            weight_decay=_WD,
            foreach=False,
        )
        store, noises, times, rows = {}, [], [], []
        glob = _MICRO * world_size  # 256
        for step in range(n_steps):
            lr = _lr_at(step)
            for pg in opt.param_groups:
                pg["lr"] = lr
            # rank-0 fanout: pull world_size successive micro-batches -> the global-256 batch.
            chunks = [next(it) for _ in range(world_size)]
            step_noise = rs.randn(glob, *chunks[0][1].shape[1:]).astype(np.float32)
            step_time = (
                rs.beta(1.5, 1.0, size=(glob,)).astype(np.float32) * 0.999 + 0.001
            )
            noises.append(step_noise)
            times.append(step_time)
            opt.zero_grad(set_to_none=True)
            step_loss, st_h, ac_h = 0.0, [], []
            for c, (observation, actions) in enumerate(chunks):
                lo = c * _MICRO
                obs = dataclasses.replace(
                    observation,
                    **{
                        f.name: _mv(getattr(observation, f.name))
                        for f in dataclasses.fields(observation)
                    },
                )
                act = actions.to(device)
                nz = torch.from_numpy(step_noise[lo : lo + _MICRO]).to(device)
                tm = torch.from_numpy(step_time[lo : lo + _MICRO]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = (
                        model.compute_loss(
                            obs, act, train=True, rng=None, noise=nz, time=tm
                        )
                        .float()
                        .mean()
                    )
                (loss * (_MICRO / glob)).backward()
                step_loss += float(loss) * (_MICRO / glob)
                for k in _IMG:
                    store.setdefault(f"image__{k}", []).append(
                        _np(observation.images[k])
                    )
                    m = getattr(observation, "image_masks", None)
                    if m is not None:
                        store.setdefault(f"image_mask__{k}", []).append(_np(m[k]))
                store.setdefault("state", []).append(_np(observation.state))
                store.setdefault("tokenized_prompt", []).append(
                    _np(observation.tokenized_prompt)
                )
                tpm = getattr(observation, "tokenized_prompt_mask", None)
                if tpm is not None:
                    store.setdefault("tokenized_prompt_mask", []).append(_np(tpm))
                store.setdefault("actions", []).append(_np(actions))
                st_h.append(_hash(_np(observation.state)))
                ac_h.append(_hash(_np(actions)))
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_CLIP)
            opt.step()
            rows.append(
                {
                    "step": step,
                    "loss": round(step_loss, 6),
                    "grad_norm": round(float(gn), 5),
                    "state_hash": _hash(np.array(st_h, dtype="S16")),
                    "actions_hash": _hash(np.array(ac_h, dtype="S16")),
                    "noise_hash": _hash(step_noise),
                    "time_hash": _hash(step_time),
                }
            )
            print(
                f"REF_PINNED step {step}: loss={step_loss:.5f} grad_norm={float(gn):.4f}",
                flush=True,
            )

        np.savez(
            f"{out_dir}/ref_pinned_batches.npz",
            **{k: np.stack(v) for k, v in store.items()},
        )
        np.savez(
            f"{out_dir}/ref_pinned_noise_time.npz",
            noise=np.stack(noises),
            time=np.stack(times),
        )
        result.update(
            ok=True,
            steps=rows,
            meta={
                "config": _CONFIG,
                "n_steps": n_steps,
                "world_size": world_size,
                "global_batch": glob,
                "micro": _MICRO,
                "seed": _SEED,
                "ref_src": _REF_SRC,
                "ref_git_rev": _git_rev(_REF_SRC.rsplit("/src", 1)[0]),
                "ref_command": " ".join(sys.argv),
                "returncode": 0,
                "model": "openpi.models_pytorch_new.pi0.Pi0 (fp32 weights + autocast bf16, rank-0 fanout)",
                "batches_npz": f"{out_dir}/ref_pinned_batches.npz",
                "noise_time_npz": f"{out_dir}/ref_pinned_noise_time.npz",
            },
        )
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["ok"] = False
        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1200:]

    # Always persist the dump (success OR failure) so the orchestrator can audit a failed arm.
    with open(f"{out_dir}/ref_pinned_dump.json", "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")

    print(
        "REF_PINNED " + json.dumps({k: result[k] for k in ("ok", "err") if k in result})
    )
    if "tb" in result and not result["ok"]:
        print(result["tb"])
    if not result.get("ok"):
        sys.exit(1)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(out, n, ws)
