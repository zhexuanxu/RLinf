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

"""Reference side of the R28 SAME-INPUT N-step training replay.

Codex R27 review required a MEASURED same-input comparison (not code reading):
feed RLinf's and the reference's stacks the IDENTICAL fixed batches + fixed
noise/time from identical base weights for N steps, with the production
optimizer/clip/LR, and compare the per-step loss trajectories.

This script (run in the reference ``openpi-comet-pytorch-mixed`` py-3.11 venv)
draws N fixed batches from the real reference loader, samples N fixed noise/time
(numpy, so the RLinf consumer reproduces them exactly), and runs N optimizer
steps on the real ``models_pytorch_new.Pi0`` with the SAME single-GPU emulation
the RLinf probe uses: fp32 master weights + bf16 compute params (the R26-validated
bf16 forward), ``torch.optim.AdamW`` (betas (0.9,0.95), eps 1e-8, wd 1e-10),
``clip_grad_norm_(max_norm=1.0)``, and the openpi_cosine warmup LR
(peak 2.5e-5, warmup 1000 -> init peak/(warmup+1)). It dumps the batches
(``ref_replay_batches.npz``), the noise/time (``ref_replay_noise_time.npz``), and
per-step metrics + meta (``ref_replay_dump.json``). The RLinf-side
``tools/sft_replay_parity_probe.py`` replays the SAME inputs through RLinf's Pi0
and compares the per-step loss trajectories.

NOTE: single-GPU (no FSDP sharding); the manual fp32-master + bf16-compute step is
identical on both sides, so a trajectory match proves the model+optimizer+clip+LR
are equivalent on identical inputs. It does NOT exercise the 8-rank FSDP all-reduce.

    .venv/bin/python tests/unit_tests/_ref_train_replay_dump.py <out_dir>
"""

import dataclasses
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_N_STEPS = 20
_BATCH_SIZE = 8
_SEED = 1234
# AdamW + LR (read from the reference config; identical on both sides).
_BETAS = (0.9, 0.95)
_EPS = 1e-8
_WD = 1e-10
_CLIP = 1.0
_PEAK_LR = 2.5e-5
_WARMUP = 1000
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _lr_at(step):
    # openpi_cosine warmup (train_pytorch_new.py:491-494); first _N_STEPS << warmup.
    init_lr = _PEAK_LR / (_WARMUP + 1)
    return init_lr + (_PEAK_LR - init_lr) * step / _WARMUP


def run_replay(model, compute_loss, batches, noises, times, device):
    """Shared N-step fp32-master + bf16-compute training loop.

    ``model`` params are already bf16 (compute); ``compute_loss(model, obs, act,
    noise, time) -> scalar`` is the side-specific forward. Returns per-step rows.
    """
    import torch

    master = [p.detach().clone().float() for p in model.parameters()]
    for m in master:
        m.requires_grad_(True)
    opt = torch.optim.AdamW(master, lr=_PEAK_LR, betas=_BETAS, eps=_EPS, weight_decay=_WD)
    params = list(model.parameters())
    rows = []
    for step in range(len(batches)):
        lr = _lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        with torch.no_grad():  # sync fp32 master -> bf16 compute params
            for p, m in zip(params, master):
                p.data = m.data.to(torch.bfloat16)
        model.zero_grad(set_to_none=True)
        for m in master:
            m.grad = None
        loss = compute_loss(model, batches[step], noises[step], times[step]).float().mean()
        loss.backward()
        for p, m in zip(params, master):  # bf16 grad -> fp32 master grad
            m.grad = p.grad.detach().float() if p.grad is not None else None
        gn = torch.nn.utils.clip_grad_norm_(master, max_norm=_CLIP)
        opt.step()
        rows.append(
            {
                "step": step,
                "loss": float(loss),
                "grad_norm": float(gn),
                "clipped_grad_norm": min(float(gn), _CLIP),
                "was_clipped": float(gn) > _CLIP,
                "lr": lr,
            }
        )
        print(f"REF step {step}: loss={float(loss):.5f} grad_norm={float(gn):.4f} lr={lr:.3e}", flush=True)
    return rows


def main(out_dir):
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
        base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        run_cfg = dataclasses.replace(cfg, data=data, batch_size=_BATCH_SIZE, num_workers=0)
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))

        model = pi0_new.Pi0(pi0_config_new.Pi0Config(**cfg.model.__dict__)).to(device)
        model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
        model = model.to(torch.bfloat16).train()

        def _mv(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _mv(v) for k, v in x.items()}
            return x

        rs = np.random.RandomState(_SEED)
        store, batches, noises, times = {}, [], [], []
        for _ in range(_N_STEPS):
            observation, actions = next(it)
            for k in _IMG:
                store.setdefault(f"image__{k}", []).append(_np(observation.images[k]))
                m = getattr(observation, "image_masks", None)
                if m is not None:
                    store.setdefault(f"image_mask__{k}", []).append(_np(m[k]))
            store.setdefault("state", []).append(_np(observation.state))
            store.setdefault("tokenized_prompt", []).append(_np(observation.tokenized_prompt))
            tpm = getattr(observation, "tokenized_prompt_mask", None)
            if tpm is not None:
                store.setdefault("tokenized_prompt_mask", []).append(_np(tpm))
            store.setdefault("actions", []).append(_np(actions))

            B = actions.shape[0]
            np_noise = rs.randn(*actions.shape).astype(np.float32)
            np_time = rs.beta(1.5, 1.0, size=(B,)).astype(np.float32) * 0.999 + 0.001
            noises.append(np_noise)
            times.append(np_time)

            obs = dataclasses.replace(
                observation,
                **{
                    f.name: _mv(getattr(observation, f.name))
                    for f in dataclasses.fields(observation)
                },
            )
            batches.append((obs, actions.to(device)))

        noise_t = [torch.from_numpy(n).to(device) for n in noises]
        time_t = [torch.from_numpy(t).to(device) for t in times]

        def _compute_loss(m, batch, noise, time):
            obs, act = batch
            return m.compute_loss(obs, act, train=True, rng=None, noise=noise, time=time)

        rows = run_replay(model, _compute_loss, batches, noise_t, time_t, device)

        np.savez(
            f"{out_dir}/ref_replay_batches.npz",
            **{k: np.stack(v) for k, v in store.items()},
        )
        np.savez(
            f"{out_dir}/ref_replay_noise_time.npz",
            noise=np.stack(noises),
            time=np.stack(times),
        )
        result.update(
            ok=True,
            steps=rows,
            meta={
                "config": _CONFIG,
                "weights": _WEIGHTS,
                "n_steps": _N_STEPS,
                "batch_size": _BATCH_SIZE,
                "seed": _SEED,
                "ref_src": _REF_SRC,
                "model": "openpi.models_pytorch_new.pi0.Pi0 (fp32 master + bf16 compute, train=True/rng=None)",
                "optimizer": f"torch.optim.AdamW betas={_BETAS} eps={_EPS} wd={_WD}",
                "clip": _CLIP,
                "lr": f"openpi_cosine warmup peak={_PEAK_LR} warmup={_WARMUP} init=peak/(warmup+1)",
                "batches_npz": f"{out_dir}/ref_replay_batches.npz",
                "noise_time_npz": f"{out_dir}/ref_replay_noise_time.npz",
            },
        )
        with open(f"{out_dir}/ref_replay_dump.json", "w", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1200:]

    print("REF_TRAIN_REPLAY " + json.dumps({k: result[k] for k in ("ok", "err") if k in result}))
    if "tb" in result and not result["ok"]:
        print(result["tb"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
