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

"""Dump the REFERENCE model's GRADIENT norms on fixed batches (same-batch parity).

Run with the reference (``openpi-comet-pytorch-mixed``) Python 3.11 venv + the
reference src on path, on a GPU:

    .venv/bin/python tests/unit_tests/_ref_model_grad_dump.py <out_dir>

Mirrors ``_ref_model_loss_dump.py`` but instruments forward + backward (no
``no_grad``) to measure the per-batch GRADIENT, at ``train=True, rng=None`` (the
production deterministic-crop training path) on the real ``models_pytorch_new.Pi0``
loaded from ``pi05_base_pytorch_new`` (cast to bf16, like the RLinf training build).
Records, per batch, the global grad norm, per-major-module grad norms, the number of
params with a non-None grad, the clipped (max_norm=1.0) norm, and the loss. Writes
``<out_dir>/ref_grad_batches.npz`` (the batches, the source of truth for the RLinf
probe), ``<out_dir>/ref_grad_dump.json`` (the reference grad metrics + noise/time),
and prints one ``REF_MODEL_GRAD <json>`` line. The RLinf-side
``tools/sft_grad_parity_probe.py`` loads the SAME batches + noise/time and compares.
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
_N_BATCHES = 4
_BATCH_SIZE = 8
_SEED = 1234
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
# Top-level module-name prefixes to bucket per-module grad norms.
_MODULES = ("llm", "img", "action_in_proj", "action_out_proj", "state_proj", "time_mlp")


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _grad_metrics(model):
    named = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
    sq = dict.fromkeys(_MODULES, 0.0)
    other = 0.0
    total_sq = 0.0
    for n, p in named:
        g2 = float(p.grad.detach().float().pow(2).sum())
        total_sq += g2
        bucket = next((m for m in _MODULES if n.startswith(m + ".") or n == m), None)
        if bucket is not None:
            sq[bucket] += g2
        else:
            other += g2
    gn = total_sq**0.5
    per_module = {m: sq[m] ** 0.5 for m in _MODULES}
    per_module["_other"] = other**0.5
    clipped = min(gn, 1.0)  # clip_grad_norm_(max_norm=1.0) scales to <=1.0
    return {
        "global_grad_norm": gn,
        "per_module_grad_norm": per_module,
        "n_params_with_grad": len(named),
        "clipped_grad_norm": clipped,
        "was_clipped": gn > 1.0,
    }


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
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        run_cfg = dataclasses.replace(
            cfg, data=data, batch_size=_BATCH_SIZE, num_workers=0
        )
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))

        model = pi0_new.Pi0(pi0_config_new.Pi0Config(**cfg.model.__dict__)).to(device)
        model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
        # bf16 compute (RLinf training build); eval() for a deterministic backward.
        model = model.to(torch.bfloat16).eval()

        def _mv(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            if isinstance(x, dict):
                return {k: _mv(v) for k, v in x.items()}
            return x

        rs = np.random.RandomState(_SEED)
        store, grads, losses, noises, times = {}, [], [], [], []
        for _ in range(_N_BATCHES):
            observation, actions = next(it)
            for k in _IMG:
                store.setdefault(f"image__{k}", []).append(_np(observation.images[k]))
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

            B = actions.shape[0]
            np_noise = rs.randn(*actions.shape).astype(np.float32)
            np_time = rs.beta(1.5, 1.0, size=(B,)).astype(np.float32) * 0.999 + 0.001
            noises.append(np_noise)
            times.append(np_time)
            noise = torch.from_numpy(np_noise).to(device)
            time = torch.from_numpy(np_time).to(device)

            obs = dataclasses.replace(
                observation,
                **{
                    f.name: _mv(getattr(observation, f.name))
                    for f in dataclasses.fields(observation)
                },
            )
            act = actions.to(device)
            model.zero_grad(set_to_none=True)
            loss = (
                model.compute_loss(
                    obs, act, train=True, rng=None, noise=noise, time=time
                )
                .float()
                .mean()
            )
            loss.backward()
            losses.append(float(loss))
            grads.append(_grad_metrics(model))

        np.savez(
            f"{out_dir}/ref_grad_batches.npz",
            **{k: np.stack(v) for k, v in store.items()},
        )
        np.savez(
            f"{out_dir}/ref_grad_noise_time.npz",
            noise=np.stack(noises),
            time=np.stack(times),
        )
        result.update(
            ok=True,
            loss=losses,
            grad=grads,
            meta={
                "config": _CONFIG,
                "weights": _WEIGHTS,
                "n_batches": _N_BATCHES,
                "batch_size": _BATCH_SIZE,
                "seed": _SEED,
                "ref_src": _REF_SRC,
                "model": "openpi.models_pytorch_new.pi0.Pi0 (bf16, eval, train=True/rng=None)",
                "batches_npz": f"{out_dir}/ref_grad_batches.npz",
                "noise_time_npz": f"{out_dir}/ref_grad_noise_time.npz",
            },
        )
        with open(f"{out_dir}/ref_grad_dump.json", "w", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        result["err"] = f"{type(e).__name__}: {str(e)[:400]}"

    print(
        "REF_MODEL_GRAD "
        + json.dumps({k: result[k] for k in ("ok", "err") if k in result})
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
