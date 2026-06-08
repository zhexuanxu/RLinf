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

"""Reference side of the step-0 2x2 cross-feed (run in the reference py-3.11 venv).

Materializes the reference's ACTUAL step-0 global batch (the first world_size micro-batches
of the production rank-0 fanout loader = 256 fully-transformed frames), generates a shared,
reproducible flow noise/time (numpy, so the RLinf side reproduces it exactly), and computes
the reference model's production-faithful step-0 loss + pre-clip fp32 GLOBAL grad norm on
BOTH the reference batch and the (externally supplied) RLinf default batch, holding
weights/norm-stats/noise/time/dtype/train-mode identical.

The grad is computed exactly as the 8-GPU production logs it: fp32 master weights + bf16
compute (the R26-validated forward), the global-mean loss over the 256-frame batch
(micro-batch 32 x world_size, backward(loss_i / world_size) -> mean grad == the FSDP
all-reduced mean), and ``clip_grad_norm_`` returns the pre-clip fp32 global norm.

    .venv/bin/python tests/unit_tests/_ref_step0_2x2_dump.py <out_dir> <rlinf_batch.npz>
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
_WORLD_SIZE = 8  # rank-0 fanout pulls world_size micro-batches per global step
_MICRO = 32
_SEED = 4242  # shared noise/time seed (numpy; reproduced verbatim on the RLinf side)
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_CLIP = 1.0


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _content_hash(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _obs_to_store(observation, actions):
    """Flatten an Observation+actions micro-batch to a dict of numpy arrays."""
    d = {"actions": _np(actions)}
    for k in _IMG:
        d[f"image__{k}"] = _np(observation.images[k])
        m = getattr(observation, "image_masks", None)
        if m is not None and k in m:
            d[f"image_mask__{k}"] = _np(m[k])
    d["state"] = _np(observation.state)
    d["tokenized_prompt"] = _np(observation.tokenized_prompt)
    tpm = getattr(observation, "tokenized_prompt_mask", None)
    if tpm is not None:
        d["tokenized_prompt_mask"] = _np(tpm)
    return d


def _frame_ids_from_store(micro_stores):
    """Value-independent (state-hash, prompt-hash) per frame as a stable identity proxy."""
    ids = []
    for st in micro_stores:
        n = st["state"].shape[0]
        for i in range(n):
            ids.append(hashlib.sha256(
                np.ascontiguousarray(st["state"][i]).tobytes()
                + np.ascontiguousarray(st["tokenized_prompt"][i]).tobytes()
            ).hexdigest()[:16])
    return ids


def _build_ref_model(device):
    import openpi.models_pytorch_new.pi0 as pi0_new
    import openpi.models_pytorch_new.pi0_config as pi0_config_new
    import openpi.training.config as _config
    import safetensors.torch
    import torch

    cfg = _config.get_config(_CONFIG)
    model = pi0_new.Pi0(pi0_config_new.Pi0Config(**cfg.model.__dict__)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model.to(torch.bfloat16).train(), cfg


def _obs_from_store(store, device, Observation):
    import torch

    def t(a, dtype=None):
        x = torch.from_numpy(np.asarray(a)).to(device)
        return x.to(dtype) if dtype is not None else x

    d = {
        "image": {k: t(store[f"image__{k}"], torch.float32) for k in _IMG},
        "image_mask": {k: t(store[f"image_mask__{k}"]) for k in _IMG if f"image_mask__{k}" in store},
        "state": t(store["state"], torch.float32),
        "tokenized_prompt": t(store["tokenized_prompt"]).long(),
    }
    if "tokenized_prompt_mask" in store:
        d["tokenized_prompt_mask"] = t(store["tokenized_prompt_mask"]).bool()
    return Observation.from_dict(d)


def _prod_loss_grad(model, micro_stores, noise, time, device):
    """Production-faithful step-0 loss + pre-clip fp32 GLOBAL grad norm over the 256-frame
    batch: fp32 master + bf16 compute, global-mean loss, mean grad == FSDP all-reduced mean."""
    import torch

    from openpi.models import model as _model  # Observation

    master = [p.detach().clone().float().requires_grad_(True) for p in model.parameters()]
    params = list(model.parameters())
    with torch.no_grad():
        for p, m in zip(params, master):
            p.data = m.data.to(torch.bfloat16)
    model.zero_grad(set_to_none=True)
    for m in master:
        m.grad = None
    n_micro = len(micro_stores)
    losses = []
    for i, st in enumerate(micro_stores):
        obs = _obs_from_store(st, device, _model.Observation)
        act = torch.from_numpy(st["actions"]).to(device, torch.float32)
        nz = torch.from_numpy(noise[i * _MICRO:(i + 1) * _MICRO]).to(device)
        tm = torch.from_numpy(time[i * _MICRO:(i + 1) * _MICRO]).to(device)
        loss_i = model.compute_loss(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
        (loss_i / n_micro).backward()
        losses.append(float(loss_i))
    for p, m in zip(params, master):
        m.grad = p.grad.detach().float() if p.grad is not None else None
    gn = float(torch.nn.utils.clip_grad_norm_(master, max_norm=_CLIP))
    return {"loss": float(np.mean(losses)), "grad_norm": gn}


def main(out_dir, rlinf_batch_npz):
    import os

    sys.path.insert(0, _REF_SRC)
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False}
    try:
        import torch

        import openpi.training.data_loader as _data_loader

        device = "cuda"
        model, cfg = _build_ref_model(device)

        base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
        assets_cfg = __import__("openpi.training.config", fromlist=["AssetsConfig"]).AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets_cfg)
        # num_workers=8 = the production fanout's 8 lanes (num_workers=0 would be 1 lane and
        # NOT the actual production step-0 data).
        run_cfg = dataclasses.replace(cfg, data=data, batch_size=_MICRO, num_workers=8)
        # The reference rank-0 fanout pulls world_size micro-batches per global step: the
        # first world_size yields are the actual step-0 global batch (256 frames).
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))
        ref_micros = []
        for _ in range(_WORLD_SIZE):
            observation, actions = next(it)
            ref_micros.append(_obs_to_store(observation, actions))

        # shared, reproducible flow noise/time over the full 256-frame batch.
        total = _WORLD_SIZE * _MICRO
        ah = int(ref_micros[0]["actions"].shape[1])
        ad = int(ref_micros[0]["actions"].shape[2])
        rs = np.random.RandomState(_SEED)
        noise = rs.randn(total, ah, ad).astype(np.float32)
        time = (rs.beta(1.5, 1.0, size=(total,)).astype(np.float32) * 0.999 + 0.001)

        # persist the reference batch + the shared noise/time for the RLinf side.
        flat = {f"m{i}__{k}": v for i, st in enumerate(ref_micros) for k, v in st.items()}
        np.savez(os.path.join(out_dir, "ref_step0_batch.npz"), n_micro=_WORLD_SIZE, **flat)
        np.savez(os.path.join(out_dir, "step0_noise_time.npz"), noise=noise, time=time)

        # load the RLinf default step-0 batch (produced on the RLinf side).
        rl = np.load(rlinf_batch_npz)
        n_rl = int(rl["n_micro"])
        rlinf_micros = [
            {k.split("__", 1)[1]: rl[k] for k in rl.files if k.startswith(f"m{i}__")}
            for i in range(n_rl)
        ]

        # ref model on BOTH batches with the SAME shared noise/time.
        ref_on_ref = _prod_loss_grad(model, ref_micros, noise, time, device)
        ref_on_rlinf = _prod_loss_grad(model, rlinf_micros, noise, time, device)

        result.update(
            ok=True,
            config=_CONFIG,
            world_size=_WORLD_SIZE,
            micro=_MICRO,
            total_frames=total,
            noise_seed=_SEED,
            ref_batch_frame_ids=_frame_ids_from_store(ref_micros),
            ref_batch_content_hash=_content_hash(
                [st["state"] for st in ref_micros] + [st["actions"] for st in ref_micros]
            ),
            cell_ref_batch_ref_model=ref_on_ref,
            cell_rlinf_batch_ref_model=ref_on_rlinf,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-2000:]
    with open(os.path.join(out_dir, "ref_step0_2x2.json"), "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("REF_STEP0_2X2", json.dumps({"ok": result["ok"], "err": result.get("err")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
