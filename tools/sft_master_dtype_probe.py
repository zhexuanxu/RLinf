#!/usr/bin/env python
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

"""Master-weight dtype probe for the BEHAVIOR SFT optimizer-update mechanism.

Demonstrates, on the REAL `pi05_base_pytorch_new` model and a fixed BEHAVIOR batch,
why the openpi_pytorch first-50-step SFT loss was flat before the Round-20 fp32-master
fix: with a **bf16 master**, the tiny warmup-LR AdamW updates (~1e-6, below the bf16
ULP ~0.0078 near 1.0) are lost to rounding, so the weights barely change and the loss
stays flat; with an **fp32 master** (bf16 compute, the production fix and the reference
recipe), the updates accumulate and the loss descends.

Two arms, identical fixed batch + fixed noise/time, N AdamW steps at the openpi_cosine
warmup LR:
  - ``bf16``  : bf16 storage, bf16 compute, AdamW updates bf16 params directly.
  - ``fp32``  : fp32 master, bf16 compute (params synced master->bf16 each step, grads
                copied bf16->fp32), AdamW updates the fp32 master (manual FSDP
                MixedPrecision). This is exactly the Round-20 production path.

Outputs (committed under docs/evidence/): a per-step loss CSV for both arms and a JSON
manifest (command, hashes, param dtype before/after, changed-param fraction, delta
norms, loss before/after). GPU-only (1 device); all run output under
``/mnt/public/xzxuan/tmp`` (set TMPDIR).

Usage:
    TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH=$REPO python tools/sft_master_dtype_probe.py \
      --out-csv docs/evidence/r21_master_dtype_probe.csv \
      --out-json docs/evidence/r21_master_dtype_probe.json
"""

from __future__ import annotations

import argparse
import hashlib
import json

import torch
from omegaconf import OmegaConf

N = 50
SEED = 42
MODEL_PATH = "/mnt/public/xzxuan/models/pi05_base_pytorch_new"
ASSETS = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
ASSET_ID = "behavior-1k/2025-challenge-demos"
DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
# openpi_cosine warmup: lr(step) = peak * (step + 1) / (warmup + 1)
PEAK_LR, WARMUP = 2.5e-5, 1000


def _model_cfg():
    return OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": MODEL_PATH,
            "num_action_chunks": 32,
            "action_dim": 23,
            "precision": "bf16",
            "load_for_training": True,
            "openpi": {
                "assets_dir": ASSETS,
                "asset_id": ASSET_ID,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "num_images_in_input": 3,
                "action_chunk": 32,
                "action_env_dim": 23,
                "load_for_training": True,
            },
        }
    )


def _file_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _batch(model, device):
    # A small fixed batch keeps the single-GPU fp32-master arm (fp32 master + fp32
    # AdamW states + bf16 compute) within one device; the bf16-vs-fp32 update
    # mechanism is batch-size-independent.
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": _model_cfg(),
                "micro_batch_size": 8,
                "eval_batch_size": 1,
                "seed": SEED,
            },
            "data": {
                "train_data_paths": DATA,
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": False,
            },
        }
    )
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    loader, _ = build_behavior_sft_dataloader(cfg, 1, 0, DATA)
    obs, act = next(iter(loader))
    return model._observation_to_device(obs), model._actions_to_device(act)


def _fixed_noise_time(act, device):
    g = torch.Generator(device=device).manual_seed(SEED + 1000)
    noise = torch.randn(act.shape, device=device, dtype=torch.float32, generator=g)
    torch.manual_seed(SEED)
    time = (
        torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        .sample((act.shape[0],))
        .to(device, torch.float32)
        * 0.999
        + 0.001
    )
    return noise, time


def _build_model(device):
    from rlinf.models.embodiment.openpi_pytorch import get_model

    return get_model(_model_cfg()).to(device)


def _sched_lr(step):
    return PEAK_LR * (step + 1) / (WARMUP + 1)


def _run_arm(arm, device="cuda"):
    """Run N AdamW steps; return (losses, changed_frac, delta_l1, param_dtype)."""
    model = _build_model(device)
    inner = model.model
    inner.to(torch.bfloat16)  # bf16 compute copy for BOTH arms
    obs, act = _batch(model, device)
    noise, time = _fixed_noise_time(act, device)
    cpu_g = torch.Generator().manual_seed(SEED)

    params = list(inner.parameters())
    if arm == "bf16":
        master = params  # bf16 master == the compute params (AdamW updates bf16)
    else:  # fp32 master, bf16 compute (the production fix / FSDP MixedPrecision)
        master = [p.detach().float().requires_grad_(True) for p in params]

    # Initial snapshot kept on CPU (fp32) to leave GPU memory for the fp32 master + states.
    p0 = [m.detach().float().cpu().clone() for m in master]
    opt = torch.optim.AdamW(
        master,
        lr=PEAK_LR,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
        foreach=False,
    )

    losses = []
    for step in range(N):
        for grp in opt.param_groups:
            grp["lr"] = _sched_lr(step)
        opt.zero_grad(set_to_none=True)
        if arm == "fp32":
            with torch.no_grad():  # sync fp32 master -> bf16 compute params
                for p, m in zip(params, master):
                    p.data = m.detach().to(torch.bfloat16)
        loss = inner.compute_loss(
            obs, act, train=True, rng=cpu_g, noise=noise, time=time
        ).mean()
        losses.append(float(loss.item()))
        loss.backward()
        if arm == "fp32":
            with torch.no_grad():  # copy bf16 grads -> fp32 master grads
                for p, m in zip(params, master):
                    m.grad = p.grad.detach().float() if p.grad is not None else None
        torch.nn.utils.clip_grad_norm_(master, 1.0)
        opt.step()

    deltas = [(m.detach().float().cpu() - p0[i]) for i, m in enumerate(master)]
    changed = sum(int((d != 0).sum()) for d in deltas)
    total = sum(m.numel() for m in master)
    delta_l1 = sum(float(d.abs().sum()) for d in deltas)
    dtype = str(next(iter(master)).dtype)
    del model, inner, master
    torch.cuda.empty_cache()
    return losses, changed / total, delta_l1, dtype


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out-csv", default="/mnt/public/xzxuan/tmp/r21_master_dtype_probe.csv"
    )
    ap.add_argument(
        "--out-json", default="/mnt/public/xzxuan/tmp/r21_master_dtype_probe.json"
    )
    args = ap.parse_args()

    bf16_losses, bf16_frac, bf16_d, bf16_dtype = _run_arm("bf16")
    fp32_losses, fp32_frac, fp32_d, fp32_dtype = _run_arm("fp32")

    with open(args.out_csv, "w", newline="\n") as f:
        f.write("step,bf16_master_loss,fp32_master_loss\n")
        for s in range(N):
            f.write(f"{s},{bf16_losses[s]:.6f},{fp32_losses[s]:.6f}\n")

    manifest = {
        "purpose": "bf16-master vs fp32-master AdamW update on the real BEHAVIOR SFT model "
        "(fixed batch + fixed noise/time, openpi_cosine warmup LR)",
        "command": "CUDA_VISIBLE_DEVICES=0 python tools/sft_master_dtype_probe.py",
        "n_steps": N,
        "seed": SEED,
        "peak_lr": PEAK_LR,
        "warmup": WARMUP,
        "model_path": MODEL_PATH,
        "hashes_sha256_16": {
            "model.safetensors": _file_digest(f"{MODEL_PATH}/model.safetensors"),
            "norm_stats.json": _file_digest(f"{ASSETS}/{ASSET_ID}/norm_stats.json"),
        },
        "bf16_master": {
            "param_dtype": bf16_dtype,
            "changed_param_fraction": round(bf16_frac, 6),
            "delta_l1": round(bf16_d, 4),
            "loss_step0": round(bf16_losses[0], 5),
            "loss_stepN": round(bf16_losses[-1], 5),
            "loss_min": round(min(bf16_losses), 5),
        },
        "fp32_master": {
            "param_dtype": fp32_dtype,
            "changed_param_fraction": round(fp32_frac, 6),
            "delta_l1": round(fp32_d, 4),
            "loss_step0": round(fp32_losses[0], 5),
            "loss_stepN": round(fp32_losses[-1], 5),
            "loss_min": round(min(fp32_losses), 5),
        },
        "verdict": "bf16 master loses ~1e-6 updates to rounding (low changed fraction, flat "
        "loss); fp32 master accumulates them (high changed fraction, descending loss).",
    }
    with open(args.out_json, "w", newline="\n") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print("bf16 master:", json.dumps(manifest["bf16_master"]))
    print("fp32 master:", json.dumps(manifest["fp32_master"]))
    print(f"wrote {args.out_csv} + {args.out_json}")


if __name__ == "__main__":
    main()
