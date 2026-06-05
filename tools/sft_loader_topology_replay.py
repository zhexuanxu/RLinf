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

"""R32 topology-faithful replay at the CORRECT effective batch (32).

R32's composition analysis (docs/evidence/r32_loader_topology.json) proved that at the
production topology (8 ranks x num_workers=8) BOTH loaders are RANK-REPLICATED: all 8 ranks
emit the same 32-frame micro-batch, so the effective per-step batch is 32 (not the configured
global 256). The effective per-step batch is therefore each loader's rank-0 32-frame stream.

This replay drives RLinf's Pi0 through the SAME loop (fp32 weights + autocast bf16,
AdamW(0.9,0.95)+clip(1.0)+openpi_cosine warmup LR) for 50 steps at batch 32 on each loader's
rank-0 first-50 stream (dumped at the production topology by tools/_loader_topology_dump.py
--with-images), from identical base weights with the SAME fixed shared noise/time -- so the
only variable is the loader. It compares each trajectory to the committed production curves
r24 (reference) / r22 (RLinf).

Single-GPU; GPU-only; output under /mnt/public/xzxuan/tmp (set TMPDIR).

Usage:
    TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
    python tools/sft_loader_topology_replay.py --out docs/evidence/r32_topology_replay.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os

import numpy as np
import torch

_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_RLINF_NPZ = "/mnt/public/xzxuan/tmp/rlinf_rank0_batches_img.npz"
_REF_NPZ = "/mnt/public/xzxuan/tmp/ref_rank0_batches_img.npz"
_R24_CSV = "docs/evidence/r24_ref_seed42_first50.csv"
_R22_CSV = "docs/evidence/r22_reduce_dtype_first50_losses.csv"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_N_STEPS = 50
_BATCH = 32
_SEED = 1234
_BETAS = (0.9, 0.95)
_EPS = 1e-8
_WD = 1e-10
_CLIP = 1.0
_PEAK_LR = 2.5e-5
_WARMUP = 1000


def _file_digest(path):
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _lr_at(step):
    init_lr = _PEAK_LR / (_WARMUP + 1)
    return init_lr + (_PEAK_LR - init_lr) * step / _WARMUP


def _curve(path, col):
    with open(path) as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def _build_obs(flat, step, device):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    return Observation.from_dict(
        {
            "image": {k: torch.from_numpy(flat[f"image__{k}"][step]).to(device, torch.float32) for k in _IMG},
            "image_mask": {k: torch.from_numpy(flat[f"image_mask__{k}"][step]).to(device) for k in _IMG},
            "state": torch.from_numpy(flat["state"][step]).to(device, torch.float32),
            "tokenized_prompt": torch.from_numpy(flat["tokenized_prompt"][step]).to(device).long(),
            "tokenized_prompt_mask": torch.from_numpy(flat["tokenized_prompt_mask"][step]).to(device).bool(),
        }
    )


def _rlinf_model(device):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model = Pi0(Pi0Config(pi05=True, action_horizon=32)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model.train()


def _replay(model, base_cpu, npz_path, noise, time, device):
    flat = np.load(npz_path)
    with torch.no_grad():
        for p, b in zip(model.parameters(), base_cpu):
            p.data.copy_(b.to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=_PEAK_LR, betas=_BETAS, eps=_EPS, weight_decay=_WD, foreach=False)
    rows = []
    for step in range(_N_STEPS):
        lr = _lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        obs = _build_obs(flat, step, device)
        act = torch.from_numpy(flat["actions"][step]).to(device, torch.float32)
        nz = torch.from_numpy(noise[step]).to(device)
        tm = torch.from_numpy(time[step]).to(device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_CLIP)
        opt.step()
        rows.append({"step": step, "loss": round(float(loss), 6), "grad_norm": round(float(gn), 5)})
        print(f"  step {step}: loss={float(loss):.5f} grad_norm={float(gn):.4f}", flush=True)
    return rows


def _track(rows, curve, k=10):
    rl = [r["loss"] for r in rows]
    last_rl, last_cv = float(np.mean(rl[-k:])), float(np.mean(curve[-k:]))
    max_abs = float(np.max([abs(a - b) for a, b in zip(rl, curve)]))
    return {
        f"replay_last{k}_mean": round(last_rl, 6),
        f"curve_last{k}_mean": round(last_cv, 6),
        "abs_last_mean_diff": round(abs(last_rl - last_cv), 6),
        "max_abs_per_step_diff": round(max_abs, 6),
        "tracks": abs(last_rl - last_cv) < 0.03,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp/r32_topology_replay.json")
    args = ap.parse_args()

    assert os.path.exists(_RLINF_NPZ) and os.path.exists(_REF_NPZ), "topology rank-0 dumps missing"
    rs = np.random.RandomState(_SEED)
    noise = rs.randn(_N_STEPS, _BATCH, 32, 32).astype(np.float32)
    time = rs.beta(1.5, 1.0, size=(_N_STEPS, _BATCH)).astype(np.float32) * 0.999 + 0.001

    device = "cuda"
    model = _rlinf_model(device)
    base_cpu = [p.detach().cpu().clone() for p in model.parameters()]

    print("Replaying the REFERENCE loader rank-0 batch-32 stream ...", flush=True)
    ref_rows = _replay(model, base_cpu, _REF_NPZ, noise, time, device)
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    print("Replaying the RLINF loader rank-0 batch-32 stream ...", flush=True)
    rlinf_rows = _replay(model, base_cpu, _RLINF_NPZ, noise, time, device)

    r24, r22 = _curve(_R24_CSV, "loss"), _curve(_R22_CSV, "rlinf_loss")
    ref_vs_r24 = _track(ref_rows, r24)
    rlinf_vs_r22 = _track(rlinf_rows, r22)
    rows = [
        {
            "step": i,
            "ref_loader_loss": ref_rows[i]["loss"],
            "rlinf_loader_loss": rlinf_rows[i]["loss"],
            "r24_ref_prod": round(r24[i], 6),
            "r22_rlinf_prod": round(r22[i], 6),
        }
        for i in range(_N_STEPS)
    ]
    rlinf_faster = rlinf_vs_r22["replay_last10_mean"] < ref_vs_r24["replay_last10_mean"] - 0.005
    result = {
        "purpose": "R32 topology-faithful replay at the CORRECT effective batch 32 (both production "
        "loaders are rank-replicated per docs/evidence/r32_loader_topology.json). RLinf's Pi0 runs the "
        "SAME loop + shared fixed noise/time on the reference vs RLinf rank-0 first-50 batch-32 streams "
        "(dumped at the production 8-rank/num_workers=8 topology); only the loader differs.",
        "provenance": {
            "command": "TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 "
            "PYTHONPATH=/mnt/public/xzxuan/repos/RLinf_pi05 python tools/sft_loader_topology_replay.py "
            "--out docs/evidence/r32_topology_replay.json",
            "repo_cwd": os.getcwd(),
            "base_weights": _WEIGHTS,
            "base_weights_sha256_16": _file_digest(_WEIGHTS),
            "rlinf_rank0_npz": _RLINF_NPZ,
            "rlinf_rank0_npz_sha256_16": _file_digest(_RLINF_NPZ),
            "ref_rank0_npz": _REF_NPZ,
            "ref_rank0_npz_sha256_16": _file_digest(_REF_NPZ),
            "n_steps": _N_STEPS,
            "batch": _BATCH,
            "noise_time_seed": _SEED,
            "topology": "rank-0 streams from the 8-rank num_workers=8 dump (both loaders rank-replicated "
            "-> rank-0 stream IS the effective per-step batch). See docs/evidence/r32_loader_topology.json.",
            "loop": "fp32 weights + autocast bf16; AdamW(0.9,0.95)/eps1e-8/wd1e-10; clip_grad_norm_(1.0); "
            "openpi_cosine warmup peak=2.5e-5 warmup=1000",
            "output": args.out,
        },
        "reference_loader_vs_r24_production": ref_vs_r24,
        "rlinf_loader_vs_r22_production": rlinf_vs_r22,
        "steps": rows,
        "verdict": (
            "At the topology-faithful effective batch 32, through the IDENTICAL loop + shared noise/time "
            f"(only the loader differs): the reference-loader rank-0 stream last-10 mean is "
            f"{ref_vs_r24['replay_last10_mean']:.4f} (r24 {ref_vs_r24['curve_last10_mean']:.4f}, "
            f"tracks={ref_vs_r24['tracks']}); the RLinf-loader rank-0 stream is "
            f"{rlinf_vs_r22['replay_last10_mean']:.4f} (r22 {rlinf_vs_r22['curve_last10_mean']:.4f}, "
            f"tracks={rlinf_vs_r22['tracks']}); RLinf-loader "
            + ("descends FASTER than the reference-loader" if rlinf_faster else "does NOT descend faster than the reference-loader")
            + ". Combined with R26->R29 (RLinf == reference on identical inputs) and the R32 topology "
            "analysis (both loaders rank-replicated, effective batch 32), the first-50 difference is "
            "the per-step batch COMPOSITION each loader's rank-0 streams. Caveats: aggregate (last-10) "
            "tracking, not step-by-step; fixed shared noise/time (vs production internal) and "
            "single-GPU/autocast deviations. task15 NOT met without the original 50/50 gate or an "
            "accepted plan evolution."
        ),
    }
    with open(args.out, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"ref_vs_r24": ref_vs_r24, "rlinf_vs_r22": rlinf_vs_r22, "rlinf_faster": rlinf_faster}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
