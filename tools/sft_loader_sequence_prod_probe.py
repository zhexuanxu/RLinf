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

"""R31 PRODUCTION-FAITHFUL loader-sequence replay at GLOBAL batch 256.

Codex R30 review: R30's loader-sequence replay was reduced-scale (batch 16,
``num_workers=0``, ``seed=0``) and its reference-loader trajectory did not track the
reference production curve, so it cannot be accepted as the production proof. This
probe reruns it at the production recipe (verified: GLOBAL batch **256** = micro 32 x 8,
``turning_on_radio``, ``peak_lr=2.5e-5``, ``warmup=1000``, ``use_skill:false``; only the
loader differs) and compares each loader's trajectory to its committed production curve.

It drives RLinf's ``Pi0`` through the SAME loop (fp32 weights + ``autocast`` bf16,
``AdamW(0.9,0.95)``, ``clip_grad_norm_(1.0)``, openpi_cosine warmup LR) for 50 steps at
global batch 256 via **gradient accumulation** (8 chunks x 32; each chunk loss scaled by
32/256 so the accumulated grad is the 256-sample mean; the logged loss is the 256-sample
mean), with the SAME fixed shared noise/time, on:

* the **reference loader** first-50 global-256 sequence
  (``create_behavior_data_loader_torch`` at the r24 recipe, dumped to npz), and
* RLinf's own **SFT loader** first-50 global-256 sequence
  (``create_behavior_sft_data_loader`` with the production builder params --
  ``batch_size=micro_batch_size=32``, ``num_workers=8``, ``seed=42``, ``use_skill=False``,
  ``tasks=[turning_on_radio]`` -- iterated live, 8 micro-batches grouped per global step).
  ``build_behavior_sft_dataloader`` is a thin wrapper that extracts exactly these params
  from the composed ``examples/sft/config/behavior_pi05_vla.yaml`` and calls
  ``create_behavior_sft_data_loader``, so the two are equivalent.

ONLY if the reference-loader trajectory tracks r24 (~0.090) AND the RLinf-loader trajectory
tracks r22 (~0.046) may the per-step INPUT be claimed as the proven driver. Single-GPU;
GPU-only; all output under ``/mnt/public/xzxuan/tmp`` (set TMPDIR).

Usage:
    TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
    python tools/sft_loader_sequence_prod_probe.py --out docs/evidence/r31_loader_sequence_prod.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess

import numpy as np
import torch

_REF_VENV_PY = "/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python"
_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed"
_REF_DUMP = "tests/unit_tests/_ref_seq_batch_dump.py"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_R24_CSV = "docs/evidence/r24_ref_seed42_first50.csv"
_R22_CSV = "docs/evidence/r22_reduce_dtype_first50_losses.csv"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_N_STEPS = 50
_GLOBAL = 256
_MICRO = 32  # production micro_batch_size; also the gradient-accumulation chunk size
_CHUNKS = _GLOBAL // _MICRO  # 8
_SEED = 1234  # shared fixed noise/time
_PROD_SEED = 42  # production data seed (RLinf loader)
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


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _curve(path, col):
    with open(path) as f:
        return [float(r[col]) for r in csv.DictReader(f)]


def _obs_from_arrays(imgs, masks, state, tp, tpm, device):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    return Observation.from_dict(
        {
            "image": {k: imgs[k].to(device, torch.float32) for k in _IMG},
            "image_mask": {k: masks[k].to(device) for k in _IMG},
            "state": state.to(device, torch.float32),
            "tokenized_prompt": tp.to(device).long(),
            "tokenized_prompt_mask": tpm.to(device).bool(),
        }
    )


def _ref_step_chunks(flat, step, device):
    """8 (obs, act) chunks of 32 from the reference npz step's 256-frame global batch."""
    chunks = []
    for c in range(_CHUNKS):
        lo, hi = c * _MICRO, (c + 1) * _MICRO
        imgs = {k: torch.from_numpy(flat[f"image__{k}"][step][lo:hi]) for k in _IMG}
        masks = {k: torch.from_numpy(flat[f"image_mask__{k}"][step][lo:hi]) for k in _IMG}
        obs = _obs_from_arrays(
            imgs,
            masks,
            torch.from_numpy(flat["state"][step][lo:hi]),
            torch.from_numpy(flat["tokenized_prompt"][step][lo:hi]),
            torch.from_numpy(flat["tokenized_prompt_mask"][step][lo:hi]),
            device,
        )
        act = torch.from_numpy(flat["actions"][step][lo:hi]).to(device, torch.float32)
        chunks.append((obs, act))
    return chunks


def _rlinf_step_chunks(loader_iter, device):
    """8 (obs, act) chunks of 32 from RLinf's live loader (8 micro-batches per global step)."""

    def _to_dev(obs, act):
        o = _obs_from_arrays(
            {k: obs.images[k] for k in _IMG},
            {k: obs.image_masks[k] for k in _IMG},
            obs.state,
            obs.tokenized_prompt,
            obs.tokenized_prompt_mask,
            device,
        )
        return o, act.to(device, torch.float32)

    return [_to_dev(*next(loader_iter)) for _ in range(_CHUNKS)]


def _rlinf_model(device):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model = Pi0(Pi0Config(pi05=True, action_horizon=32)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model.train()


def _replay(model, base_cpu, step_chunks_fn, noise, time, device):
    """Global-256 gradient-accumulation loop (8 chunks x 32) from base weights."""
    with torch.no_grad():
        for p, b in zip(model.parameters(), base_cpu):
            p.data.copy_(b.to(device))
    opt = torch.optim.AdamW(
        model.parameters(), lr=_PEAK_LR, betas=_BETAS, eps=_EPS, weight_decay=_WD, foreach=False
    )
    rows = []
    for step in range(_N_STEPS):
        lr = _lr_at(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        chunks = step_chunks_fn(step)
        opt.zero_grad(set_to_none=True)
        step_loss = 0.0
        for c, (obs, act) in enumerate(chunks):
            lo, hi = c * _MICRO, (c + 1) * _MICRO
            nz = torch.from_numpy(noise[step][lo:hi]).to(device)
            tm = torch.from_numpy(time[step][lo:hi]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
            (loss * (_MICRO / _GLOBAL)).backward()  # accumulate -> 256-sample mean grad
            step_loss += float(loss) * (_MICRO / _GLOBAL)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_CLIP)
        opt.step()
        rows.append({"step": step, "loss": round(step_loss, 6), "grad_norm": round(float(gn), 5), "lr": lr})
        print(f"  step {step}: loss={step_loss:.5f} grad_norm={float(gn):.4f}", flush=True)
    return rows


def _track(rows, curve, k=10):
    """Compare a replay trajectory tail to a production curve tail (last-k mean + max |Δ|)."""
    rl = [r["loss"] for r in rows]
    last_rl = float(np.mean(rl[-k:]))
    last_cv = float(np.mean(curve[-k:]))
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
    ap.add_argument("--tmp", default="/mnt/public/xzxuan/tmp")
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp/r31_loader_sequence_prod.json")
    ap.add_argument("--redump", action="store_true", help="force re-dump the reference sequence")
    args = ap.parse_args()

    ref_path = f"{args.tmp}/ref_seq_batches.npz"
    ref_rc = "reused"
    # Reuse an existing complete dump (>1 GB) to skip the ~13-min single-worker re-dump;
    # otherwise (re)dump, removing any small/partial stale file first.
    if os.path.exists(ref_path) and os.path.getsize(ref_path) > 1 << 30 and not args.redump:
        print(f"Reusing existing reference sequence dump ({os.path.getsize(ref_path)} bytes) ...", flush=True)
    else:
        if os.path.exists(ref_path):
            os.remove(ref_path)  # stale-output guard
        print("Dumping the reference loader GLOBAL-256 sequence (reference venv) ...", flush=True)
        proc = subprocess.run(
            [_REF_VENV_PY, _REF_DUMP, args.tmp, str(_N_STEPS), str(_GLOBAL)],
            capture_output=True,
            text=True,
            timeout=3600,
        )
        print(proc.stdout[-800:], proc.stderr[-800:], flush=True)
        assert proc.returncode == 0, f"reference dump returned {proc.returncode}"
        assert os.path.exists(ref_path), "reference sequence dump missing"
        ref_rc = proc.returncode

    # Shared fixed noise/time so the ONLY variable across the two replays is the loader.
    rs = np.random.RandomState(_SEED)
    noise = rs.randn(_N_STEPS, _GLOBAL, 32, 32).astype(np.float32)
    time = rs.beta(1.5, 1.0, size=(_N_STEPS, _GLOBAL)).astype(np.float32) * 0.999 + 0.001

    device = "cuda"
    model = _rlinf_model(device)
    base_cpu = [p.detach().cpu().clone() for p in model.parameters()]

    print("Loading the reference sequence into RAM ...", flush=True)
    # NpzFile re-reads the full array on every key access; materialize once into RAM
    # (the host has ~2 TB) so per-step slicing is cheap.
    with np.load(ref_path) as _npz:
        ref_flat = {k: _npz[k] for k in _npz.files}
    print("Replaying the REFERENCE loader GLOBAL-256 sequence ...", flush=True)
    ref_rows = _replay(model, base_cpu, lambda s: _ref_step_chunks(ref_flat, s, device), noise, time, device)

    import gc

    gc.collect()
    torch.cuda.empty_cache()

    print("Replaying the RLINF loader GLOBAL-256 sequence (live production loader) ...", flush=True)
    from rlinf.data.datasets.behavior import create_behavior_sft_data_loader

    rlinf_loader = create_behavior_sft_data_loader(
        behavior_dataset_root=_DATA_ROOT,
        assets_dir=_ASSETS_DIR,
        asset_id="behavior-1k/2025-challenge-demos",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=["turning_on_radio"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=_MICRO,  # production micro_batch_size=32
        num_workers=8,  # production worker count
        seed=_PROD_SEED,  # production data seed=42
        use_skill=False,
    )
    rlinf_iter = iter(rlinf_loader)
    rlinf_rows = _replay(
        model, base_cpu, lambda s: _rlinf_step_chunks(rlinf_iter, device), noise, time, device
    )

    r24 = _curve(_R24_CSV, "loss")
    r22 = _curve(_R22_CSV, "rlinf_loss")
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
    proven = (
        ref_vs_r24["tracks"]
        and rlinf_vs_r22["tracks"]
        and rlinf_vs_r22["replay_last10_mean"] < ref_vs_r24["replay_last10_mean"] - 0.01
    )
    result = {
        "purpose": "R31 PRODUCTION-FAITHFUL loader-sequence replay at GLOBAL batch 256: RLinf's Pi0 "
        "runs the SAME loop + shared fixed noise/time on the reference loader's vs RLinf's own "
        "production SFT loader's first-50 global-256 sequences (only the loader differs), compared to "
        "the committed reference (r24) and RLinf (r22) production curves.",
        "provenance": {
            "command": "TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 "
            "PYTHONPATH=/mnt/public/xzxuan/repos/RLinf_pi05 python tools/sft_loader_sequence_prod_probe.py "
            "--out docs/evidence/r31_loader_sequence_prod.json",
            "repo_cwd": os.getcwd(),
            "reference_venv": _REF_VENV_PY,
            "reference_src": _REF_SRC,
            "reference_dumper": _REF_DUMP,
            "ref_subprocess_returncode": ref_rc,
            "rlinf_loader": "rlinf.data.datasets.behavior.create_behavior_sft_data_loader "
            "(batch_size=micro_batch_size=32, num_workers=8, seed=42, use_skill=False, "
            "tasks=[turning_on_radio]); == build_behavior_sft_dataloader from "
            "examples/sft/config/behavior_pi05_vla.yaml (thin wrapper extracting these params).",
            "global_batch": _GLOBAL,
            "micro_chunk": _MICRO,
            "n_chunks_per_step": _CHUNKS,
            "grad_accumulation": f"{_CHUNKS} chunks x {_MICRO}; chunk loss scaled by {_MICRO}/{_GLOBAL}; "
            "accumulated grad = 256-sample mean; logged loss = 256-sample mean.",
            "base_weights": _WEIGHTS,
            "base_weights_sha256_16": _file_digest(_WEIGHTS),
            "n_steps": _N_STEPS,
            "noise_time_seed": _SEED,
            "ref_seq_npz": ref_path,
            "ref_seq_npz_sha256_16": _file_digest(ref_path),
            "loop": "fp32 weights + autocast bf16; torch.optim.AdamW(0.9,0.95)/eps1e-8/wd1e-10; "
            "clip_grad_norm_(1.0); openpi_cosine warmup peak=2.5e-5 warmup=1000",
            "deviations_from_production": "fixed shared noise/time (vs internal sampling; ~0.004 per "
            "R24); fp32+autocast (vs FSDP MixedPrecision; ~0.2% per R29); single-GPU rank-independent "
            "global-256 via accumulation (the union of the 8 ranks' micro-batches is the same 256 "
            "frames; R19 set-identity).",
            "output": args.out,
        },
        "reference_loader_vs_r24_production": ref_vs_r24,
        "rlinf_loader_vs_r22_production": rlinf_vs_r22,
        "steps": rows,
        "verdict": (
            "PRODUCTION-FAITHFUL PROOF: at global batch 256, the reference-loader replay tracks the "
            f"reference production curve r24 (replay last-10 {ref_vs_r24['replay_last10_mean']:.4f} vs "
            f"r24 {ref_vs_r24['curve_last10_mean']:.4f}) and the RLinf-loader replay tracks the RLinf "
            f"production curve r22 (replay {rlinf_vs_r22['replay_last10_mean']:.4f} vs r22 "
            f"{rlinf_vs_r22['curve_last10_mean']:.4f}) and is faster -- through the IDENTICAL loop + "
            "noise/time, the only variable being the loader. So the production first-50 faster-descent "
            "is PROVEN to be RLinf's data loader feeding a different per-step batch composition than the "
            "reference's, NOT a model/optimizer/clip/LR/FSDP bug (R26->R29 already proved RLinf == the "
            "reference on identical inputs). AC-11 plan-evolution can now be proposed."
            if proven
            else "NOT YET A PROOF: at global batch 256 the replay trajectories do NOT both track their "
            f"production curves (reference-loader last-10 {ref_vs_r24['replay_last10_mean']:.4f} vs r24 "
            f"{ref_vs_r24['curve_last10_mean']:.4f}, tracks={ref_vs_r24['tracks']}; RLinf-loader "
            f"{rlinf_vs_r22['replay_last10_mean']:.4f} vs r22 {rlinf_vs_r22['curve_last10_mean']:.4f}, "
            f"tracks={rlinf_vs_r22['tracks']}). The loader-difference remains a hypothesis; state the "
            "residual gap and the next localization step. Do NOT mark task15 met."
        ),
    }
    with open(args.out, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"ref_vs_r24": ref_vs_r24, "rlinf_vs_r22": rlinf_vs_r22, "proven": proven}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
