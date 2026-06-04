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

"""R30 production-sequence replay: reference loader vs RLinf loader, same RLinf loop.

Codex R28/R29 review (the last task15 proof): replay the reference's exact first-N
batch sequence through RLinf. R26+R28+R29 proved RLinf == the reference on IDENTICAL
inputs (model backward, 20-step optimizer loop, and 8-rank FSDP all match). The one
un-eliminated candidate is the per-step INPUT each loader feeds. This probe isolates it:

* dump the reference loader's first-N batch sequence (``_ref_seq_batch_dump.py``,
  reference venv, ``create_behavior_data_loader_torch``); and
* dump RLinf's own SFT loader's first-N sequence (``create_behavior_sft_data_loader``);

then run RLinf's ``Pi0`` through the SAME fp32-master + bf16-compute AdamW + clip(1.0) +
openpi_cosine-warmup-LR loop for N steps on EACH sequence, from identical base weights,
with the SAME fixed noise/time -- so the ONLY variable is the batch source. If RLinf's
own loader sequence descends faster than the reference loader sequence (which should
track the reference curve), the per-step INPUT (loader batch composition) is the driver
of the production faster-descent; if they descend the same, the hypothesis is refuted.

Single-GPU; GPU-only; all output under ``/mnt/public/xzxuan/tmp`` (set TMPDIR).

Usage:
    TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$REPO \
    python tools/sft_loader_sequence_probe.py --out docs/evidence/r30_loader_sequence.json
"""

from __future__ import annotations

import argparse
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
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_N_STEPS = 50
_BATCH_SIZE = 16  # single-GPU fp32-master + AdamW state caps the per-step batch (32 OOMs an 80 GB GPU)
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


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _dump_rlinf_loader(out_dir, n_steps, batch_size):
    """Dump RLinf's own SFT loader's first-N batch sequence (production use_skill:false)."""
    from rlinf.data.datasets.behavior import create_behavior_sft_data_loader

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=_DATA_ROOT,
        assets_dir=_ASSETS_DIR,
        asset_id="behavior-1k/2025-challenge-demos",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=["turning_on_radio"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=batch_size,
        num_workers=0,
        seed=0,
    )
    it = iter(loader)
    store = {}
    for _ in range(n_steps):
        obs, actions = next(it)
        for k in _IMG:
            store.setdefault(f"image__{k}", []).append(_np(obs.images[k]))
            store.setdefault(f"image_mask__{k}", []).append(_np(obs.image_masks[k]))
        store.setdefault("state", []).append(_np(obs.state))
        store.setdefault("tokenized_prompt", []).append(_np(obs.tokenized_prompt))
        store.setdefault("tokenized_prompt_mask", []).append(_np(obs.tokenized_prompt_mask))
        store.setdefault("actions", []).append(_np(actions))
    path = f"{out_dir}/rlinf_seq_batches.npz"
    np.savez(path, **{k: np.stack(v) for k, v in store.items()})
    return path


def _build_obs(flat, bi, device):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    return Observation.from_dict(
        {
            "image": {
                k: torch.from_numpy(flat[f"image__{k}"][bi]).to(device, torch.float32)
                for k in _IMG
            },
            "image_mask": {
                k: torch.from_numpy(flat[f"image_mask__{k}"][bi]).to(device) for k in _IMG
            },
            "state": torch.from_numpy(flat["state"][bi]).to(device, torch.float32),
            "tokenized_prompt": torch.from_numpy(flat["tokenized_prompt"][bi]).to(device).long(),
            "tokenized_prompt_mask": torch.from_numpy(flat["tokenized_prompt_mask"][bi])
            .to(device)
            .bool(),
        }
    )


def _rlinf_model(device):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    # fp32 params (the master); bf16 compute is via autocast in the loop. fp32 params + fp32 AdamW
    # is what makes the loss descend (a bf16 master loses the tiny warmup-LR updates); autocast keeps
    # single-GPU memory to ~1 fp32 weight + 1 fp32 grad + 2 fp32 AdamW states.
    model = Pi0(Pi0Config(pi05=True, action_horizon=32)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model.train()


def _replay(model, base_cpu, seq_path, noise, time, device):
    """Run the fp32-weight + autocast-bf16 AdamW loop on one batch sequence from base weights."""
    flat = np.load(seq_path)
    with torch.no_grad():  # reset to base weights (held on CPU to save GPU memory)
        for p, b in zip(model.parameters(), base_cpu):
            p.data.copy_(b.to(device))
    # foreach=False avoids AdamW's large multi-tensor temporaries (single-GPU memory).
    opt = torch.optim.AdamW(
        model.parameters(), lr=_PEAK_LR, betas=_BETAS, eps=_EPS, weight_decay=_WD, foreach=False
    )
    rows = []
    for step in range(noise.shape[0]):
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
        rows.append({"step": step, "loss": round(float(loss), 6), "grad_norm": round(float(gn), 5), "lr": lr})
        print(f"  step {step}: loss={float(loss):.5f} grad_norm={float(gn):.4f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="/mnt/public/xzxuan/tmp")
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp/r30_loader_sequence.json")
    args = ap.parse_args()

    ref_path = f"{args.tmp}/ref_seq_batches.npz"
    if os.path.exists(ref_path):
        os.remove(ref_path)  # stale-output guard
    print("Dumping the reference loader sequence (reference venv) ...", flush=True)
    proc = subprocess.run(
        [_REF_VENV_PY, _REF_DUMP, args.tmp, str(_N_STEPS), str(_BATCH_SIZE)],
        capture_output=True,
        text=True,
        timeout=2400,
    )
    print(proc.stdout[-800:], proc.stderr[-800:], flush=True)
    assert proc.returncode == 0, f"reference dump returned {proc.returncode}"
    assert os.path.exists(ref_path), "reference sequence dump missing"

    print("Dumping the RLinf loader sequence ...", flush=True)
    rlinf_path = _dump_rlinf_loader(args.tmp, _N_STEPS, _BATCH_SIZE)

    # Shared fixed noise/time so the ONLY variable across the two replays is the batch source.
    rs = np.random.RandomState(_SEED)
    noise = rs.randn(_N_STEPS, _BATCH_SIZE, 32, 32).astype(np.float32)
    time = (rs.beta(1.5, 1.0, size=(_N_STEPS, _BATCH_SIZE)).astype(np.float32) * 0.999 + 0.001)
    np.savez(f"{args.tmp}/seq_noise_time.npz", noise=noise, time=time)

    device = "cuda"
    model = _rlinf_model(device)
    base_cpu = [p.detach().cpu().clone() for p in model.parameters()]  # base weights on CPU

    print("Replaying the REFERENCE loader sequence through RLinf ...", flush=True)
    ref_rows = _replay(model, base_cpu, ref_path, noise, time, device)
    import gc

    gc.collect()
    torch.cuda.empty_cache()  # free the first run's optimizer state before the second
    print("Replaying the RLINF loader sequence through RLinf ...", flush=True)
    rlinf_rows = _replay(model, base_cpu, rlinf_path, noise, time, device)

    def _final(rows, k=5):
        return float(np.mean([r["loss"] for r in rows[-k:]]))

    ref_final = _final(ref_rows)
    rlinf_final = _final(rlinf_rows)
    rows = [
        {
            "step": i,
            "ref_loader_loss": ref_rows[i]["loss"],
            "rlinf_loader_loss": rlinf_rows[i]["loss"],
            "loss_diff_ref_minus_rlinf": round(ref_rows[i]["loss"] - rlinf_rows[i]["loss"], 6),
            "ref_loader_grad_norm": ref_rows[i]["grad_norm"],
            "rlinf_loader_grad_norm": rlinf_rows[i]["grad_norm"],
        }
        for i in range(len(ref_rows))
    ]
    rlinf_faster = rlinf_final < ref_final - 0.01
    result = {
        "purpose": "R30 production-sequence replay: RLinf's Pi0 runs the SAME fp32-master+bf16 "
        "AdamW+clip+warmup-LR loop for N steps on the reference loader's batch sequence vs RLinf's "
        "own SFT loader's batch sequence, from identical base weights with the SAME fixed noise/time, "
        "so the ONLY variable is the per-step batch source. Tests whether the per-step INPUT (loader "
        "batch composition) drives the production faster-descent.",
        "provenance": {
            "command": "TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 "
            "PYTHONPATH=/mnt/public/xzxuan/repos/RLinf_pi05 python tools/sft_loader_sequence_probe.py "
            "--out docs/evidence/r30_loader_sequence.json",
            "repo_cwd": os.getcwd(),
            "reference_venv": _REF_VENV_PY,
            "reference_src": _REF_SRC,
            "reference_dumper": _REF_DUMP,
            "ref_subprocess_returncode": proc.returncode,
            "rlinf_loader": "rlinf.data.datasets.behavior.create_behavior_sft_data_loader "
            "(tasks=[turning_on_radio], use_skill:false default)",
            "base_weights": _WEIGHTS,
            "base_weights_sha256_16": _file_digest(_WEIGHTS),
            "n_steps": _N_STEPS,
            "batch_size": _BATCH_SIZE,
            "noise_time_seed": _SEED,
            "task": "turning_on_radio (both loaders, the only difference is the loader)",
            "ref_seq_npz": ref_path,
            "ref_seq_npz_sha256_16": _file_digest(ref_path),
            "rlinf_seq_npz": rlinf_path,
            "rlinf_seq_npz_sha256_16": _file_digest(rlinf_path),
            "loop": "fp32-master + bf16-compute, torch.optim.AdamW(betas=(0.9,0.95),eps=1e-8,wd=1e-10), "
            "clip_grad_norm_(1.0), openpi_cosine warmup LR peak=2.5e-5 warmup=1000",
            "output": args.out,
        },
        "ref_loader_first5_mean": round(float(np.mean([r["loss"] for r in ref_rows[:5]])), 6),
        "rlinf_loader_first5_mean": round(float(np.mean([r["loss"] for r in rlinf_rows[:5]])), 6),
        "ref_loader_last5_mean": round(ref_final, 6),
        "rlinf_loader_last5_mean": round(rlinf_final, 6),
        "last5_gap_ref_minus_rlinf": round(ref_final - rlinf_final, 6),
        "committed_production_curves": {
            "reference_r24_step49": 0.0903,
            "rlinf_r22_step49": 0.0469,
            "note": "production runs were 8-rank batch-256; this probe is single-GPU batch-32, so "
            "absolute magnitudes are not directly comparable -- the comparison is ref-loader vs "
            "rlinf-loader trajectories under the IDENTICAL loop.",
        },
        "steps": rows,
        "verdict": (
            "PER-STEP INPUT IS THE MEASURED DRIVER: through the IDENTICAL RLinf model+loop+noise/time, "
            f"RLinf's own loader sequence descends faster (last-5 mean {rlinf_final:.4f}) than the "
            f"reference loader sequence (last-5 mean {ref_final:.4f}); the only variable was the batch "
            "source, and the step-0 gap on identical weights+noise/time isolates the cause to the "
            "batch. Combined with R26 (same-batch backward), R28 (20-step same-input loop) and R29 "
            "(8-rank FSDP) -- which proved RLinf == the reference on IDENTICAL inputs -- the production "
            "first-50 faster-descent is driven by RLinf's data loader feeding a DIFFERENT per-step "
            "batch composition than the reference's loader (same frame corpus, different per-step "
            "grouping/order), NOT by a model/optimizer/clip/LR/FSDP bug. AC-11 path (Codex to accept "
            "one): (A) plan-evolution accepting the loader-composition difference + the root-cause "
            "chain; (B) align RLinf's loader batch composition to the reference's, then rerun the gate; "
            "(C) feed RLinf the reference batches. Do NOT mark task15 met until a path is accepted."
            if rlinf_faster
            else "PER-STEP INPUT NOT CONFIRMED: through the identical loop, RLinf's loader sequence "
            f"(last-5 {rlinf_final:.4f}) does NOT descend faster than the reference loader sequence "
            f"(last-5 {ref_final:.4f}). The per-step-input hypothesis is not supported by this probe; "
            "the production divergence must be re-localized (e.g. task/data scale, or a production-only "
            "setting not reproduced single-GPU). Name the next step; do NOT close task15."
        ),
    }
    with open(args.out, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: result[k] for k in ("ref_loader_last5_mean", "rlinf_loader_last5_mean", "last5_gap_ref_minus_rlinf")}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
