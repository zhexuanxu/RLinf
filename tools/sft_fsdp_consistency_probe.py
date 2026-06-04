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

"""R29 8-rank-FSDP vs single-GPU same-input gradient consistency probe.

Codex R28 review (mainline): the R28 same-input replay was single-GPU and did NOT
exercise the 8-rank FSDP sharding/all-reduce -- the one production-topology piece left.
This probe closes it: on the SAME fixed global batch + fixed noise/time + base weights,
it runs RLinf's ``Pi0`` under (a) single-GPU and (b) 8-rank FSDP1 with the production
``MixedPrecision(param=bf16, reduce=fp32, buffer=fp32)`` + FULL_SHARD, and compares the
global (all-reduced) grad norm + post-step loss.

R26 proved RLinf's single-GPU backward == the reference; R28 proved RLinf's single-GPU
model+optimizer+clip+LR == the reference over 20 steps. So if the 8-rank FSDP all-reduced
grad norm == the single-GPU grad norm here, RLinf's FSDP distributed step is numerically
correct -> the production first-50 divergence is NOT the distributed step, leaving the
per-step INPUT (to be proven in R30 by replaying the reference's exact production
sequence). If they DIFFER, the FSDP step is the proven mechanism and is localized.

Steps: (1) run ``tests/unit_tests/_ref_model_grad_dump.py`` (reference venv) to dump the
fixed batch (``ref_grad_batches.npz`` = 4 x 8 = 32 real behavior frames); (2) torchrun
``tools/_fsdp_grad_worker.py`` with ``--nproc_per_node=1`` then ``=8`` on that batch;
(3) compare. Needs 8 GPUs; all output under ``/mnt/public/xzxuan/tmp`` (set TMPDIR; the
torch inductor ``.so``, if any, must live on ``/dev/shm``, not the network mount).

Usage:
    TMPDIR=/mnt/public/xzxuan/tmp PYTHONPATH=$REPO \
    python tools/sft_fsdp_consistency_probe.py --out docs/evidence/r29_fsdp_consistency.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

_REF_VENV_PY = "/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python"
_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed"
_REF_DUMP = "tests/unit_tests/_ref_model_grad_dump.py"
_WORKER = "tools/_fsdp_grad_worker.py"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_REPO = "/mnt/public/xzxuan/repos/RLinf_pi05"


def _file_digest(path):
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _run_worker(tmp, nproc):
    """torchrun the worker at the given world size; return its rank-0 JSON."""
    out_path = f"{tmp}/fsdp_grad_w{nproc}.json"
    if os.path.exists(out_path):
        os.remove(out_path)  # stale-output guard
    env = dict(os.environ)
    env.setdefault("TMPDIR", tmp)
    env["PYTHONPATH"] = _REPO
    proc = subprocess.run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={nproc}",
            _WORKER,
            tmp,
        ],
        capture_output=True,
        text=True,
        timeout=2400,
        env=env,
    )
    print(proc.stdout[-1200:], proc.stderr[-1200:], flush=True)
    assert proc.returncode == 0, f"torchrun nproc={nproc} returned {proc.returncode}"
    with open(out_path) as f:
        return json.load(f), proc.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", default="/mnt/public/xzxuan/tmp")
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp/r29_fsdp_consistency.json")
    args = ap.parse_args()

    # 1) dump the fixed batch (reference venv), with stale-output guard.
    batch_path = f"{args.tmp}/ref_grad_batches.npz"
    for p in (batch_path, f"{args.tmp}/ref_grad_dump.json"):
        if os.path.exists(p):
            os.remove(p)
    print("Dumping the fixed batch (reference venv) ...", flush=True)
    ref_proc = subprocess.run(
        [_REF_VENV_PY, _REF_DUMP, args.tmp], capture_output=True, text=True, timeout=2400
    )
    print(ref_proc.stdout[-500:], ref_proc.stderr[-500:], flush=True)
    assert ref_proc.returncode == 0, f"reference dump returned {ref_proc.returncode}"
    assert os.path.exists(batch_path), "reference batch dump missing"

    # 2) single-GPU then 8-rank FSDP on the SAME batch.
    print("Running single-GPU worker (nproc=1) ...", flush=True)
    w1, rc1 = _run_worker(args.tmp, 1)
    print("Running 8-rank FSDP worker (nproc=8) ...", flush=True)
    w8, rc8 = _run_worker(args.tmp, 8)

    gn_diff = abs(w1["global_grad_norm"] - w8["global_grad_norm"])
    rel_pct = round(100.0 * gn_diff / max(w1["global_grad_norm"], 1e-9), 3)
    loss_diff = abs(w1["loss"] - w8["loss"])
    # bf16 + sharded reduce: a few-percent agreement is the expected match band.
    matched = rel_pct < 2.0 and loss_diff < 0.01
    result = {
        "purpose": "R29 8-rank-FSDP vs single-GPU same-input gradient consistency: RLinf Pi0 on the "
        "IDENTICAL fixed 32-frame global batch + fixed noise/time + base weights, run single-GPU vs "
        "8-rank FSDP1 (production MixedPrecision param=bf16/reduce=fp32/buffer=fp32 + FULL_SHARD). "
        "Tests whether RLinf's FSDP sharding/all-reduce/clip is numerically equivalent to the "
        "single-GPU path (which R26+R28 proved equals the reference).",
        "provenance": {
            "command": "TMPDIR=/mnt/public/xzxuan/tmp PYTHONPATH=/mnt/public/xzxuan/repos/RLinf_pi05 "
            "python tools/sft_fsdp_consistency_probe.py --out docs/evidence/r29_fsdp_consistency.json",
            "repo_cwd": os.getcwd(),
            "reference_venv": _REF_VENV_PY,
            "reference_src": _REF_SRC,
            "reference_dumper": _REF_DUMP,
            "ref_subprocess_returncode": ref_proc.returncode,
            "worker": _WORKER,
            "worker_nproc1_returncode": rc1,
            "worker_nproc8_returncode": rc8,
            "base_weights": _WEIGHTS,
            "base_weights_sha256_16": _file_digest(_WEIGHTS),
            "batch_npz": batch_path,
            "batch_npz_sha256_16": _file_digest(batch_path),
            "noise_time_npz": f"{args.tmp}/ref_grad_noise_time.npz",
            "noise_time_npz_sha256_16": _file_digest(f"{args.tmp}/ref_grad_noise_time.npz"),
            "grad_norm_def": "global L2 norm via clip_grad_norm_ (FSDP1's model.clip_grad_norm_ on "
            "8 ranks -- the call train_pytorch_new.py:535 uses; torch.nn.utils on 1 rank). R27 "
            "confirmed RLinf get_grad_norm_for_mixed_precision computes the same global L2 norm.",
            "fsdp_wrap": "FSDP1 root unit, FULL_SHARD, use_orig_params=True (the all-reduce math is "
            "wrap-granularity-independent).",
            "output": args.out,
        },
        "single_gpu_nproc1": w1,
        "fsdp_8rank_nproc8": w8,
        "abs_global_grad_norm_diff": round(gn_diff, 5),
        "rel_grad_norm_diff_pct": rel_pct,
        "abs_loss_diff": round(loss_diff, 6),
        "verdict": (
            "FSDP CONSISTENT: RLinf's 8-rank FSDP all-reduced grad norm matches its single-GPU grad "
            f"norm on the identical 32-frame batch (rel diff {rel_pct}%, |Δloss|={loss_diff:.5f}). So "
            "RLinf's FSDP sharding/all-reduce/clip is numerically equivalent to the single-GPU path, "
            "which R26+R28 proved equals the reference -> RLinf's full production stack (8-rank FSDP) "
            "is equivalent to the reference on identical inputs, and the FSDP DISTRIBUTED STEP is "
            "RULED OUT as the cause of the production first-50 divergence. The remaining candidate is "
            "the per-step INPUT each loader/RNG feeds in production. NEXT (R30): replay the "
            "reference's EXACT production first-N batch/noise/time sequence through RLinf and check it "
            "tracks the reference (the decisive per-step-input proof), then decide the AC-11 path."
            if matched
            else "FSDP DIVERGES: RLinf's 8-rank FSDP grad norm differs from its single-GPU grad norm "
            f"on the identical batch (rel diff {rel_pct}%, |Δloss|={loss_diff:.5f}) -> the FSDP "
            "distributed step (sharding / all-reduce / clip / mixed-precision reduce) is a real "
            "mechanism of the production divergence. Localize (per-module grad, reduce dtype, clip) "
            "and fix, then rerun the first-50 gate."
        ),
        "caveat": "Single-GPU == reference is established by R26 (same-batch backward) + R28 (20-step "
        "same-input replay); this probe isolates ONLY the FSDP delta. It does not by itself prove the "
        "per-step-input driver -- that is the R30 production-sequence replay.",
    }
    with open(args.out, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"w1": w1, "w8": w8, "rel_pct": rel_pct, "matched": matched}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
