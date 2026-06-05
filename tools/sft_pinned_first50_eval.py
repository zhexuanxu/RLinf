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

"""Evaluate the actual-stack PINNED first-50 SFT run against the reference (R37).

The pinned 8-GPU SFT run (``train_vla_sft.py`` with ``data.pinned_inputs_npz`` +
``data.pinned_noise_time_npz`` set) replays the reference rank-0-fanout first-50
global-256 batch sequence + the SHARED noise/time through the REAL FSDP training
stack. This tool extracts the rank-averaged ``train/loss`` (and ``learning_rate``
/ ``grad_norm``) from its tensorboard event file and compares each step to the
reference model's loss on the IDENTICAL pinned inputs (the reference arm of the
R36 experiment, ``ref_pinned_dump.json``) under the original DEC-1(b)
``|Δ| <= 0.03`` gate. It also records the reference PRODUCTION first-50 curve
(``r16_first50_losses.csv`` ``ref_loss``) for curve-shape context.

If the pinned run is 50/50 within 0.03 of the reference-pinned loss, the actual
SFT stack reproduces the reference on identical inputs and task15 closes under
the original gate.

Writes ``docs/evidence/r37_pinned_first50.{csv,json}``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess


def _file_digest(path):
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_rev(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _find_event_file(results_dir):
    cands = sorted(
        glob.glob(
            os.path.join(results_dir, "**", "events.out.tfevents.*"), recursive=True
        )
    )
    if not cands:
        raise FileNotFoundError(f"no tensorboard event file under {results_dir}")
    # Prefer the event file that actually carries train/loss.
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for path in cands:
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        if "train/loss" in ea.Tags().get("scalars", []):
            return path, ea
    raise ValueError(f"no event file under {results_dir} has a train/loss scalar")


def _scalars(ea, tag, n_steps):
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {s.step: s.value for s in ea.Scalars(tag)}


def _ref_prod_losses(csv_path):
    out = {}
    if not csv_path or not os.path.exists(csv_path):
        return out
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            out[int(row["step"])] = float(row["ref_loss"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--ref-dump", default="/mnt/public/xzxuan/tmp/ref_pinned_dump.json")
    ap.add_argument("--r36-json", default="docs/evidence/r36_pinned_residual.json")
    ap.add_argument("--ref-prod-csv", default="docs/evidence/r16_first50_losses.csv")
    ap.add_argument("--out-csv", default="docs/evidence/r37_pinned_first50.csv")
    ap.add_argument("--out-json", default="docs/evidence/r37_pinned_first50.json")
    ap.add_argument("--run-command", default="")
    ap.add_argument("--run-returncode", default="")
    ap.add_argument("--n-steps", type=int, default=50)
    args = ap.parse_args()

    n = args.n_steps
    event_path, ea = _find_event_file(args.results_dir)
    loss = _scalars(ea, "train/loss", n)
    lr = _scalars(ea, "train/learning_rate", n)
    gn = _scalars(ea, "train/grad_norm", n)
    steps = sorted(loss)[:n]
    if len(steps) < n:
        raise ValueError(f"only {len(steps)} train/loss steps found, need {n}")

    with open(args.ref_dump) as f:
        ref = json.load(f)
    ref_pinned = {row["step"]: row["loss"] for row in ref["steps"]}
    ref_prod = _ref_prod_losses(args.ref_prod_csv)
    r36 = {}
    if os.path.exists(args.r36_json):
        with open(args.r36_json) as f:
            r36 = json.load(f)
    r36_prov = r36.get("provenance", {})

    rows = []
    for i, step in enumerate(steps):
        rl = float(loss[step])
        rp = float(ref_pinned[i])
        rows.append(
            {
                "step": i,
                "rlinf_pinned_loss": round(rl, 6),
                "ref_pinned_loss": round(rp, 6),
                "abs_delta": round(abs(rl - rp), 6),
                "within_0.03": int(abs(rl - rp) <= 0.03),
                "ref_prod_loss": round(ref_prod.get(i, float("nan")), 6),
                "rlinf_lr": lr.get(step, float("nan")),
                "rlinf_grad_norm": round(gn.get(step, float("nan")), 5),
            }
        )
    within = sum(r["within_0.03"] for r in rows)
    maxd = max(r["abs_delta"] for r in rows)
    passed = within == n

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    ref_meta = ref.get("meta", {})
    result = {
        "purpose": "R37 actual-stack PINNED first-50 SFT: the real 8-GPU FSDP train_vla_sft.py replays the "
        "reference rank-0-fanout first-50 global-256 batches + SHARED noise/time (data.pinned_inputs_npz + "
        "data.pinned_noise_time_npz) and is compared per-step to the reference model's loss on the IDENTICAL "
        "pinned inputs under the original DEC-1(b) |delta|<=0.03 gate. 50/50 => the actual SFT stack "
        "reproduces the reference first-50 curve on identical inputs (task15 closes under the original gate).",
        "provenance": {
            "run_command": args.run_command,
            "run_returncode": args.run_returncode,
            "rlinf_git_rev": _git_rev(os.getcwd()),
            "tensorboard_event_file": event_path,
            "loss_tag": "train/loss (rank-AVG)",
            "ref_pinned_dump": args.ref_dump,
            "ref_pinned_arm_returncode": ref_meta.get("returncode"),
            "ref_pinned_command": ref_meta.get("ref_command"),
            "ref_git_rev": ref_meta.get("ref_git_rev"),
            "pinned_batches_npz": ref_meta.get("batches_npz"),
            "pinned_batches_npz_sha256_16": _file_digest(
                ref_meta.get("batches_npz", "")
            ),
            "pinned_noise_time_npz": ref_meta.get("noise_time_npz"),
            "pinned_noise_time_npz_sha256_16": _file_digest(
                ref_meta.get("noise_time_npz", "")
            ),
            "r36_inputs_identical_all_steps": r36.get("inputs_identical_all_steps"),
            "r36_base_weights_sha256_16": r36_prov.get("base_weights_sha256_16"),
            "out_csv": args.out_csv,
        },
        "within_0.03": "%d/%d" % (within, n),
        "max_abs_delta": round(maxd, 6),
        "gate_passed_50_50": passed,
        "steps": rows,
        "verdict": (
            "PINNED ACTUAL-STACK MATCH (task15 closes under the original gate): the real 8-GPU FSDP SFT run, "
            "fed the reference's IDENTICAL rank-0-fanout first-50 batches + shared noise/time, matches the "
            f"reference model's per-step loss {within}/{n} within 0.03 (max |delta|={maxd:.4f}). So the actual "
            "SFT training stack reproduces the reference first-50 curve on identical inputs; the R35 production "
            "40/50 residual was purely the independent noise/time-RNG + shuffle, now removed."
            if passed
            else "PINNED ACTUAL-STACK RESIDUAL REMAINS: the real 8-GPU FSDP SFT run matches the reference-pinned "
            f"loss only {within}/{n} (max |delta|={maxd:.4f}) on identical inputs -- a residual remains in the "
            "production stack (precision/FSDP/order); localize it. Do NOT close task15."
        ),
    }
    with open(args.out_json, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({"within_0.03": within, "max_abs_delta": maxd, "passed": passed}))
    print(f"wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
