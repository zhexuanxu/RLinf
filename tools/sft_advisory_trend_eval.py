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

"""Advisory ~1-hour long-run loss-trend evidence (R38, DEC-1(c) ADVISORY tier).

The normal (unpinned) ``use_skill:false`` 8-GPU production SFT
(``train_vla_sft.py --config-name behavior_pi05_vla``) is run for a ~1-hour
window. This tool extracts the rank-averaged ``train/loss`` (and
``learning_rate`` / ``grad_norm``) per step from its tensorboard event file and
compares the long-run TREND/direction to the reference run's committed log
(``pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log``, whose clean ``Step N:
grad_norm=..., learning_rate=..., loss=...`` lines give the reference curve).

This is the ADVISORY tier of DEC-1 (a direction/shape match over the long run,
NOT a hard per-step gate and NEVER a CI gate). It records the descent direction,
first-N / last-N means for both curves, a Pearson correlation of the aligned
curves, the per-step within-band count (context only), and full provenance.

Writes ``docs/evidence/r38_advisory_1h_trend.{csv,json}``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess

_REF_STEP = re.compile(
    r"Step (\d+): grad_norm=([\d.eE+-]+), learning_rate=([\d.eE+-]+), loss=([\d.eE+-]+)"
)


def _file_digest(path: str) -> str:
    if not path or not os.path.exists(path):
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_rev(repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _ref_curve(ref_log: str) -> dict[int, tuple[float, float, float]]:
    """Parse the reference log's clean ``Step N:`` lines into {step: (loss, lr, grad_norm)}."""
    out = {}
    with open(ref_log, errors="ignore") as f:
        for line in f:
            for m in _REF_STEP.finditer(line):
                step = int(m.group(1))
                out[step] = (float(m.group(4)), float(m.group(3)), float(m.group(2)))
    return out


def _find_event_file(results_dir: str):
    cands = sorted(
        glob.glob(
            os.path.join(results_dir, "**", "events.out.tfevents.*"), recursive=True
        )
    )
    if not cands:
        raise FileNotFoundError(f"no tensorboard event file under {results_dir}")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    for path in cands:
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        if "train/loss" in ea.Tags().get("scalars", []):
            return path, ea
    raise ValueError(f"no event file under {results_dir} has a train/loss scalar")


def _scalars(ea, tag: str) -> dict[int, float]:
    if tag not in ea.Tags().get("scalars", []):
        return {}
    return {s.step: s.value for s in ea.Scalars(tag)}


def _mean(xs) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _pearson(a, b) -> float:
    n = len(a)
    if n < 2:
        return float("nan")
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a) ** 0.5
    db = sum((y - mb) ** 2 for y in b) ** 0.5
    return num / (da * db) if da and db else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument(
        "--ref-log",
        default="/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/"
        "pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log",
    )
    ap.add_argument(
        "--model",
        default="/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors",
    )
    ap.add_argument(
        "--norm-stats",
        default="/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets/"
        "behavior-1k/2025-challenge-demos/norm_stats.json",
    )
    ap.add_argument(
        "--tokenizer",
        default="/mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model",
    )
    ap.add_argument("--out-csv", default="docs/evidence/r38_advisory_1h_trend.csv")
    ap.add_argument("--out-json", default="docs/evidence/r38_advisory_1h_trend.json")
    ap.add_argument("--run-command", default="")
    ap.add_argument("--run-returncode", default="")
    args = ap.parse_args()

    event_path, ea = _find_event_file(args.results_dir)
    loss = _scalars(ea, "train/loss")
    lr = _scalars(ea, "train/learning_rate")
    gn = _scalars(ea, "train/grad_norm")
    steps = sorted(loss)
    if not steps:
        raise ValueError("no train/loss steps in the event file")

    ref = _ref_curve(args.ref_log)
    rows = []
    for step in steps:
        rl = float(loss[step])
        rf = ref.get(step, (float("nan"),))[0]
        rows.append(
            {
                "step": step,
                "rlinf_loss": round(rl, 6),
                "ref_loss": round(rf, 6) if rf == rf else "",
                "abs_delta": round(abs(rl - rf), 6) if rf == rf else "",
                "rlinf_lr": lr.get(step, float("nan")),
                "rlinf_grad_norm": round(gn.get(step, float("nan")), 5),
            }
        )

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    n = len(steps)
    head = min(10, n)
    tail = min(50, n)
    rl_series = [float(loss[s]) for s in steps]
    paired = [(float(loss[s]), ref[s][0]) for s in steps if s in ref]
    rl_paired = [a for a, _ in paired]
    rf_paired = [b for _, b in paired]
    within = sum(1 for a, b in paired if abs(a - b) <= 0.03)
    rl_first, rl_last = _mean(rl_series[:head]), _mean(rl_series[-tail:])
    rf_first = _mean([ref[s][0] for s in steps[:head] if s in ref])
    rf_last = _mean([ref[s][0] for s in steps[-tail:] if s in ref])
    corr = _pearson(rl_paired, rf_paired)
    both_descend = (rl_last < rl_first) and (rf_last < rf_first)

    if both_descend:
        verdict = (
            "ADVISORY TREND MATCH: over the ~1h window the unpinned RLinf use_skill:false loss descends with "
            "the same direction as the reference (RLinf first%d-mean %.4f -> last%d-mean %.4f; reference "
            "%.4f -> %.4f; aligned-curve Pearson r=%.3f). This is the DEC-1(c) advisory tier -- a "
            "direction/shape match over the long run, not a hard per-step gate (the two production runs use "
            "independent noise/time + shuffle, so per-step values differ; %d/%d aligned steps are within "
            "0.03, reported for context only)."
            % (
                head,
                rl_first,
                tail,
                rl_last,
                rf_first,
                rf_last,
                corr,
                within,
                len(paired),
            )
        )
    else:
        verdict = (
            "ADVISORY TREND MISMATCH: RLinf's long-run loss does not descend like the reference "
            "(RLinf %.4f -> %.4f; reference %.4f -> %.4f; r=%.3f). Investigate before relying on the "
            "long-run alignment." % (rl_first, rl_last, rf_first, rf_last, corr)
        )

    result = {
        "purpose": "R38 ADVISORY ~1h long-run loss-trend evidence (DEC-1(c) advisory tier, never a CI gate): "
        "the NORMAL unpinned use_skill:false 8-GPU production SFT (train_vla_sft.py --config-name "
        "behavior_pi05_vla) run for a ~1h window, with its per-step train/loss trend compared in "
        "DIRECTION/shape to the reference long-run log. This is an advisory direction match, not a hard "
        "per-step gate.",
        "provenance": {
            "run_command": args.run_command,
            "run_returncode": args.run_returncode,
            "rlinf_git_rev": _git_rev(os.getcwd()),
            "tensorboard_event_file": event_path,
            "loss_tag": "train/loss (rank-AVG)",
            "reference_log": args.ref_log,
            "reference_git_rev": _git_rev(
                "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed"
            ),
            "model": args.model,
            "model_sha256_16": _file_digest(args.model),
            "norm_stats": args.norm_stats,
            "norm_stats_sha256_16": _file_digest(args.norm_stats),
            "tokenizer": args.tokenizer,
            "tokenizer_sha256_16": _file_digest(args.tokenizer),
            "steps_run": n,
            "out_csv": args.out_csv,
        },
        "trend": {
            "rlinf_steps": n,
            "reference_steps_aligned": len(paired),
            "rlinf_first%d_mean" % head: round(rl_first, 6),
            "rlinf_last%d_mean" % tail: round(rl_last, 6),
            "reference_first%d_mean" % head: round(rf_first, 6),
            "reference_last%d_mean" % tail: round(rf_last, 6),
            "both_descend": both_descend,
            "pearson_correlation_aligned": round(corr, 5),
            "within_0.03_aligned": "%d/%d" % (within, len(paired)),
        },
        "verdict": verdict,
    }
    with open(args.out_json, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result["trend"], indent=2))
    print(f"wrote {args.out_csv} and {args.out_json}")


if __name__ == "__main__":
    main()
