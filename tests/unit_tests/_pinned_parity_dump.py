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

"""Generate the pinned first-N training-step parity evidence.

The reference dumper (`_ref_pinned_run.py`, reference venv GPU) replays the
reference rank-0-fanout first-N global-256 batches + a SHARED noise/time and
dumps the reference model's per-step loss + the batches/noise-time npz. The real
8-GPU FSDP `train_vla_sft.py` then replays the IDENTICAL pinned inputs under the
current canonical config (`+data.pinned_inputs_npz` / `+data.pinned_noise_time_npz`)
and logs its per-step loss to tensorboard. This reads both and writes the per-step
comparison under the original `|Δ|≤0.03` gate -- a re-confirmation that the FSDP /
optimizer / forward stack reproduces the reference per-step on identical inputs.

Run from the repo root (after both runs)::

    python tests/unit_tests/_pinned_parity_dump.py \
        --rlinf-tb-dir <log_path>/tensorboard \
        --ref-dump <out>/ref_pinned_dump.json \
        --git-rev <sha> --out docs/evidence/phase4_pinned_first50_parity.json
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import pathlib

_GATE = 0.03


def _tb_scalars(tb_dir: pathlib.Path, tag: str) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_file = sorted(glob.glob(str(tb_dir / "events*")))[0]
    acc = EventAccumulator(event_file, size_guidance={"scalars": 0})
    acc.Reload()
    return {s.step: s.value for s in acc.Scalars(tag)}


def build_evidence(tb_dir: pathlib.Path, ref_dump: pathlib.Path, git_rev: str) -> dict:
    rl = _tb_scalars(tb_dir, "train/loss")
    lr = _tb_scalars(tb_dir, "train/learning_rate")
    gn = _tb_scalars(tb_dir, "train/grad_norm")
    refd = json.loads(ref_dump.read_text())
    ref = {s["step"]: s for s in refd["steps"]}

    rows, within, max_abs = [], 0, 0.0
    for step in sorted(ref):
        if step not in rl:
            continue
        delta = abs(rl[step] - ref[step]["loss"])
        within += delta <= _GATE
        max_abs = max(max_abs, delta)
        rows.append(
            {
                "step": step,
                "rlinf_pinned_loss": round(rl[step], 6),
                "ref_pinned_loss": round(ref[step]["loss"], 6),
                "abs_delta": round(delta, 6),
                "within_0.03": int(delta <= _GATE),
                "rlinf_lr": lr.get(step),
                "rlinf_grad_norm": round(gn.get(step, 0.0), 5),
                "ref_grad_norm": round(ref[step]["grad_norm"], 5),
            }
        )
    n = len(rows)
    return {
        "purpose": (
            "Phase-4 AC-5 re-confirmation: the real 8-GPU FSDP train_vla_sft.py "
            "replays the reference rank-0-fanout first-N pinned batches + shared "
            "noise/time under the CURRENT canonical config, compared per-step to "
            "the reference model on identical inputs under |delta|<=0.03. "
            "Re-confirms the prior R37 50/50."
        ),
        "gate": _GATE,
        "n_steps": n,
        "within_0.03_count": f"{within}/{n}",
        "max_abs_delta": round(max_abs, 6),
        "gate_met": within == n and n > 0,
        "provenance": {
            "rlinf_git_rev": git_rev,
            "rlinf_run_command": (
                "python examples/sft/train_vla_sft.py --config-name behavior_pi05_vla "
                "runner.max_steps=50 runner.save_interval=100000 "
                "+data.pinned_inputs_npz=ref_pinned_batches.npz "
                "+data.pinned_noise_time_npz=ref_pinned_noise_time.npz "
                "(8x A800 FSDP, canonical config)"
            ),
            "ref_command": refd["meta"].get("ref_command"),
            "ref_git_rev": refd["meta"].get("ref_git_rev"),
            "global_batch": refd["meta"].get("global_batch"),
            "world_size": refd["meta"].get("world_size"),
        },
        "steps": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-tb-dir", required=True)
    parser.add_argument("--ref-dump", required=True)
    parser.add_argument("--git-rev", default="unknown")
    parser.add_argument(
        "--out", default="docs/evidence/phase4_pinned_first50_parity.json"
    )
    args = parser.parse_args()
    evidence = build_evidence(
        pathlib.Path(args.rlinf_tb_dir), pathlib.Path(args.ref_dump), args.git_rev
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    with out.with_suffix(".csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(evidence["steps"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(evidence["steps"])
    print(
        f"within {evidence['within_0.03_count']} | max |Δ| {evidence['max_abs_delta']} "
        f"| gate_met {evidence['gate_met']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
