# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Audit the BEHAVIOR action-training frame distribution: level-0 vs level-1.

Level-0 (action-only baseline) streams every frame of each episode. Level-1
(the vlm_vla recipe) keeps only frames the subtask resolver maps to a skill —
dropping frames outside ``valid_duration``, trailing gaps, and (when
``enable_gap`` is false) inter-window and leading gaps. If level-1 trains the
action expert on a much smaller or skewed frame set than the working level-0
baseline, that is a candidate cause for the vlm_vla action-quality gap.

This is a pure-CPU audit over the annotation JSON + LeRobot meta; it imports
only the dependency-free segment resolver, never the heavy dataset package. Run:

    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/vlm_vla_diagnostics/audit_frame_distribution.py \
        --dataset-root /mnt/public/xzxuan/data/2025-challenge-demos \
        --task-id 0 --num-subtasks 4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

# Import the pure resolver directly from its file so the heavy dataset package
# __init__ (omnigibson / lerobot) is never triggered by this audit.
_RESOLVER_PATH = (
    pathlib.Path(__file__).resolve().parents[2]
    / "rlinf/data/datasets/openpi_pytorch/behavior/skill_segments.py"
)
_spec = importlib.util.spec_from_file_location("_skill_segments", _RESOLVER_PATH)
_seg = importlib.util.module_from_spec(_spec)
# Register before exec so the module's @dataclass can resolve its own __module__.
sys.modules[_spec.name] = _seg
_spec.loader.exec_module(_seg)
build_skill_segments = _seg.build_skill_segments
resolve_frame_subtask = _seg.resolve_frame_subtask


def _episode_lengths(dataset_root: pathlib.Path) -> dict[int, int]:
    """Map episode_index -> frame length from meta/episodes.jsonl."""
    lengths: dict[int, int] = {}
    with open(dataset_root / "meta" / "episodes.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            lengths[int(row["episode_index"])] = int(row["length"])
    return lengths


def _count_level1(segments, length: int, enable_gap: bool) -> tuple[int, list[int]]:
    """Count frames level-1 keeps over [0, length) and the per-skill histogram."""
    # resolve_frame_subtask only ever returns an index from segments.skill_indices,
    # so sizing per_skill to max(skill_indices) + 1 always covers every result.
    n_subtasks = (max(segments.skill_indices) + 1) if segments.skill_indices else 0
    per_skill = [0] * n_subtasks
    used = 0
    for frame in range(length):
        idx = resolve_frame_subtask(segments, frame, enable_gap)
        if idx is None:
            continue
        used += 1
        per_skill[idx] += 1
    return used, per_skill


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--num-subtasks", type=int, default=4)
    args = ap.parse_args()

    root = pathlib.Path(args.dataset_root)
    ann_dir = root / "annotations" / f"task-{args.task_id:04d}"
    lengths = _episode_lengths(root)

    ann_files = sorted(ann_dir.glob("episode_*.json"))
    if not ann_files:
        raise SystemExit(f"no annotation files under {ann_dir}")

    tot_level0 = tot_valid = 0
    tot_l1_gap = tot_l1_nogap = 0
    per_skill_gap = [0] * args.num_subtasks
    n_eps = 0
    n_len_mismatch = 0

    for ann_file in ann_files:
        ep_id = int(ann_file.stem.split("_")[-1])
        if ep_id not in lengths:
            continue
        ann = json.load(open(ann_file))
        segments = build_skill_segments(ann, args.num_subtasks, ep_id)
        length = lengths[ep_id]
        valid = segments.valid_end - segments.valid_start
        l1_gap, hist_gap = _count_level1(segments, length, enable_gap=True)
        l1_nogap, _ = _count_level1(segments, length, enable_gap=False)

        tot_level0 += length
        tot_valid += valid
        tot_l1_gap += l1_gap
        tot_l1_nogap += l1_nogap
        for i in range(min(len(hist_gap), args.num_subtasks)):
            per_skill_gap[i] += hist_gap[i]
        if segments.valid_end > length:
            n_len_mismatch += 1
        n_eps += 1

    print(f"=== Level-0 vs Level-1 action-frame audit (task-{args.task_id:04d}) ===")
    print(f"episodes with annotation+meta: {n_eps}")
    print(f"episodes whose valid_end exceeds meta length: {n_len_mismatch}")
    print(f"level-0 frames (full episodes, baseline action data): {tot_level0}")
    print(f"valid_duration frames (sum):                         {tot_valid}")
    print(
        f"level-1 frames (enable_gap=True, the config default): {tot_l1_gap}"
        f"  ({100.0 * tot_l1_gap / max(tot_level0, 1):.1f}% of level-0)"
    )
    print(
        f"level-1 frames (enable_gap=False):                    {tot_l1_nogap}"
        f"  ({100.0 * tot_l1_nogap / max(tot_level0, 1):.1f}% of level-0)"
    )
    dropped = tot_level0 - tot_l1_gap
    print(
        f"frames level-1(gap=True) drops vs level-0:            {dropped}"
        f"  ({100.0 * dropped / max(tot_level0, 1):.1f}%)"
    )
    print("per-subtask frames (enable_gap=True):")
    for i, c in enumerate(per_skill_gap):
        share = 100.0 * c / max(tot_l1_gap, 1)
        print(f"  skill {i}: {c}  ({share:.1f}%)")


if __name__ == "__main__":
    main()
