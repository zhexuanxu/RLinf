#!/usr/bin/env python3
"""Analyze VLM subtask quality from a trajectory JSONL file.

Usage:
    python tools/analyze_trajectory.py <path_to_trajectory.jsonl>

Reports:
    1. How many of the 64 trajectories have strictly ordered 4-skill sequences
    2. How many VLM calls output one of the 4 valid skills
    3. Per-trajectory compressed skill sequence (deduplicated consecutive runs)
"""

import json
import sys
from collections import Counter
from pathlib import Path

VALID_SKILLS = {
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
}
SKILL_ORDER = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]
SHORT = {
    "move to radio": "move_to",
    "pick up radio from coffee table": "pick_up",
    "press radio": "press",
    "place radio on coffee table": "place",
}


def compress_seq(subtasks: list[str]) -> list[str]:
    """Deduplicate consecutive identical subtasks."""
    out = []
    for st in subtasks:
        if not out or out[-1] != st:
            out.append(st)
    return out


def classify_trajectory(subtasks: list[str]) -> str:
    """Classify a trajectory into OK / ~~ / XX.

    OK: compressed sequence is exactly [move_to, pick_up, press, place],
        no invalid skills, no duplicates — the perfect 4-skill sequence.
    ~~: all skills are valid, ordering is strictly non-decreasing, AND
        no skill is skipped (each transition is +0 or +1 in skill index,
        never jumps like pick_up -> place skipping press).
    XX: everything else (invalid skills, ordering regression, or skips).
    """
    all_valid = all(st in VALID_SKILLS for st in subtasks)
    compressed = compress_seq(subtasks)
    compressed_valid = [st for st in compressed if st in VALID_SKILLS]

    # --- OK: perfect 4-skill sequence ---
    if all_valid and compressed_valid == SKILL_ORDER:
        return "OK"

    # --- ~~: non-decreasing + no skip ---
    if not all_valid:
        return "XX"

    indices = [SKILL_ORDER.index(st) for st in subtasks]
    # Must be non-decreasing
    if not all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1)):
        return "XX"
    # Each step must be +0 or +1 (no skip)
    for i in range(len(indices) - 1):
        if indices[i + 1] - indices[i] > 1:
            return "XX"

    return "~~"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trajectory.jsonl>")
        sys.exit(1)

    jsonl_path = Path(sys.argv[1])
    with open(jsonl_path) as f:
        lines = [json.loads(l) for l in f]

    print(f"File: {jsonl_path}")
    print(f"Total VLM calls: {len(lines)}")
    print()

    # ── 1. Group into (worker, env) then split into 128-step epochs ──
    env_groups: dict[tuple, list] = {}
    for e in lines:
        key = (e["worker_idx"], e["env_idx"])
        env_groups.setdefault(key, []).append(e)

    trajectories = []
    for key in sorted(env_groups):
        entries = env_groups[key]
        n_epochs = len(entries) // 128
        for epoch in range(n_epochs):
            chunk = entries[epoch * 128 : (epoch + 1) * 128]
            subtasks = [e["vlm_outputs"]["subtasks"] for e in chunk]
            trajectories.append(
                {"w": key[0], "env": key[1], "epoch": epoch, "subtasks": subtasks}
            )

    n_traj = len(trajectories)

    # ── 2. Per-VLM-call valid rate ──
    all_subtasks = [st for t in trajectories for st in t["subtasks"]]
    valid_count = sum(1 for st in all_subtasks if st in VALID_SKILLS)
    freq = Counter(all_subtasks)

    print("=" * 70)
    print(f"  SUBTASK VALIDITY  ({valid_count}/{len(all_subtasks)} "
          f"= {100 * valid_count / len(all_subtasks):.2f}% valid)")
    print("=" * 70)
    for st, cnt in freq.most_common():
        tag = "OK" if st in VALID_SKILLS else "BAD"
        print(f"  [{tag}] {st:45s}  {cnt:5d}  ({100 * cnt / len(all_subtasks):.2f}%)")
    print()

    # ── 3. Classify each trajectory ──
    for t in trajectories:
        t["grade"] = classify_trajectory(t["subtasks"])

    n_ok = sum(1 for t in trajectories if t["grade"] == "OK")
    n_mid = sum(1 for t in trajectories if t["grade"] == "~~")
    n_xx = sum(1 for t in trajectories if t["grade"] == "XX")

    print("=" * 70)
    print("  TRAJECTORY GRADING")
    print("=" * 70)
    print(f"  [OK] Perfect 4-skill (move→pick→press→place) : {n_ok}/{n_traj} "
          f"({100 * n_ok / n_traj:.1f}%)")
    print(f"  [~~] Non-decreasing, no skip, incomplete      : {n_mid}/{n_traj} "
          f"({100 * n_mid / n_traj:.1f}%)")
    print(f"  [XX] Regression / skip / invalid skill        : {n_xx}/{n_traj} "
          f"({100 * n_xx / n_traj:.1f}%)")
    print()

    # ── 4. Per-trajectory compressed skill sequence ──
    print("=" * 70)
    print("  PER-TRAJECTORY SKILL SEQUENCE  (consecutive duplicates compressed)")
    print("=" * 70)

    for t in trajectories:
        compressed = compress_seq(t["subtasks"])
        short_seq = []
        for st in compressed:
            short_seq.append(SHORT.get(st, f"[{st}]"))
        tag = t["grade"]
        line = ", ".join(short_seq)
        print(f"  [{tag}] w{t['w']}_e{t['env']} ep{t['epoch']:d}:  {line}")

    print()
    print("=" * 70)
    print("  LEGEND")
    print("=" * 70)
    print("  [OK] = perfect: exactly move_to → pick_up → press → place, no extras")
    print("  [~~] = acceptable: non-decreasing order, no skip, but incomplete")
    print("  [XX] = bad: ordering regression, skill skip, or invalid skill")
    print("  [some_text] in sequence = non-standard skill (not in training set)")
    print()


if __name__ == "__main__":
    main()
