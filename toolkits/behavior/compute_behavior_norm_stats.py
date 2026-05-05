#!/usr/bin/env python3
"""Compute normalization statistics for BEHAVIOR VLA SFT.

Follows the openpi-comet norm-stats computation pipeline exactly:

1. Load metadata via BehaviorLerobotDatasetMetadata (ported into RLinf_pi05).
   This aggregates per-episode stats using the BEHAVIOR-specific aggregation
   that computes q01 = percentile(per_episode_q01_values, 1) and
   q99 = percentile(per_episode_q99_values, 99).

2. Apply extract_state_from_proprio() to the full 257-dim observation.state
   statistics, producing 23-dim state stats.

3. Pad state and action stats to the model's action_dim (32) with zeros.

4. Save in the openpi NormStats JSON format used by the pipeline's Normalize
   transform.

Usage:
    # Compute from 2025-challenge-demos for task-0000 (default)
    python toolkits/behavior/compute_behavior_norm_stats.py

    # Compute for all tasks
    python toolkits/behavior/compute_behavior_norm_stats.py --all-tasks

    # Verify against openpi-comet reference
    python toolkits/behavior/compute_behavior_norm_stats.py --verify \\
        repos/openpi-comet/task0000_sft/29999/assets/behavior-1k/2025-challenge-demos/norm_stats.json
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Ensure repo root is on the path so rlinf can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rlinf.models.embodiment.openpi.dataconfig.behavior_dataset import (
    BehaviorLerobotDatasetMetadata,
)
from rlinf.models.embodiment.openpi.policies.behavior_policy import (
    extract_state_from_proprio,
)

DATASET_ROOT = Path("/mnt/public/xzxuan/data/2025-challenge-demos")
DEFAULT_OUTPUT = Path(
    "/mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats_1.json"
)
MODEL_ACTION_DIM = 32


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad the last dimension with zeros to target_dim."""
    current_dim = arr.shape[-1]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - current_dim)]
        return np.pad(arr, pad_width, constant_values=0.0)
    return arr


def compute_norm_stats(
    dataset_root: Path,
    tasks: list[str] | None = None,
    action_dim: int = MODEL_ACTION_DIM,
) -> dict:
    """Compute norm stats matching openpi-comet's pipeline.

    Args:
        dataset_root: Path to 2025-challenge-demos.
        tasks: List of task names (None = all tasks).
        action_dim: Model action dim for padding (default 32).

    Returns:
        Dict in the NormStats JSON format: {"norm_stats": {"state": {...}, "actions": {...}}}
    """
    print(f"Loading BehaviorLerobotDatasetMetadata from {dataset_root}")
    print(f"Tasks: {tasks or 'all'}")

    metadata = BehaviorLerobotDatasetMetadata(
        repo_id="behavior-1k/2025-challenge-demos",
        root=dataset_root,
        tasks=tasks,
        modalities=[],   # No video modalities needed for stats
        cameras=[],       # No cameras needed for stats
    )

    stats = metadata.stats
    print(f"Loaded stats for {len(metadata.episodes)} episodes")

    # Extract 23-dim state from full proprio stats, then pad to action_dim
    norm_stats = {"state": {}, "actions": {}}
    for key in ["mean", "std", "q01", "q99"]:
        state_raw = stats["observation.state"][key]
        state_extracted = extract_state_from_proprio(state_raw)
        norm_stats["state"][key] = pad_to_dim(state_extracted, action_dim).tolist()

        action_raw = stats["action"][key]
        norm_stats["actions"][key] = pad_to_dim(action_raw, action_dim).tolist()

    return {"norm_stats": norm_stats}


def verify_against_reference(computed: dict, reference_path: Path) -> bool:
    """Compare computed norm stats against a reference JSON file."""
    with open(reference_path) as f:
        reference = json.load(f)

    all_match = True
    for section in ["state", "actions"]:
        for stat_key in ["mean", "std", "q01", "q99"]:
            c = computed["norm_stats"][section][stat_key]
            r = reference["norm_stats"][section][stat_key]
            if len(c) != len(r):
                print(f"  MISMATCH {section}.{stat_key}: len {len(c)} vs {len(r)}")
                all_match = False
                continue
            max_diff = max(abs(a - b) for a, b in zip(c, r))
            if max_diff > 1e-10:
                print(f"  MISMATCH {section}.{stat_key}: max_diff={max_diff:.2e}")
                all_match = False
            else:
                print(f"  OK {section}.{stat_key}: max_diff={max_diff:.2e}")

    return all_match


def main():
    parser = argparse.ArgumentParser(description="Compute BEHAVIOR norm stats")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Path to 2025-challenge-demos dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output path for norm_stats.json",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["turning_on_radio"],
        help="Task names to compute stats for (default: turning_on_radio)",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Compute stats for all 50 tasks",
    )
    parser.add_argument(
        "--verify",
        type=Path,
        default=None,
        help="Path to reference norm_stats.json to verify against",
    )
    args = parser.parse_args()

    tasks = None if args.all_tasks else args.tasks

    result = compute_norm_stats(args.dataset_root, tasks=tasks)

    # Print summary
    for section in ["state", "actions"]:
        print(f"\n{section.capitalize()} stats (first 5 dims):")
        for k in ["mean", "std", "q01", "q99"]:
            vals = result["norm_stats"][section][k][:5]
            print(f"  {k}: {[f'{v:.6f}' for v in vals]}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nNorm stats saved to: {args.output}")

    # Verify
    if args.verify:
        print(f"\nVerifying against reference: {args.verify}")
        if verify_against_reference(result, args.verify):
            print("PASS: All stats match reference")
        else:
            print("FAIL: Some stats do not match reference")
            sys.exit(1)


if __name__ == "__main__":
    main()
