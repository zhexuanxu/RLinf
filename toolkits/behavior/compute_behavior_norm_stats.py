#!/usr/bin/env python3
"""Compute task-0000-specific normalization statistics for BEHAVIOR VLA SFT.

Reads the re-indexed task-0000 dataset, extracts state from proprio (256->23 dims),
pads both state and actions to 32 dims, and computes mean/std/q01/q99.

Usage:
    python toolkits/behavior/compute_behavior_norm_stats.py

Prerequisite: Run prepare_behavior_task0000.py first.

Output: /mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json
"""

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Import state extraction from the behavior policy
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from rlinf.models.embodiment.openpi.policies.behavior_policy import extract_state_from_proprio

DATASET_DIR = Path("/mnt/public/xzxuan/data/behavior-task0000-reindexed")
OUTPUT_PATH = Path("/mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json")

MODEL_DIM = 32  # pi0.5 model dimension for state/action padding
RAW_ACTION_DIM = 23
RAW_STATE_DIM = 23  # after extraction from 256-dim proprio


def pad_to_dim(arr: np.ndarray, target_dim: int) -> np.ndarray:
    """Pad the last dimension with zeros to target_dim."""
    if arr.shape[-1] >= target_dim:
        return arr[..., :target_dim]
    pad_width = [(0, 0)] * (arr.ndim - 1) + [(0, target_dim - arr.shape[-1])]
    return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def main():
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output:  {OUTPUT_PATH}")

    # Collect all parquet files
    parquet_files = sorted(DATASET_DIR.rglob("data/**/*.parquet"))
    print(f"Found {len(parquet_files)} parquet files")

    all_states = []
    all_actions = []

    for i, pf in enumerate(parquet_files):
        table = pq.read_table(pf)

        # Extract raw state (256-dim) and action (23-dim)
        state_col = table.column("observation.state").to_pylist()
        action_col = table.column("action").to_pylist()

        for state_raw, action_raw in zip(state_col, action_col):
            state_raw = np.array(state_raw, dtype=np.float32)
            action_raw = np.array(action_raw, dtype=np.float32)

            # Extract 23-dim state from 256-dim proprio
            state_extracted = extract_state_from_proprio(state_raw)

            # Pad to model dimension
            state_padded = pad_to_dim(state_extracted, MODEL_DIM)
            action_padded = pad_to_dim(action_raw, MODEL_DIM)

            all_states.append(state_padded)
            all_actions.append(action_padded)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(parquet_files)} files ({len(all_states)} frames)")

    all_states = np.stack(all_states, axis=0)  # [N, 32]
    all_actions = np.stack(all_actions, axis=0)  # [N, 32]
    print(f"\nTotal frames: {len(all_states)}")
    print(f"State shape: {all_states.shape}, Action shape: {all_actions.shape}")

    # Compute statistics
    def compute_stats(data: np.ndarray) -> dict:
        return {
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "q01": np.percentile(data, 1, axis=0).tolist(),
            "q99": np.percentile(data, 99, axis=0).tolist(),
        }

    state_stats = compute_stats(all_states)
    action_stats = compute_stats(all_actions)

    # Print summary
    print(f"\nState stats (first 5 dims):")
    for k in ["mean", "std", "q01", "q99"]:
        print(f"  {k}: {[f'{v:.4f}' for v in state_stats[k][:5]]}")
    print(f"\nAction stats (first 5 dims):")
    for k in ["mean", "std", "q01", "q99"]:
        print(f"  {k}: {[f'{v:.4f}' for v in action_stats[k][:5]]}")

    # Save in openpi NormStats format
    norm_stats = {
        "norm_stats": {
            "state": state_stats,
            "actions": action_stats,
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\nNorm stats saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
