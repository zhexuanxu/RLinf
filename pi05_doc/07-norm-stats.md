# 07 - Normalization Statistics for Pi0.5 BEHAVIOR SFT

## What Are Norm Stats?

Pi0.5 uses **quantile normalization** to scale proprioceptive state and action values to the [-1, 1] range before feeding them to the model. The formula is:

```
normalized = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

This maps the [1st percentile, 99th percentile] range to [-1, 1], which is more robust to outliers than z-score normalization. The norm stats file stores four statistics per dimension: `mean`, `std`, `q01`, `q99`.

## Why Pi0.5 Needs Them

The base pi0.5 model was pre-trained with specific normalization assumptions per robot platform. For BEHAVIOR's R1Pro robot:
- **State**: 23-dim extracted from 257-dim proprioception (base velocity, trunk position, arm joints, gripper widths), padded to 32
- **Actions**: 23-dim joint targets, padded to 32

Without correct norm stats, the model receives inputs at the wrong scale, causing degraded performance.

## How Computation Works

The norm stats computation uses `BehaviorLerobotDatasetMetadata` to aggregate per-episode statistics from the dataset:

1. Load episode-level stats from `meta/episodes_stats.jsonl` in the dataset
2. Filter to the target task(s) (e.g., `turning_on_radio` = task-0000)
3. Aggregate across episodes using `aggregate_feature_stats()`:
   - **mean/std**: Weighted parallel aggregation (exact)
   - **q01**: `np.percentile(per_episode_q01_values, 1, axis=0)` -- takes the 1st percentile of all episodes' q01 values
   - **q99**: `np.percentile(per_episode_q99_values, 99, axis=0)` -- analogous
4. Apply `extract_state_from_proprio()` to the full-dim observation.state stats, producing 23-dim state stats
5. Pad state and action stats to model action_dim (32) with zeros
6. Save as JSON in the openpi NormStats format

## Command

```bash
# From the repo root, using .venv_pi:
.venv_pi/bin/python toolkits/behavior/compute_behavior_norm_stats.py

# With explicit arguments:
.venv_pi/bin/python toolkits/behavior/compute_behavior_norm_stats.py \
    --dataset-root /mnt/public/xzxuan/data/2025-challenge-demos \
    --tasks turning_on_radio \
    --output /mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json

# Compute for all 50 tasks:
.venv_pi/bin/python toolkits/behavior/compute_behavior_norm_stats.py --all-tasks
```

## Output Location

```
/mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json
```

The training pipeline loads this file via the `AssetsConfig(asset_id="physical-intelligence/behavior")` setting in the `pi05_behavior_b1k_local` config.

## JSON Format

```json
{
  "norm_stats": {
    "state": {
      "mean": [32 floats],
      "std":  [32 floats],
      "q01":  [32 floats],
      "q99":  [32 floats]
    },
    "actions": {
      "mean": [32 floats],
      "std":  [32 floats],
      "q01":  [32 floats],
      "q99":  [32 floats]
    }
  }
}
```

## Verification Against openpi-comet

To verify the computed norm stats match the openpi-comet reference exactly:

```bash
.venv_pi/bin/python toolkits/behavior/compute_behavior_norm_stats.py \
    --output /tmp/test_norm_stats.json \
    --verify repos/openpi-comet/task0000_sft/29999/assets/behavior-1k/2025-challenge-demos/norm_stats.json
```

Expected output:
```
  OK state.mean: max_diff=0.00e+00
  OK state.std: max_diff=0.00e+00
  OK state.q01: max_diff=0.00e+00
  OK state.q99: max_diff=0.00e+00
  OK actions.mean: max_diff=0.00e+00
  OK actions.std: max_diff=0.00e+00
  OK actions.q01: max_diff=0.00e+00
  OK actions.q99: max_diff=0.00e+00
PASS: All stats match reference
```

All eight stat vectors must show `max_diff=0.00e+00` (bit-for-bit match).
