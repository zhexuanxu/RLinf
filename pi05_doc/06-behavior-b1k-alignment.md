# 06 - BEHAVIOR B1K SFT Alignment with openpi-comet

This document describes how the RLinf_pi05 SFT pipeline for BEHAVIOR task-0000 was aligned with the `openpi-comet` reference implementation to achieve equivalent training quality.

## Background

Both `openpi-comet` (JAX) and `RLinf_pi05` (PyTorch) train pi0.5 VLA-only on the same BEHAVIOR task-0000 (turning_on_radio) dataset. The openpi-comet pipeline produced better results. After analysis, **6 differences** were identified and corrected.

## Differences Found and Fixed

### 1. Training Steps (CRITICAL)

| | Before | After (aligned) |
|---|---|---|
| Effective steps | ~16,790 | 30,000 |
| Config | `max_epochs: 10` | `max_epochs: 20` |

**Root cause:** `sft_runner.py:set_max_steps()` computes `min(steps_per_epoch * max_epochs, max_steps)`. With ~1679 steps/epoch and 10 epochs, the cap was 16,790. openpi-comet's data loader iterates infinitely and always trains for exactly 30,000 steps.

**Fix:** Set `max_epochs: 20` so `min(1679 * 20, 30000) = 30,000`.

### 2. LR Schedule End Value (HIGH)

| | Before | After (aligned) |
|---|---|---|
| min_lr | 2.5e-6 | 0.0 |

**Root cause:** openpi-comet uses `optax.warmup_cosine_decay_schedule(end_value=0.0)`. RLinf used HuggingFace's `get_cosine_with_min_lr_schedule_with_warmup(min_lr=2.5e-6)`.

**Impact:** At step 15000, LR was ~9% higher than openpi-comet's. The divergence widened towards end of training.

### 3. Normalization Statistics (HIGH)

| | Before | After (aligned) |
|---|---|---|
| Quantile aggregation | Standard LeRobot per-episode stats | openpi-comet's `aggregate_feature_stats()` |
| q01/q99 difference | Up to 30% on some dims | Bit-for-bit match |

**Root cause:** openpi-comet aggregates per-episode quantiles using `q01 = np.percentile(per_episode_q01_values, 1)`, which pushes quantiles toward extremes. Standard LeRobot uses a different aggregation formula.

**Fix:** Ported `BehaviorLerobotDatasetMetadata` (with its `aggregate_stats` logic) into RLinf. The script `toolkits/behavior/compute_behavior_norm_stats.py` now produces identical output.

### 4. Data Source and Dataset Class (MEDIUM)

| | Before | After (aligned) |
|---|---|---|
| Data path | `data/behavior-task0000-reindexed` | `data/2025-challenge-demos` |
| Dataset class | Standard `LeRobotDataset` | `BehaviorLeRobotDataset` (ported) |
| Video loading | Random access via delta_timestamps | Keyframe-based chunk streaming (GOP=250) |
| Timestamp tolerance | 1.0s (monkey-patched) | 1e-4s (native) |

**Fix:** Ported `BehaviorLeRobotDataset` and all its dependencies (omnigibson constants, video loaders, orchestrator helpers) into `rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py` with zero omnigibson imports.

### 5. Repack Transform Keys (LOW)

| | Before | After (aligned) |
|---|---|---|
| Head camera | `observation/image` | `observation/egocentric_camera` |
| Left wrist | `observation/left_wrist_image` | `observation/wrist_image_left` |
| Right wrist | `observation/right_wrist_image` | `observation/wrist_image_right` |
| Input transform | `BehaviorInputs` | `B1kInputs` |

**Fix:** Added `B1kInputs` class to `behavior_policy.py` matching openpi-comet's key names. Created `LeRobotB1KDataConfig` factory with matching repack transform.

### 6. Seed (LOW)

| | Before | After (aligned) |
|---|---|---|
| Seed | 0 | 42 |

## File Inventory

### New files
- `rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py` -- Ported `BehaviorLeRobotDataset`, `BehaviorLerobotDatasetMetadata`, inlined omnigibson deps
- `rlinf/models/embodiment/openpi/dataconfig/behavior_b1k_dataconfig.py` -- `LeRobotB1KDataConfig` factory
- `rlinf/models/embodiment/openpi/dataconfig/behavior_data_loader.py` -- `create_behavior_data_loader()` function

### Modified files
- `rlinf/models/embodiment/openpi/policies/behavior_policy.py` -- Added `B1kInputs`, `B1kOutputs`
- `rlinf/models/embodiment/openpi/dataconfig/__init__.py` -- Registered `pi05_behavior_b1k_local` config
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` -- Added B1K data loader branch in `build_dataloader()`
- `examples/sft/config/behavior_pi05_vla.yaml` -- All config fixes
- `toolkits/behavior/compute_behavior_norm_stats.py` -- Native norm-stats computation

## Data Pipeline Architecture (after alignment)

```
BehaviorLeRobotDataset (chunk streaming, GOP=250)
    reads from: data/2025-challenge-demos
    filters: task = turning_on_radio (200 episodes)
    |
    v
PromptFromLeRobotItem  (task -> prompt)
    |
    v
RepackTransform  (observation.images.rgb.head -> observation/egocentric_camera, etc.)
    |
    v
B1kInputs  (extract 23-dim state from 257-dim proprio, parse images to uint8 HWC)
    |
    v
Normalize  (quantile normalization using openpi-comet-aligned norm_stats)
    |
    v
Model Transforms  (ResizeImages 224x224, TokenizePrompt, PadStatesAndActions to 32)
    |
    v
PyTorch DataLoader  (DistributedSampler, batch_size=32/rank, infinite iteration)
```

## Parameters Verified as Matching

| Parameter | Value |
|---|---|
| Optimizer | AdamW (b1=0.9, b2=0.95, eps=1e-8, wd=1e-10) |
| Gradient clipping | L2 norm 1.0, before optimizer step |
| Peak LR / Warmup | 2.5e-5 / 1000 steps |
| Batch size | 256 global (8 GPUs x 32) |
| Total steps | 30,000 |
| Model | pi0.5, action_horizon=32, action_dim=32 |
| Trainable params | All (no freeze filter, no LoRA) |
| EMA | None |
| Loss | Flow matching (MSE on velocity field) |

## Launch Command

```bash
cd repos/RLinf_pi05
bash examples/sft/run_vla_sft.sh behavior_pi05_vla
```
