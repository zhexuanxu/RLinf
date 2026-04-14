# Standalone VLA Evaluation on BEHAVIOR

This guide covers **standalone Vision-Language-Action (VLA) evaluation** on BEHAVIOR-1K tasks, where the policy takes observations and task descriptions directly and produces robot actions without high-level subtask planning.

## Overview

The **standalone VLA** approach uses only the low-level action model (OpenPI pi0.5):

```
Task Description (text)  ┐
                          ├─► VLA (pi0.5) ──► Actions (23-dim, 32 chunks)
Observation (images)     ┘
```

The task description (e.g., "Turn on the radio receiver that's on the table in the living room") is passed directly to the VLA, which must learn to interpret both the language instruction and visual observations to generate appropriate actions.

## Execution Flow

### 1. Configuration

The standalone VLA configuration is **`behavior_ppo_openpi_pi05.yaml`**. This config:
- **Excludes** the `vlm:` section (no VLM model)
- Uses `EmbodiedEvalRunner` (not `DualSystemEvalRunner`)
- Calls `predict()` → `vla_model.predict_action_batch()` directly

Key configuration sections:
```yaml
env:
  eval:
    total_num_envs: 4
    max_episode_steps: 4096
    max_steps_per_rollout_epoch: 4096

rollout:
  model:
    model_path: /path/to/pi05_b1kpt50_pt

actor:
  model:
    model_path: /path/to/pi05_b1kpt50_pt
    num_action_chunks: 32
    openpi:
      config_name: "pi05_behavior"
```

No `vlm:` section is present.

### 2. Evaluation Flow

The evaluation runs in two main processes: **EnvWorker** and **RolloutWorker**.

#### EnvWorker Flow
```
┌─ env_worker.evaluate() ─────────────────────────┐
│                                                 │
│  For each step:                                 │
│    1. BehaviorEnv.chunk_step(actions)           │
│       - Step the env 32 substeps (1 chunk)      │
│       - Collect obs, rewards, dones              │
│    2. Track metrics (success, returns, etc.)    │
│    3. Send obs + dones to RolloutWorker         │
│    4. Receive actions from RolloutWorker        │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### RolloutWorker Flow
```
┌─ rollout.evaluate() ───────────────────────────┐
│                                                │
│  1. Load VLA model (pi0.5)                     │
│  2. For each step:                             │
│     a. Receive obs from EnvWorker              │
│     b. Extract images, states, task desc       │
│     c. Call vla.predict_action_batch()         │
│        obs → [actions tensor]                  │
│     d. Send actions back to EnvWorker          │
│                                                │
└────────────────────────────────────────────────┘
```

#### Metric Collection
- **Rising-edge detection**: Metrics collected only when an episode transitions from "running" to "done" (not every step)
- **Per-env tracking**: Tracks success, episode length, returns independently per environment
- **Aggregation**: Results from all env workers aggregated by `compute_evaluate_metrics()`

### 3. Core Code Paths

**Entry point**: `examples/embodiment/eval_embodied_agent.py`
```python
runner_cls = DualSystemEvalRunner if cfg.get("vlm") else EmbodiedEvalRunner
runner = runner_cls(cfg=cfg, rollout=rollout_group, env=env_group)
```
→ Uses `EmbodiedEvalRunner` since no `vlm:` section in config.

**Runner**: `rlinf/runners/embodied_eval_runner.py::EmbodiedEvalRunner`
- `evaluate()`: Launches env and rollout workers, waits for results, aggregates metrics
- `run()`: Calls `evaluate()` once and logs results

**Environment**: `rlinf/envs/behavior/behavior_env.py::BehaviorEnv`
- `chunk_step()`: Steps the environment 32 times (chunk size), tracks step count, enforces truncation at `max_episode_steps`
- `_record_metrics()`: Accumulates returns and success flags per environment

**Rollout**: `rlinf/workers/rollout/hf/huggingface_worker.py::MultiStepRolloutWorker`
- `predict()`: Routes obs → `vla_model.predict_action_batch(obs, mode="eval")`
  - When no `vlm:` section, agentloop is `None`, so this path is skipped
- `evaluate()`: Calls `predict()` repeatedly during eval, collects trajectories

## Launching Evaluation

### Basic Command

```bash
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05
```

This uses the default task configuration in `behavior_r1pro.yaml`, which is "turning_on_radio" with fixed initialization.

### Switching Tasks

To evaluate on a different task, override the task via YAML or CLI:

**Option 1: Modify the config YAML**

Edit `examples/embodiment/config/env/behavior_r1pro.yaml`:
```yaml
omni_config:
  task:
    activity_name: rearranging_kitchen_furniture  # change to any task
    activity_definition_id: 0
```

Then run:
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05
```

**Option 2: Override via Hydra CLI** (recommended)

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

**Option 3: Create a custom config**

Copy `behavior_ppo_openpi_pi05.yaml` to `behavior_ppo_openpi_pi05_task8.yaml` and edit:
```yaml
defaults:
  - env/behavior_r1pro@env.train
  - env/behavior_r1pro@env.eval
  ...

env:
  eval:
    omni_config:
      task:
        activity_name: rearranging_kitchen_furniture
```

Then run:
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_task8
```

### Switching Scene Initialization

To use offline-sampled scenes instead of the fixed default:

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/cached_instances/ \
  env.eval.omni_config.task.instance_file_format=tro_state
```

Or for online sampling:
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=online \
  env.eval.omni_config.task.online_object_sampling=True \
  env.eval.omni_config.task.use_presampled_robot_pose=False
```

See [behavior_config.md](behavior_config.md) for detailed initialization documentation.

## Expected Output

### Metrics

The evaluation logs:
```
eval/success_once: 0.5   (success rate)
eval/episode_len: 3456   (average episode length before truncation)
eval/num_trajectories: 8 (total completed episodes)
```

Printed to tensorboard and logged file.

### Video

If `env.eval.video_cfg.save_video: True`, videos are saved to:
```
{log_path}/video/eval/
```

### No Trajectory Logging

Unlike the VLM+VLA path, the standalone VLA evaluation does **not** log detailed trajectories (`trajectory.jsonl`). To add this, you would need to:
1. Extend `EmbodiedEvalRunner` or
2. Switch to `DualSystemEvalRunner` and configure a `vlm:` section

## Performance Tips

1. **Episode length**: Increase `max_episode_steps` and `max_steps_per_rollout_epoch` together (default 4096 for pi0.5).
2. **Task diversity**: Use `instance_resample_mode: offline` with cached instances for diverse evaluation without online sampling overhead.
3. **Batch size**: Increase `env.eval.total_num_envs` to evaluate more environments in parallel (limited by GPU memory).
4. **Scene optimization**: Reduce `load_room_types` to only necessary rooms (e.g., `["kitchen"]`) to speed up initialization.

## Comparison with VLM+VLA

| Aspect | Standalone VLA | VLM+VLA |
|--------|----------------|---------|
| **Input Language** | Task description only | Task description + optional memory |
| **Planning** | None (end-to-end) | High-level subtask from VLM |
| **Model Count** | 1 (VLA only) | 2 (VLM + VLA) |
| **Inference Speed** | Faster | Slower (VLM overhead) |
| **Interpretability** | Low (end-to-end) | Higher (subtasks visible) |
| **Trajectory Logging** | Basic (metrics only) | Full (VLM I/O, images) |
| **Config** | No `vlm:` section | Has `vlm:` section |
| **Runner** | `EmbodiedEvalRunner` | `DualSystemEvalRunner` |

## Troubleshooting

### Issues with OmniGibson Timeout

If episodes truncate before `max_episode_steps`, check:
1. OmniGibson's timeout is correctly overridden by RLinf
2. BehaviorEnv's step counter is tracking correctly

Solution: Verify in logs that all episodes reach the expected max step count.

### Missing Task Description

If the VLA receives empty task descriptions, ensure:
1. `behavior_task.jsonl` exists in `rlinf/envs/behavior/`
2. `omni_config.task.activity_name` is a valid task name

### Low Success Rate

Common causes:
1. Task is too difficult for the pi0.5 checkpoint used
2. Scene initialization doesn't match training distribution
3. Episode length is too short to complete the task

Try:
- Using `instance_resample_mode: offline` for familiar scenes
- Increasing `max_episode_steps`
- Verifying the checkpoint was trained on the same task

## See Also

- [behavior_config.md](behavior_config.md) — Full configuration reference for BEHAVIOR tasks and initialization
- [vlm_vla_eval.md](vlm_vla_eval.md) — VLM+VLA approach with subtask planning
- [CLAUDE.md](../CLAUDE.md) — Overall dual-system agentloop architecture
