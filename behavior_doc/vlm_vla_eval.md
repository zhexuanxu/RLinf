# VLM+VLA (Dual-System) Evaluation on BEHAVIOR

This guide covers **dual-system (VLM + VLA) evaluation** on BEHAVIOR-1K tasks, where a high-level Vision-Language Model generates intermediate subtasks, which a low-level VLA then executes as precise robot actions.

## Overview

The **dual-system** approach combines:
1. **VLM** (Vision-Language Model): High-level planning — interprets tasks and observations to generate subtasks
2. **VLA** (Vision-Language-Action): Low-level control — executes subtasks as continuous robot actions

```
┌──────────────────────────────────┐
│  Observation + Task Description  │
│  (+ optional memory)              │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  VLM (Qwen2.5-VL)                │
│  ├─ Generate subtask             │
│  └─ Update memory (if enabled)   │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Observation + Subtask           │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  VLA (OpenPI pi0.5)              │
│  └─ Generate actions             │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  Actions (23-dim, 32 chunks)     │
└──────────────────────────────────┘
```

## Execution Flow

### 1. Configuration

The VLM+VLA configuration is **`behavior_ppo_openpi_agentic.yaml`**. Key features:

```yaml
vlm:
  enable_memory: True        # MEM-style language memory
  k_images: -1              # Save all trajectory images
  
  model:
    model_type: "qwen2.5_vl_embodied"  # or "qwen3_vl_embodied"
    model_path: /path/to/Qwen2.5-VL
    min_pixels: 3136
    max_pixels: 12845056
  
  sampling_params:
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    max_new_tokens: 512

rollout:
  model:
    model_path: /path/to/pi05_b1kpt50_pt

env:
  eval:
    total_num_envs: 2
    max_episode_steps: 4096
```

The presence of the `vlm:` section triggers:
- `DualSystemEvalRunner` (not `EmbodiedEvalRunner`)
- `DualSystemAgentLoop` initialization in the rollout worker
- Full trajectory logging with VLM I/O and images

### 2. Worker Initialization

**RolloutWorker**:
```python
if cfg.get("vlm"):
    self._init_vlm()  # Load VLM + create agentloop
```

This:
1. Loads the VLM (Qwen2.5-VL or Qwen3-VL) on the rollout GPU
2. Creates `DualSystemAgentLoop(vlm_model, vla_model, enable_memory=True/False)`
3. Initializes per-env memory state (if `enable_memory=True`)

### 3. Evaluation Flow

#### EnvWorker Flow
Same as standalone VLA:
```
For each step:
  1. BehaviorEnv.chunk_step(actions)
     - Step the env 32 substeps
     - Collect obs, rewards, dones
  2. Track metrics
  3. Send obs + dones to RolloutWorker
  4. Receive actions from RolloutWorker
```

#### RolloutWorker Flow (Dual-System)
```
For each step:
  1. Receive obs from EnvWorker
  2. Call agentloop.run_step(obs, mode="eval")
     a. Turn 1 — VLM:
        - Input: obs + task_desc (+ memory)
        - Output: subtask (+ updated memory)
     b. Turn 2 — VLA:
        - Input: obs + subtask
        - Output: actions
  3. Send actions back to EnvWorker
  4. Log trajectory (VLM I/O, VLA I/O, images)
```

#### Memory Alignment (Auto-Reset)

When `env.eval.auto_reset=True` (enabled in eval):
```
EnvWorker                    RolloutWorker
     │                              │
     ├─ detect dones               │
     ├─ auto-reset done envs       │
     ├─ send obs + dones ─────────►│
     │                        agentloop.reset_memory_for_envs(dones)
     │◄─────── actions ────────────┤
```

This ensures memory is strictly per-episode: old memory from finished episodes doesn't leak into new episodes.

#### Memory State

If `enable_memory=True`:
```python
self._memories: list[str]  # one per env, accumulated across steps

At each reset:
  agentloop.reset_memory() or reset_memory_for_envs(dones)
  # Clears memory for fresh episodes
```

### 4. Core Code Paths

**Entry point**: `examples/embodiment/eval_embodied_agent.py`
```python
runner_cls = DualSystemEvalRunner if cfg.get("vlm") else EmbodiedEvalRunner
runner = runner_cls(cfg=cfg, rollout=rollout_group, env=env_group)
# → Uses DualSystemEvalRunner
```

**Runner**: `rlinf/agents/dualsystem/eval_runner.py::DualSystemEvalRunner`
- Inherits from `EmbodiedEvalRunner`
- Adds full trajectory logging: `.jsonl` file + `.jpg` images
- `_log_trajectories()`: Writes step-by-step VLM/VLA I/O to disk

**Agentloop**: `rlinf/agents/dualsystem/dual_system_agent_loop.py::DualSystemAgentLoop`
```python
def run_step(obs, mode="eval", vla_kwargs=None):
    # Turn 1: VLM generates subtask (+ memory)
    raw_outputs = vlm_model.generate_subtask(obs, prompt=prompts)
    subtasks, memories = parse_subtask_and_memory(raw_outputs)
    
    # Turn 2: VLA generates actions
    actions, vla_result = vla_model.predict_action_batch(obs_with_subtask)
    
    return actions, result_dict
```

**VLM models**:
- `rlinf/models/embodiment/VLM/qwen2_5_vl_policy.py::Qwen2_5_VLPolicy`
  - Uses `qwen_vl_utils.process_vision_info` for proper image handling
  - Supports batched inference with per-sample prompts (for different memories)
- `rlinf/models/embodiment/VLM/qwen3_vl_policy.py::Qwen3_VLPolicy`
  - Thinking mode always on; `<think>` tokens stripped before parsing

**Prompts**: `rlinf/agents/dualsystem/prompts.py`
- `DEFAULT_VLM_PROMPT`: Task → subtask (memoryless)
- `MEMORY_VLM_PROMPT`: Task + memory → subtask + memory (with MEM formulation)
- `parse_subtask_and_memory()`: Robust JSON parsing with fallback

## Launching Evaluation

### Basic Command

```bash
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic
```

This uses the default task "turning_on_radio" with fixed initialization and memory enabled.

### Switching Tasks

Same as standalone VLA, but now the VLM also needs to understand the new task:

**Option 1: CLI override**
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

**Option 2: Custom config YAML**
```yaml
# behavior_ppo_openpi_agentic_task8.yaml
defaults:
  - behavior_ppo_openpi_agentic

env:
  eval:
    omni_config:
      task:
        activity_name: rearranging_kitchen_furniture
```

Then:
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic_task8
```

### Switching Scene Initialization

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/cached_instances/ \
  env.eval.omni_config.task.instance_file_format=tro_state
```

### Toggling Memory

To disable memory (memoryless VLM mode):
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.enable_memory=False
```

This switches the VLM prompt to `DEFAULT_VLM_PROMPT` (no memory input/output).

### Switching VLM Model

To use Qwen3-VL-Thinking instead of Qwen2.5-VL:
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.model.model_type=qwen3_vl_embodied \
  vlm.model.model_path=/path/to/Qwen3-VL-4B-Thinking
```

## Trajectory Logging

Unlike the standalone VLA, the dual-system approach logs complete trajectories:

### Output Structure

```
{log_path}/trajectories/
├── trajectory.jsonl     # One JSON line per step
└── images/
    ├── w0_s0000.jpg     # worker 0, step 0
    ├── w0_s0001.jpg
    └── ...
```

### JSONL Record Format

Each line is a JSON object:
```json
{
  "worker": 0,
  "step": 3,
  "vlm_inputs": {
    "task_descriptions": ["Turn on the radio..."],
    "input_memories": ["Found the radio location..."],
    "image_path": "images/w0_s0003.jpg"
  },
  "vlm_outputs": {
    "raw_outputs": ["Subtask: Extend arm to reach radio..."],
    "subtasks": ["Extend arm to reach radio knob..."],
    "output_memories": ["Found radio, now attempting to turn on..."]
  },
  "vla_inputs": {
    "subtasks": ["Extend arm to reach radio knob..."]
  },
  "vla_outputs": {
    "actions": "skip now, it's too long"
  }
}
```

### Controlling Image Saving

```yaml
vlm:
  k_images: 10     # Save 10 uniformly sampled steps per trajectory
  k_images: -1     # Save all steps
  k_images: 0      # Don't save any images
```

## Expected Output

### Console Log

```
eval/success_once: 0.75
eval/episode_len: 3456
eval/num_trajectories: 8
Trajectory logged to {log_path}/trajectories/trajectory.jsonl (2 workers)
```

### Trajectory Files

```
ls -la {log_path}/trajectories/
trajectory.jsonl (1000+ lines)
images/w0_s0000.jpg
images/w0_s0001.jpg
...
images/w1_s0999.jpg
```

## Memory Module (MEM Formulation)

When `enable_memory=True`, the VLM implements the MEM (Memory-Enabled planner) formulation:

```
π_HL(l_{t+1}, m_{t+1} | o_t, m_t, g)

where:
  o_t = observation at time t
  m_t = memory at time t (compressed history)
  g   = goal (task description)
  l_t = subtask (high-level action)
  π_HL = high-level policy (VLM)
```

**At each step**:
1. VLM receives: task + prev memory + current obs
2. VLM outputs: next subtask + updated memory
3. Memory accumulates across steps (like a conversation history)
4. When episode ends or resets, memory is cleared

**Memory update strategy** (from prompt):
> Keep only information that is relevant for completing the remaining goal.

This encourages the VLM to compress and summarize progress, not just concatenate step-by-step logs.

## Performance Tips

1. **Memory context**: With `enable_memory=True`, each step's VLM input is longer (includes accumulated memory). Monitor VLM inference speed.
2. **VLM model size**: Qwen2.5-VL-3B is fast; Qwen3-VL-4B-Thinking is slower but may be more capable.
3. **Batch prompts**: The VLM API accepts `prompt: str | list[str]`. When memories differ per-env, all B envs are processed in one batched call (not B separate calls).
4. **Sampling params**: Lower `temperature` during eval (default 0.6) for more deterministic subtask generation.

## Comparison with Standalone VLA

| Aspect | Standalone VLA | VLM+VLA |
|--------|----------------|---------|
| **Planning** | None (end-to-end) | High-level subtask from VLM |
| **Model Count** | 1 (VLA only) | 2 (VLM + VLA) |
| **Inference Speed** | ~100 ms/step | ~500 ms/step (VLM overhead) |
| **Interpretability** | Low | High (subtasks logged) |
| **Trajectory Logging** | Metrics only | Full I/O + images |
| **Memory** | N/A | Optional (MEM formulation) |
| **Config** | No `vlm:` | Has `vlm:` section |

## Troubleshooting

### VLM Crashes or OOM

1. Reduce batch size: `env.eval.total_num_envs=1`
2. Reduce image resolution: `vlm.model.max_pixels=9437184` (was 12845056)
3. Use lighter VLM: `vlm.model.model_type=qwen2.5_vl_embodied` (3B vs 4B)

### Poor Subtask Quality

1. Increase `vlm.sampling_params.temperature` to 1.0 for more diverse outputs
2. Increase `max_new_tokens` (default 512) if subtasks are truncated
3. Check that `vlm.model.model_path` points to correct checkpoint

### Memory Not Updating

Check logs for:
```
[DualSystem] Memory: <memory text>
```

If memory is always empty `(no memory yet)`, check:
1. `vlm.enable_memory=True`
2. VLM is outputting valid JSON with `{"subtask": "...", "memory": "..."}`
3. Parse fallbacks in `parse_subtask_and_memory()` in `prompts.py`

### Trajectory File Corruption

If `trajectory.jsonl` is incomplete or truncated:
1. Check disk space
2. Ensure no other process is writing to the same log directory
3. Verify write permissions: `chmod 755 {log_path}/trajectories/`

## See Also

- [behavior_config.md](behavior_config.md) — BEHAVIOR task and scene configuration
- [vla_eval.md](vla_eval.md) — Standalone VLA approach for comparison
- [CLAUDE.md](../CLAUDE.md) — Overall dual-system agentloop architecture
