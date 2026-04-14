# CLAUDE.md — Dual-System Embodied Agentloop (VLM + VLA)

## Overview

This project implements a **dual-system embodied evaluation pipeline** inside the
RLinf distributed RL framework. A high-level **VLM** (Vision-Language Model) generates
subtask instructions, which a low-level **VLA** (Vision-Language-Action model) executes
as continuous robot actions in the **Behavior** simulation environment.

```
    ┌─────────────────────────────────────────────────────────────────┐
    │                     Evaluation Loop                            │
    │                                                                │
    │   Env obs ──► VLM (Qwen2.5-VL / Qwen3-VL-Thinking)           │
    │                  │                                             │
    │                  ├── subtask (text)                            │
    │                  └── memory  (text, optional)                  │
    │                        │                                       │
    │                        ▼                                       │
    │              VLA (OpenPI pi0.5)                                │
    │                  │                                             │
    │                  └── actions (23-dim, 32 chunks)               │
    │                        │                                       │
    │                        ▼                                       │
    │              Behavior Env ──► next obs ──► loop                │
    └─────────────────────────────────────────────────────────────────┘
```

## Architecture

### Component Placement

Standard `actor, env, rollout` placement. The VLM and VLA both live on the
**rollout** GPU inside the existing `MultiStepRolloutWorker`. No separate
"agentloop" worker or GPU group.

```yaml
cluster:
  component_placement:
    actor: 0-1
    env,rollout: 0-1
```

### Key Files

| File | Role |
|------|------|
| `rlinf/agents/dualsystem/dual_system_agent_loop.py` | Core coordination: VLM→subtask→VLA→action. Manages per-env memory state. |
| `rlinf/agents/dualsystem/prompts.py` | Prompt templates (memoryless + memory-enabled) and JSON output parsers. |
| `rlinf/agents/dualsystem/eval_runner.py` | `DualSystemEvalRunner`: inherits `EmbodiedEvalRunner`, adds `.jsonl`+`.jpg` trajectory logging. |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `MultiStepRolloutWorker`: `_init_vlm()` loads VLM + creates agentloop; `predict()` routes through it; `evaluate()` collects trajectories + handles memory reset on env auto-reset. |
| `rlinf/models/embodiment/VLM/qwen2_5_vl_policy.py` | Qwen2.5-VL wrapper. Uses `qwen_vl_utils.process_vision_info`. |
| `rlinf/models/embodiment/VLM/qwen3_vl_policy.py` | Qwen3-VL-Thinking wrapper. Thinking mode always on; `<think>` tokens stripped. |
| `rlinf/models/embodiment/VLM/__init__.py` | Factory dispatching `qwen2.5_vl_embodied` / `qwen3_vl_embodied`. |
| `rlinf/config.py` | `SupportedModel` enum: `QWEN2_5_VL_EMBODIED`, `QWEN3_VL_EMBODIED`. |
| `rlinf/data/embodied_io_struct.py` | Extended `ChunkStepResult`, `Trajectory`, `EmbodiedRolloutResult` with `subtasks` field. |
| `examples/embodiment/config/behavior_ppo_openpi_agentic.yaml` | Eval config with VLM settings. |
| `examples/embodiment/eval_embodied_agent.py` | Launch script (shared for standard and agentic eval). |
| `examples/embodiment/eval_embodiment.sh` | Shell entry point; auto-detects `"agentic"` in config name. |

## How It Works

### 1. Worker Initialization (`init_worker`)

1. `MultiStepRolloutWorker.init_worker()` loads the VLA (pi0.5) via `get_model()`.
2. If `cfg.vlm` is present, `_init_vlm()` is called:
   - Loads the VLM (Qwen2.5-VL or Qwen3-VL) via `get_model()`.
   - Creates a `DualSystemAgentLoop(vlm_model, vla_model, ...)`.
3. `setup_sample_params()` sets up both VLA and VLM sampling params:
   - VLA: `_train_sampling_params`, `_eval_sampling_params` (from `cfg.algorithm`).
   - VLM: `_vlm_sampling_params` (from `cfg.vlm.sampling_params`).

### 2. Predict Flow (`predict`)

```python
def predict(self, env_obs, mode):
    # 1. Compute VLA kwargs (model-type-specific)
    kwargs = ...  # OPENPI → {"mode": mode}, etc.

    # 2. If agentloop is active, route through it
    if self.agentloop is not None:
        actions, result = self.agentloop.run_step(obs, mode, vla_kwargs=kwargs)
        result["expert_label_flag"] = False
        return actions, result

    # 3. Otherwise, standard VLA-only path
    ...
```

The agentloop uses the **same kwargs** as the standard path — no divergence.

### 3. DualSystemAgentLoop.run_step()

```
obs + task_description (+ memory) → VLM → raw_output
                                          ↓
                               parse subtask (+ updated memory)
                                          ↓
                         obs + subtask → VLA → actions
```

- **Memoryless** (`enable_memory: False`): VLM outputs a plain subtask string.
- **Memory-enabled** (`enable_memory: True`): VLM outputs JSON `{"subtask": "...", "memory": "..."}`.
  Follows the MEM paper formulation: `πHL(lt+1, mt+1 | ot, mt, g)`.

### 4. Memory Module

When memory is enabled:
- `self._memories: list[str]` — one string per environment in the batch.
- Each step: the VLM receives the previous memory `mt` in the prompt, and outputs
  an updated memory `mt+1` alongside the subtask.
- **Epoch reset**: `reset_memory()` clears all memories at the start of each eval epoch.
- **Auto-reset alignment**: When the Behavior env auto-resets a done environment,
  `dones` is propagated from `env_worker` → channel → rollout worker.
  `reset_memory_for_envs(dones)` clears memory only for the done batch indices,
  preventing memory from a finished episode from leaking into the next one.

### 5. Env Auto-Reset and Memory Alignment

The Behavior env has `auto_reset=True` in eval mode. When an episode terminates
(e.g., after 20 of 100 steps), the env resets that specific environment in the
batch and continues. The key flow:

1. `BehaviorEnv.chunk_step()` detects `dones`, calls `_handle_auto_reset()`.
2. `env_evaluate_step()` now includes `dones` in the `EnvOutput`.
3. `env_worker.evaluate()` sends `dones` over the channel alongside `obs`/`final_obs`.
4. `MultiStepRolloutWorker.evaluate()` receives `dones` and calls
   `agentloop.reset_memory_for_envs(dones)` **before** the next `predict()` call,
   so the VLM starts with clean memory for the new episode.

This ensures memory is strictly per-episode and per-environment — no cross-contamination.

### 6. Trajectory Logging

`DualSystemEvalRunner` logs to `{log_path}/trajectories/`:
- `trajectory.jsonl`: one JSON line per step with structured VLM/VLA I/O (see below).
- `images/w{worker}_s{step:04d}.jpg`: observation images for selected steps.
- `k_images` config: number of images per trajectory (`-1`=all, `0`=none, default `10`).

Each JSONL record contains:
```json
{
  "worker": 0, "step": 3,
  "vlm_inputs":  {"task_descriptions": [...], "input_memories": [...], "image_path": "..."},
  "vlm_outputs": {"raw_outputs": [...], "subtasks": [...], "output_memories": [...]},
  "vla_inputs":  {"subtasks": [...]},
  "vla_outputs": {"actions": [[...]]}
}
```

## Config Reference

```yaml
vlm:
  log_subtasks: True
  enable_memory: False          # Toggle MEM-style language memory
  k_images: 10                  # Images saved per trajectory (-1=all)

  sampling_params:              # VLM generation params
    do_sample: True
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    repetition_penalty: 1.0
    max_new_tokens: 512

  model:
    model_type: "qwen2.5_vl_embodied"   # or "qwen3_vl_embodied"
    model_path: /path/to/model
    precision: "bf16"
    is_lora: False
    min_pixels: 3136
    max_pixels: 12845056
```

## Running

```bash
# Standard eval (non-agentic)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05

# Dual-system agentic eval (auto-detects "agentic" in config name)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic
```

## VLM Models

| Model | Config `model_type` | Class | Notes |
|-------|-------------------|-------|-------|
| Qwen2.5-VL | `qwen2.5_vl_embodied` | `Qwen2_5_VLPolicy` | Uses `process_vision_info` from `qwen_vl_utils`. Two-step inference. |
| Qwen3-VL-Thinking | `qwen3_vl_embodied` | `Qwen3_VLPolicy` | Thinking always on. One-step `apply_chat_template(tokenize=True)`. `<think>` tokens stripped. Requires `transformers>=4.57`. |

## Prompt Design

### Memoryless (default)
```
Task: {task_description}. Based on the image, describe the immediate
next subtask for the robot arm in one sentence.
```

### Memory-enabled
```
You are a robot task planner with a persistent memory.

Goal: {task_description}

Previous memory (summary of what has happened so far):
{memory}

Based on the current image observation and your previous memory:
1. Decide the immediate next subtask for the robot arm (one concise sentence).
2. Update the memory to incorporate what you observe now. Keep only information
   that is relevant for completing the remaining goal.

Respond with ONLY a JSON object:
{"subtask": "<next subtask>", "memory": "<updated memory>"}
```

Output is parsed by `parse_subtask_and_memory()` in `prompts.py` with robust
fallback handling (markdown fences, regex extraction, graceful degradation).

---

## BEHAVIOR-1K Evaluation

RLinf supports comprehensive evaluation on the BEHAVIOR-1K benchmark with multiple approaches:

### Evaluation Modes

1. **Standalone VLA** (`behavior_ppo_openpi_pi05.yaml`)
   - End-to-end policy: obs + task description → actions directly
   - No subtask planning; single VLA model
   - Faster inference, lower interpretability
   - [Detailed guide](behavior_doc/vla_eval.md)

2. **VLM+VLA (Dual-System)** (`behavior_ppo_openpi_agentic.yaml`)
   - Two-stage pipeline: VLM generates subtasks, VLA executes them
   - Optional memory module (MEM formulation) for multi-step reasoning
   - Slower inference, higher interpretability, trajectory logging
   - [Detailed guide](behavior_doc/vlm_vla_eval.md)

### Configuration & Scene Initialization

- **50 supported tasks** (turning_on_radio, rearranging_kitchen_furniture, etc.)
- **3 initialization modes**: fixed (disabled), offline-sampled (cached), online (live BDDL)
- **Download or generate** cached instances for offline evaluation
- [Full configuration reference](behavior_doc/behavior_config.md)

