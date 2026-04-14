# BEHAVIOR-1K Evaluation Documentation

This directory contains comprehensive documentation for evaluating Vision-Language models on the BEHAVIOR-1K household manipulation benchmark using RLinf.

## Quick Start

Choose your evaluation approach:

1. **Standalone VLA** — Direct policy learning from images and task descriptions
   - Config: `behavior_ppo_openpi_pi05.yaml`
   - Single model, fast inference, end-to-end learning
   - [Read guide →](vla_eval.md)

2. **VLM+VLA (Dual-System)** — Hierarchical planning with subtask generation
   - Config: `behavior_ppo_openpi_agentic.yaml`
   - Two-stage pipeline: VLM generates subtasks, VLA executes them
   - Optional language memory for multi-step reasoning
   - [Read guide →](vlm_vla_eval.md)

## Documentation Structure

### Core Guides

| Document | Contents |
|----------|----------|
| [vla_eval.md](vla_eval.md) | **Standalone VLA Evaluation**<br/>End-to-end policy learning, execution flow, task switching, scene initialization |
| [vlm_vla_eval.md](vlm_vla_eval.md) | **VLM+VLA Evaluation**<br/>Dual-system architecture, memory module, trajectory logging, VLM models |
| [behavior_config.md](behavior_config.md) | **Configuration Reference**<br/>All 50 BEHAVIOR-1K tasks, scene initialization modes, YAML configuration items |

### Key Topics

**Switching Tasks**
- [VLA Guide: Switching Tasks](vla_eval.md#switching-tasks)
- [VLM+VLA Guide: Switching Tasks](vlm_vla_eval.md#switching-tasks)

**Scene Initialization**
- [Disabled Mode (fixed scenes)](behavior_config.md#mode-1-disabled--fixed-instance)
- [Offline Mode (cached scenes)](behavior_config.md#mode-2-offline--random-sampled-from-cache)
- [Online Mode (live BDDL sampling)](behavior_config.md#mode-3-online--live-bddl-sampling)
- [Downloading & Generating Instances](behavior_config.md#downloading-and-generating-initialization-samples)

**Configuration**
- [All Configuration Fields](behavior_config.md#configuration-reference)
- [Example: Multi-Task Evaluation](behavior_config.md#example-multi-task-offline-evaluation)

## Launch Commands

### Standalone VLA (Pi0.5)

```bash
# Default: turning_on_radio, fixed initialization
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05

# Different task (via CLI override)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture

# Offline-sampled scenes
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/
```

### VLM+VLA (Dual-System with Qwen2.5-VL)

```bash
# Default: with memory enabled
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic

# Different task
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=picking_up_trash

# Disable memory (memoryless VLM mode)
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.enable_memory=False

# Switch to Qwen3-VL-Thinking
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.model.model_type=qwen3_vl_embodied \
  vlm.model.model_path=/path/to/Qwen3-VL-4B-Thinking
```

## Setup

### Prerequisites

```bash
# Set required environment variables
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

# Create Isaac Sim installation (if needed)
curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip \
  -o isaac-sim.zip && unzip isaac-sim.zip

# Download BEHAVIOR assets
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"

# Download challenge task instances (optional, for offline evaluation)
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

### Model Downloads

**OpenPI pi0.5** (for both VLA and VLM+VLA):
```bash
huggingface-hub download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior
```

**Qwen2.5-VL** (for VLM+VLA):
```bash
# Download from Hugging Face Hub
huggingface-hub download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL
```

**Qwen3-VL-Thinking** (optional, for VLM+VLA):
```bash
# Download from Hugging Face Hub
huggingface-hub download Qwen/Qwen3-VL-4B-Thinking --local-dir Qwen3-VL-4B-Thinking
```

## File Structure

```
behavior/
├── README.md                 # This file
├── vla_eval.md              # Standalone VLA evaluation guide
├── vlm_vla_eval.md          # VLM+VLA evaluation guide
└── behavior_config.md       # Complete configuration reference
```

## Common Tasks

### Task 0: Turning On Radio
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=turning_on_radio
```

### Task 8: Rearranging Kitchen Furniture
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

### Task 20: Sorting Vegetables
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=sorting_vegetables
```

See [behavior_config.md](behavior_config.md) for all 50 tasks.

## Troubleshooting

**OmniGibson Timeout Issues**
→ See [vla_eval.md: Troubleshooting](vla_eval.md#troubleshooting)

**VLM Out of Memory**
→ See [vlm_vla_eval.md: Troubleshooting](vlm_vla_eval.md#troubleshooting)

**Task Not Found**
→ Verify task name in [behavior_config.md](behavior_config.md#supported-tasks-049)

## Architecture Comparison

| Aspect | VLA | VLM+VLA |
|--------|-----|---------|
| **Models** | 1 (OpenPI) | 2 (Qwen VLM + OpenPI) |
| **Speed** | Fast (~100 ms/step) | Slower (~500 ms/step) |
| **Interpretability** | Low | High (subtasks logged) |
| **Memory** | N/A | Optional MEM module |
| **Trajectory Logging** | Metrics only | Full I/O + images |
| **Batch Inference** | Single VLA call | VLM supports per-sample prompts |

## References

- **RLinf Framework**: See [CLAUDE.md](../CLAUDE.md) for dual-system architecture overview
- **BEHAVIOR-1K**: https://behavior.stanford.edu/
- **OpenPI**: https://huggingface.co/openvla/openvla-7b
- **Qwen VLMs**: https://huggingface.co/Qwen

## Contributing

When updating documentation:
1. Keep guides separate by evaluation approach (vla_eval.md vs vlm_vla_eval.md)
2. Maintain configuration reference in behavior_config.md (don't duplicate elsewhere)
3. Include command examples for all major workflows
4. Reference related sections with markdown links

---

**Quick Links**: [VLA Guide](vla_eval.md) | [VLM+VLA Guide](vlm_vla_eval.md) | [Config Reference](behavior_config.md)
