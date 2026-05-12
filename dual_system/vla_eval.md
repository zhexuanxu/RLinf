# 独立 VLA BEHAVIOR 评估指南

本指南介绍 BEHAVIOR-1K 任务上的**独立 VLA（视觉语言动作模型）评估**。在该方案中，策略直接接收观测和任务描述，生成机器人动作，不涉及高层子任务规划。

## 概述

**独立 VLA** 方案仅使用底层动作模型（OpenPI pi0.5）：

```
任务描述（文本）  ┐
                  ├─► VLA（pi0.5）──► 动作（23 维，32 chunks）
观测（图像）     ┘
```

任务描述（例如 "Turn on the radio receiver that's on the table in the living room"）直接传递给 VLA，VLA 必须同时理解语言指令和视觉观测来生成合适的动作。

## 执行流程

### 1. 配置

独立 VLA 的配置文件为 **`behavior_ppo_openpi_pi05.yaml`**。该配置：
- **不包含** `vlm:` 部分（无 VLM 模型）
- 使用 `EmbodiedEvalRunner`（而非 `DualSystemEvalRunner`）
- 直接调用 `predict()` -> `vla_model.predict_action_batch()`

关键配置项：
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

配置中没有 `vlm:` 部分。

### 2. 评估流程

评估在两个主要进程中运行：**EnvWorker** 和 **RolloutWorker**。

#### EnvWorker 流程
```
┌─ env_worker.evaluate() ─────────────────────────┐
│                                                 │
│  每一步：                                        │
│    1. BehaviorEnv.chunk_step(actions)           │
│       - 执行 32 个子步（1 个 chunk）             │
│       - 收集 obs、rewards、dones                │
│    2. 追踪指标（成功率、回报等）                  │
│    3. 将 obs + dones 发送给 RolloutWorker        │
│    4. 从 RolloutWorker 接收 actions              │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### RolloutWorker 流程
```
┌─ rollout.evaluate() ───────────────────────────┐
│                                                │
│  1. 加载 VLA 模型（pi0.5）                      │
│  2. 每一步：                                    │
│     a. 从 EnvWorker 接收 obs                    │
│     b. 提取图像、状态、任务描述                  │
│     c. 调用 vla.predict_action_batch()          │
│        obs → [动作张量]                         │
│     d. 将 actions 发回 EnvWorker                │
│                                                │
└────────────────────────────────────────────────┘
```

#### 指标收集
- **上升沿检测**：仅在 episode 从"运行中"转为"完成"时收集指标（并非每步都收集）
- **逐环境追踪**：对每个环境独立追踪成功率、episode 长度、回报
- **聚合**：所有 env worker 的结果由 `compute_evaluate_metrics()` 汇总

### 3. 核心代码路径

**入口**：`examples/embodiment/eval_embodied_agent.py`
```python
runner_cls = DualSystemEvalRunner if cfg.get("vlm") else EmbodiedEvalRunner
runner = runner_cls(cfg=cfg, rollout=rollout_group, env=env_group)
```
-> 由于配置中无 `vlm:` 部分，使用 `EmbodiedEvalRunner`。

**Runner**：`rlinf/runners/embodied_eval_runner.py::EmbodiedEvalRunner`
- `evaluate()`：启动 env 和 rollout worker，等待结果，汇总指标
- `run()`：调用 `evaluate()` 一次并记录结果

**环境**：`rlinf/envs/behavior/behavior_env.py::BehaviorEnv`
- `chunk_step()`：连续执行 32 步（chunk size），追踪步数计数，在 `max_episode_steps` 处强制截断
- `_record_metrics()`：按环境累积回报和成功标志

**Rollout**：`rlinf/workers/rollout/hf/huggingface_worker.py::MultiStepRolloutWorker`
- `predict()`：将 obs 路由到 `vla_model.predict_action_batch(obs, mode="eval")`
  - 无 `vlm:` 部分时，agentloop 为 `None`，跳过双系统路径
- `evaluate()`：评估期间反复调用 `predict()`，收集轨迹

## 启动评估

### 基本命令

```bash
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05
```

默认使用 `behavior_r1pro.yaml` 中的任务配置，即 "turning_on_radio"（固定初始化）。

### 切换任务

要在不同任务上评估，可通过 YAML 或 CLI 覆盖任务名称：

**方式 1：修改配置 YAML**

编辑 `examples/embodiment/config/env/behavior_r1pro.yaml`：
```yaml
omni_config:
  task:
    activity_name: rearranging_kitchen_furniture  # 改为任意任务
    activity_definition_id: 0
```

然后执行：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05
```

**方式 2：Hydra CLI 覆盖**（推荐）

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

**方式 3：创建自定义配置**

将 `behavior_ppo_openpi_pi05.yaml` 复制为 `behavior_ppo_openpi_pi05_task8.yaml` 并编辑：
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

然后执行：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_task8
```

### 切换场景初始化

使用 offline 采样场景替代默认固定场景：

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/cached_instances/ \
  env.eval.omni_config.task.instance_file_format=tro_state
```

或使用 online 采样：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=online \
  env.eval.omni_config.task.online_object_sampling=True \
  env.eval.omni_config.task.use_presampled_robot_pose=False
```

详见 [behavior_config.md](behavior_config.md) 了解场景初始化的完整文档。

## 预期输出

### 指标

评估日志输出：
```
eval/success_once: 0.5   （成功率）
eval/episode_len: 3456   （截断前的平均 episode 长度）
eval/num_trajectories: 8 （已完成的 episode 总数）
```

输出到 TensorBoard 和日志文件。

### 视频

若 `env.eval.video_cfg.save_video: True`，视频保存到：
```
{log_path}/video/eval/
```

### 无轨迹日志

与 VLM+VLA 路径不同，独立 VLA 评估**不会**记录详细轨迹（`trajectory.jsonl`）。如需添加此功能，需要：
1. 扩展 `EmbodiedEvalRunner`，或
2. 切换到 `DualSystemEvalRunner` 并配置 `vlm:` 部分

## 性能优化建议

1. **Episode 长度**：将 `max_episode_steps` 和 `max_steps_per_rollout_epoch` 一起增大（pi0.5 默认 4096）。
2. **任务多样性**：使用 `instance_resample_mode: offline` 配合缓存实例，无需 online 采样开销即可实现多样化评估。
3. **批量大小**：增大 `env.eval.total_num_envs` 以并行评估更多环境（受 GPU 显存限制）。
4. **场景优化**：将 `load_room_types` 限制为仅必需的房间（如 `["kitchen"]`），加快初始化速度。

## 与 VLM+VLA 的对比

| 方面 | 独立 VLA | VLM+VLA |
|------|----------|---------|
| **输入语言** | 仅任务描述 | 任务描述 + 可选记忆 |
| **规划能力** | 无（端到端） | VLM 生成高层子任务 |
| **模型数量** | 1（仅 VLA） | 2（VLM + VLA） |
| **推理速度** | 更快 | 较慢（VLM 开销） |
| **可解释性** | 低（端到端） | 较高（子任务可见） |
| **轨迹日志** | 基础（仅指标） | 完整（VLM 输入输出、图像） |
| **配置区别** | 无 `vlm:` 部分 | 有 `vlm:` 部分 |
| **Runner** | `EmbodiedEvalRunner` | `DualSystemEvalRunner` |

## 故障排查

### OmniGibson 超时问题

若 episode 在 `max_episode_steps` 前被截断，请检查：
1. RLinf 是否正确覆盖了 OmniGibson 的超时设置
2. BehaviorEnv 的步数计数器是否正常追踪

解决方法：在日志中验证所有 episode 是否达到预期的最大步数。

### 任务描述缺失

若 VLA 收到空的任务描述，请确认：
1. `behavior_task.jsonl` 存在于 `rlinf/envs/behavior/` 中
2. `omni_config.task.activity_name` 为有效的任务名称

### 成功率低

常见原因：
1. 任务对当前 pi0.5 checkpoint 来说过于困难
2. 场景初始化与训练分布不匹配
3. Episode 长度不足以完成任务

尝试：
- 使用 `instance_resample_mode: offline` 加载熟悉的场景
- 增大 `max_episode_steps`
- 验证 checkpoint 是否在相同任务上训练过

## 相关文档

- [behavior_config.md](behavior_config.md) -- BEHAVIOR 任务与场景初始化的完整配置参考
- [vlm_vla_eval.md](vlm_vla_eval.md) -- 带子任务规划的 VLM+VLA 方案
- [data.md](data.md) -- 数据参考手册
