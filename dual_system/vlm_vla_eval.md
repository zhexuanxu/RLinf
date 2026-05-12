# VLM+VLA（双系统）BEHAVIOR 评估指南

本指南介绍 BEHAVIOR-1K 任务上的**双系统（VLM + VLA）评估**。在该方案中，高层 VLM 生成中间子任务，底层 VLA 将子任务转化为精确的机器人动作。

## 概述

**双系统**方案结合了：
1. **VLM**（视觉语言模型）：高层规划 -- 解读任务和观测，生成子任务
2. **VLA**（视觉语言动作模型）：底层控制 -- 将子任务执行为连续机器人动作

```
┌──────────────────────────────────┐
│  观测 + 任务描述                  │
│  （+ 可选记忆）                   │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  VLM（Qwen2.5-VL）               │
│  ├─ 生成子任务                   │
│  └─ 更新记忆（如已启用）          │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  观测 + 子任务                    │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  VLA（OpenPI pi0.5）              │
│  └─ 生成动作                     │
└──────────────────────────────────┘
              │
              ▼
┌──────────────────────────────────┐
│  动作（23 维，32 chunks）         │
└──────────────────────────────────┘
```

## 执行流程

### 1. 配置

VLM+VLA 的配置文件为 **`behavior_ppo_openpi_agentic.yaml`**。关键特性：

```yaml
vlm:
  enable_memory: True        # MEM 风格的语言记忆
  k_images: -1              # 保存所有轨迹图像
  
  model:
    model_type: "qwen2.5_vl_embodied"  # 或 "qwen3_vl_embodied"
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

配置中出现 `vlm:` 部分会触发：
- 使用 `DualSystemEvalRunner`（而非 `EmbodiedEvalRunner`）
- 在 rollout worker 中初始化 `DualSystemAgentLoop`
- 完整的轨迹日志记录（含 VLM 输入输出和图像）

### 2. Worker 初始化

**RolloutWorker**：
```python
if cfg.get("vlm"):
    self._init_vlm()  # 加载 VLM + 创建 agentloop
```

该过程：
1. 在 rollout GPU 上加载 VLM（Qwen2.5-VL 或 Qwen3-VL）
2. 创建 `DualSystemAgentLoop(vlm_model, vla_model, enable_memory=True/False)`
3. 初始化每个环境的记忆状态（若 `enable_memory=True`）

### 3. 评估流程

#### EnvWorker 流程
与独立 VLA 相同：
```
每一步：
  1. BehaviorEnv.chunk_step(actions)
     - 执行 32 个子步（1 个 chunk）
     - 收集 obs、rewards、dones
  2. 追踪指标
  3. 将 obs + dones 发送给 RolloutWorker
  4. 从 RolloutWorker 接收 actions
```

#### RolloutWorker 流程（双系统）
```
每一步：
  1. 从 EnvWorker 接收 obs
  2. 调用 agentloop.run_step(obs, mode="eval")
     a. 第一轮 -- VLM：
        - 输入：obs + task_desc（+ 记忆）
        - 输出：子任务（+ 更新后的记忆）
     b. 第二轮 -- VLA：
        - 输入：obs + 子任务
        - 输出：actions
  3. 将 actions 发回 EnvWorker
  4. 记录轨迹（VLM 输入输出、VLA 输入输出、图像）
```

#### 记忆对齐（自动重置）

当 `env.eval.auto_reset=True`（评估时默认启用）时：
```
EnvWorker                    RolloutWorker
     │                              │
     ├─ 检测 dones                 │
     ├─ 自动重置已完成的环境        │
     ├─ 发送 obs + dones ─────────►│
     │                        agentloop.reset_memory_for_envs(dones)
     │◄─────── actions ────────────┤
```

这确保记忆严格按 episode 隔离：已结束 episode 的旧记忆不会泄漏到新 episode 中。

#### 记忆状态

若 `enable_memory=True`：
```python
self._memories: list[str]  # 每个环境一条记忆，跨步累积

每次重置时：
  agentloop.reset_memory() 或 reset_memory_for_envs(dones)
  # 为新 episode 清空记忆
```

### 4. 核心代码路径

**入口**：`examples/embodiment/eval_embodied_agent.py`
```python
runner_cls = DualSystemEvalRunner if cfg.get("vlm") else EmbodiedEvalRunner
runner = runner_cls(cfg=cfg, rollout=rollout_group, env=env_group)
# -> 使用 DualSystemEvalRunner
```

**Runner**：`rlinf/agents/dualsystem/eval_runner.py::DualSystemEvalRunner`
- 继承自 `EmbodiedEvalRunner`
- 新增完整轨迹日志：`.jsonl` 文件 + `.jpg` 图像
- `_log_trajectories()`：逐步将 VLM/VLA 输入输出写入磁盘

**Agentloop**：`rlinf/agents/dualsystem/dual_system_agent_loop.py::DualSystemAgentLoop`
```python
def run_step(obs, mode="eval", vla_kwargs=None):
    # 第一轮：VLM 生成子任务（+ 记忆）
    raw_outputs = vlm_model.generate_subtask(obs, prompt=prompts)
    subtasks, memories = parse_subtask_and_memory(raw_outputs)
    
    # 第二轮：VLA 生成动作
    actions, vla_result = vla_model.predict_action_batch(obs_with_subtask)
    
    return actions, result_dict
```

**VLM 模型**：
- `rlinf/models/embodiment/VLM/qwen2_5_vl_policy.py::Qwen2_5_VLPolicy`
  - 使用 `qwen_vl_utils.process_vision_info` 正确处理图像
  - 支持批量推理，每个样本可使用不同 prompt（适配不同记忆内容）
- `rlinf/models/embodiment/VLM/qwen3_vl_policy.py::Qwen3_VLPolicy`
  - 始终启用思考模式；解析前会移除 `<think>` token

**Prompt 模板**：`rlinf/agents/dualsystem/prompts.py`
- `DEFAULT_VLM_PROMPT`：任务 -> 子任务（无记忆模式）
- `MEMORY_VLM_PROMPT`：任务 + 记忆 -> 子任务 + 记忆（MEM 模式）
- `parse_subtask_and_memory()`：鲁棒的 JSON 解析，带回退机制

## 启动评估

### 基本命令

```bash
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic
```

默认使用 "turning_on_radio" 任务、固定初始化，并启用记忆模块。

### 切换任务

与独立 VLA 相同，但 VLM 也需要理解新任务：

**方式 1：CLI 覆盖**
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

**方式 2：自定义配置 YAML**
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

然后执行：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic_task8
```

### 切换场景初始化

```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/cached_instances/ \
  env.eval.omni_config.task.instance_file_format=tro_state
```

### 开关记忆模块

禁用记忆（无记忆 VLM 模式）：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.enable_memory=False
```

这会将 VLM prompt 切换为 `DEFAULT_VLM_PROMPT`（无记忆输入/输出）。

### 切换 VLM 模型

使用 Qwen3-VL-Thinking 替代 Qwen2.5-VL：
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.model.model_type=qwen3_vl_embodied \
  vlm.model.model_path=/path/to/Qwen3-VL-4B-Thinking
```

## 轨迹日志

与独立 VLA 不同，双系统方案会记录完整轨迹：

### 输出结构

```
{log_path}/trajectories/
├── trajectory.jsonl     # 每步一行 JSON
└── images/
    ├── w0_s0000.jpg     # worker 0, step 0
    ├── w0_s0001.jpg
    └── ...
```

### JSONL 记录格式

每行为一个 JSON 对象：
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

### 控制图像保存

```yaml
vlm:
  k_images: 10     # 每条轨迹均匀采样保存 10 步
  k_images: -1     # 保存所有步
  k_images: 0      # 不保存图像
```

## 预期输出

### 控制台日志

```
eval/success_once: 0.75
eval/episode_len: 3456
eval/num_trajectories: 8
Trajectory logged to {log_path}/trajectories/trajectory.jsonl (2 workers)
```

### 轨迹文件

```
ls -la {log_path}/trajectories/
trajectory.jsonl (1000+ 行)
images/w0_s0000.jpg
images/w0_s0001.jpg
...
images/w1_s0999.jpg
```

## 记忆模块（MEM 公式化）

当 `enable_memory=True` 时，VLM 实现 MEM（记忆增强规划器）公式化：

```
π_HL(l_{t+1}, m_{t+1} | o_t, m_t, g)

其中：
  o_t = 时刻 t 的观测
  m_t = 时刻 t 的记忆（压缩历史）
  g   = 目标（任务描述）
  l_t = 子任务（高层动作）
  π_HL = 高层策略（VLM）
```

**每一步的执行流程**：
1. VLM 接收：任务 + 上一步记忆 + 当前观测
2. VLM 输出：下一个子任务 + 更新后的记忆
3. 记忆跨步累积（类似对话历史）
4. 当 episode 结束或重置时，记忆被清空

**记忆更新策略**（来自 prompt）：
> 仅保留与完成剩余目标相关的信息。

这鼓励 VLM 压缩和总结进展，而非简单地逐步拼接日志。

## 性能优化建议

1. **记忆上下文**：启用 `enable_memory=True` 后，每步 VLM 的输入更长（包含累积记忆）。需监控 VLM 推理速度。
2. **VLM 模型大小**：Qwen2.5-VL-3B 速度快；Qwen3-VL-4B-Thinking 速度较慢但能力可能更强。
3. **批量 prompt**：VLM API 接受 `prompt: str | list[str]`。当各环境的记忆不同时，所有 B 个环境在一次批量调用中处理（而非 B 次独立调用）。
4. **采样参数**：评估时降低 `temperature`（默认 0.6），使子任务生成更确定性。

## 与独立 VLA 的对比

| 方面 | 独立 VLA | VLM+VLA |
|------|----------|---------|
| **规划能力** | 无（端到端） | VLM 生成高层子任务 |
| **模型数量** | 1（仅 VLA） | 2（VLM + VLA） |
| **推理速度** | 约 100 ms/步 | 约 500 ms/步（VLM 开销） |
| **可解释性** | 低 | 高（子任务有日志记录） |
| **轨迹日志** | 仅指标 | 完整输入输出 + 图像 |
| **记忆** | 无 | 可选（MEM 公式化） |
| **配置区别** | 无 `vlm:` 部分 | 有 `vlm:` 部分 |

## 故障排查

### VLM 崩溃或内存溢出

1. 减小批量大小：`env.eval.total_num_envs=1`
2. 降低图像分辨率：`vlm.model.max_pixels=9437184`（原为 12845056）
3. 使用更轻量的 VLM：`vlm.model.model_type=qwen2.5_vl_embodied`（3B 对比 4B）

### 子任务质量不佳

1. 将 `vlm.sampling_params.temperature` 提高到 1.0 以获得更多样化的输出
2. 增大 `max_new_tokens`（默认 512），防止子任务被截断
3. 检查 `vlm.model.model_path` 是否指向正确的 checkpoint

### 记忆未更新

检查日志中是否有：
```
[DualSystem] Memory: <记忆文本>
```

若记忆始终为空 `(no memory yet)`，请检查：
1. `vlm.enable_memory=True`
2. VLM 是否输出了包含 `{"subtask": "...", "memory": "..."}` 的有效 JSON
3. `prompts.py` 中 `parse_subtask_and_memory()` 的回退解析逻辑

### 轨迹文件损坏

若 `trajectory.jsonl` 不完整或被截断：
1. 检查磁盘空间
2. 确保没有其他进程写入同一日志目录
3. 验证写权限：`chmod 755 {log_path}/trajectories/`

## 相关文档

- [behavior_config.md](behavior_config.md) -- BEHAVIOR 任务与场景配置
- [vla_eval.md](vla_eval.md) -- 独立 VLA 方案（用于对比）
- [data.md](data.md) -- 数据参考手册
