# RLinf Pi0.5 + 双系统具身智能框架

## 项目概览

RLinf 是一个将强化学习（RL）应用到具身智能（Embodied AI）的分布式框架。本仓库 `RLinf_pi05` 在 RLinf 基础上实现了以下核心能力：

- **满血版 pi0.5 模型**：在 Physical Intelligence 的 pi0/pi0.5 基础上新增 CoT 推理文本输出 + action 输出，支持 VLM CE loss 训练和 KV cache 高效推理
- **双系统评估架构**：高层 VLM（Qwen2.5-VL / Qwen3-VL-Thinking）生成 subtask 指令，底层 VLA（pi0.5）执行连续 action，形成"先想再做"的层次化决策

```
Env obs --> VLM (Qwen) --> subtask (text) --> VLA (pi0.5) --> actions --> Env
                |                                                  |
                +-- memory (可选) ----> 下一步 VLM 输入 <----------+
```

### 相关仓库

| 仓库 | 说明 |
|------|------|
| `/mnt/public/xzxuan/repos/openpi` | pi 官方仓库，残血版 pi0.5（仅 flow matching，无 CoT） |
| `/mnt/public/xzxuan/repos/vla_lib` | 第三方完整实现，包含 CoT + action + KV cache |
| `/mnt/public/xzxuan/repos/RLinf_pi05` | **本仓库**，在 RLinf 框架中实现完整 pi0.5 + 双系统评估 |

---

## 核心架构

### 组件布局

标准 `actor, env, rollout` 分布式部署。VLM 和 VLA 均运行在 rollout GPU 上（`MultiStepRolloutWorker` 内部），无需额外 worker 或 GPU 组。

```yaml
cluster:
  component_placement:
    actor: 0-1
    env,rollout: 0-1
```

### 核心文件索引

**pi0.5 模型子系统**

| 文件 | 说明 |
|------|------|
| `rlinf/models/embodiment/openpi/openpi_full_pi05_model.py` | 满血版 pi0.5 模型核心，含 `sft_forward()` / `generate_language()` / `sample_with_reasoning()` |
| `rlinf/models/embodiment/openpi/static_kv_cache.py` | StaticKVCache + left_to_right_align |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 父类（残血版），新增 config 字段 |
| `rlinf/models/embodiment/openpi/__init__.py` | `get_model()` 路由逻辑 |
| `rlinf/models/embodiment/openpi/dataconfig/cot_transform.py` | CoT 数据变换（VLM+VLA 模式） |
| `rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py` | `BehaviorLeRobotDataset`，内联 omnigibson 依赖 |
| `rlinf/models/embodiment/openpi/dataconfig/behavior_b1k_dataconfig.py` | `LeRobotB1KDataConfig` 工厂（匹配 openpi-comet repack keys） |
| `rlinf/models/embodiment/openpi/dataconfig/behavior_data_loader.py` | `create_behavior_data_loader()`（chunk streaming + DistributedSampler） |
| `rlinf/models/embodiment/openpi/policies/behavior_policy.py` | `B1kInputs` / `B1kOutputs`（匹配 openpi-comet key 名） |
| `rlinf/models/embodiment/base_policy.py` | `ForwardType` 枚举（含 `GENERATE_LANGUAGE`） |

**双系统子系统**

| 文件 | 说明 |
|------|------|
| `rlinf/agents/dualsystem/dual_system_agent_loop.py` | 核心协调：VLM -> subtask -> VLA -> action，管理逐环境记忆状态 |
| `rlinf/agents/dualsystem/prompts.py` | prompt 模板（无记忆/有记忆模式）及 JSON 输出解析器 |
| `rlinf/agents/dualsystem/eval_runner.py` | `DualSystemEvalRunner`：继承 `EmbodiedEvalRunner`，新增 JSONL+图片轨迹日志 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | `MultiStepRolloutWorker`：`_init_vlm()` 加载 VLM 并创建 agentloop；`predict()` 路由分发 |
| `rlinf/models/embodiment/VLM/qwen2_5_vl_policy.py` | Qwen2.5-VL 封装 |
| `rlinf/models/embodiment/VLM/qwen3_vl_policy.py` | Qwen3-VL-Thinking 封装（thinking 始终开启，`<think>` token 被剥离） |
| `rlinf/models/embodiment/VLM/__init__.py` | 工厂分发 `qwen2.5_vl_embodied` / `qwen3_vl_embodied` |

**SFT / PPO 训练**

| 文件 | 说明 |
|------|------|
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | 统一 SFT worker（处理 VLA / VLM / VLM+VLA 训练 + eval） |
| `rlinf/workers/sft/fsdp_vlm_sft_worker.py` | VLM-only SFT worker（BEHAVIOR skill prediction） |
| `rlinf/runners/sft_runner.py` | SFT runner，含 `val_at_step_0` 训练前 baseline eval 钩子 |
| `rlinf/data/embodied_io_struct.py` | 扩展 `ChunkStepResult` / `Trajectory` / `EmbodiedRolloutResult`，新增 `subtasks` 字段 |

**配置文件**

| 文件 | 说明 |
|------|------|
| `examples/sft/config/behavior_pi05_vla.yaml` | BEHAVIOR VLA-only SFT |
| `examples/sft/config/behavior_pi05_vlm_sft.yaml` | BEHAVIOR skill prediction VLM-only SFT（pi0.5） |
| `examples/sft/config/behavior_qwen2_5_vlm_sft.yaml` | BEHAVIOR skill prediction VLM-only SFT（Qwen2.5） |
| `examples/embodiment/config/behavior_ppo_openpi_agentic.yaml` | 双系统 agentic eval 配置 |

---

## Pi0.5 模型

### pi0 vs pi0.5 核心区别

| 特性 | pi0 | pi0.5 (残血版) | pi0.5 (满血版) |
|------|-----|---------------|---------------|
| State 输入 | 连续向量 (suffix) | 离散化到 token (prefix) | 同残血版 |
| 时间注入 | concat + MLP | adaRMSNorm | 同残血版 |
| Action 输出 | Flow matching | Flow matching | Flow matching |
| CoT 文本输出 | - | - | 支持 |
| VLM CE loss | - | - | 支持 |
| KV cache 推理 | - | - | 支持 |

满血版核心增量：VLM 先输出一段 CoT 推理文本（subtask），然后 action expert 再基于该上下文输出 action。

### 配置开关

```yaml
openpi:
  full_pi05: true      # true=满血版, false=残血版
  forward_mode: "vla"   # "vla" | "vlm" | "vlm_vla"
```

### 三种训练模式

模式自动检测逻辑（`sft_forward` 内部）：

| forward_mode | 数据源 | 监督信号 | 自动检测条件 |
|-------------|--------|---------|-------------|
| `"vlm"` | BEHAVIOR skill annotation | 文本 (CE loss) | `actions=None` |
| `"vla"` | LeRobot (BEHAVIOR demos) | action (flow matching) | `actions` + `loss_mask` 全 False |
| `"vlm_vla"` | LeRobot + CoT | 文本 + action | `actions` + `loss_mask` 有 True |

统一数据格式（所有模式均产生 `(observation, actions)` 二元组）：

```python
observation = {
    "image": {"image_0": [H, W, 3]},       # float32 [-1, 1]
    "image_mask": {"image_0": True/False},
    "state": [state_dim],                    # float32
    "tokenized_prompt": [max_token_len],     # int32
    "tokenized_prompt_mask": [max_token_len], # bool
    "token_ar_mask": [max_token_len],        # int32 (0=双向, 1=因果)
    "token_loss_mask": [max_token_len],      # bool (True=计算CE loss)
    "token_kv_cache_mask": [max_token_len],  # bool (True=在action KV cache中)
}
actions = Tensor[action_horizon, action_dim]  # VLA/VLM+VLA
       or None                                # VLM-only
```

### SFT 训练

**统一设计**

| 组件 | 统一入口 |
|------|---------|
| 启动脚本 | `examples/sft/run_vla_sft.sh <config_name>` |
| 训练 worker | `rlinf/workers/sft/fsdp_vla_sft_worker.py` |
| 模型 forward | `openpi_full_pi05_model.py: sft_forward()` |

**模式 1：VLM-only SFT（BEHAVIOR skill prediction）**

使用 BEHAVIOR 人类演示的 skill annotation 作为 VLM 训练数据，训练模型根据当前图像预测下一步 skill 描述。

```bash
bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_sft
```

配置文件：`examples/sft/config/behavior_pi05_vlm_sft.yaml`

关键参数：

```yaml
data:
  type: vlm
  dataset_name: "behavior_skill_pi05"
  train_data_paths: "/mnt/public/xzxuan/data/2025-challenge-demos"

actor:
  model:
    openpi:
      full_pi05: True
      forward_mode: "vlm"
      num_images_in_input: 1
```

注意事项：
- VLM-only 模式不需要 norm_stats/asset，可使用不带 asset 的 base 模型
- `gradient_checkpointing` 必须设为 `False`
- `precision` 必须设为 `null`
- Eval 流程详见 `pi05_doc/04-sft-eval.md`

**模式 2：VLA-only SFT（BEHAVIOR 数据）**

```bash
bash examples/sft/run_vla_sft.sh behavior_pi05_vla
```

使用前需先运行数据准备脚本，详见 `pi05_doc/05-behavior-vla-sft.md`。

如需在其他 LeRobot 数据上使用满血版：

```yaml
openpi:
  full_pi05: True
  forward_mode: "vla"    # VLA-only，与残血版行为一致
```

**模式 3：VLM+VLA SFT（未来扩展）**

需要带有 `cot_text` 标注的 LeRobot 数据集。

### PPO 训练（带 CoT 推理）

```bash
bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05_full
```

CoT 仅用于推理阶段（生成 subtask 文本后采样 action），PPO 只训练 action 部分，不训练文本。详细流程见 `PPO.md`。

### BEHAVIOR B1K 对齐

本仓库的 BEHAVIOR task-0000 SFT 管线已与 `openpi-comet` JAX 实现严格对齐。

| 项目 | 对齐前 | 对齐后 |
|------|--------|--------|
| 训练步数 | ~16,790 | 30,000 |
| LR 衰减终值 | 2.5e-6 | 0.0 |
| Norm stats | 标准 LeRobot 聚合 | openpi-comet 聚合（percentile of percentiles） |
| 数据集类 | 标准 `LeRobotDataset` | `BehaviorLeRobotDataset`（chunk streaming） |

Norm stats 生成：

```bash
.venv_pi/bin/python toolkits/behavior/compute_behavior_norm_stats.py
```

详见 `pi05_doc/06-behavior-b1k-alignment.md` 和 `pi05_doc/07-norm-stats.md`。

---

## 双系统评估（VLM + VLA）

### 工作原理

**Worker 初始化**：`MultiStepRolloutWorker.init_worker()` 先加载 VLA（pi0.5），若 config 中存在 `cfg.vlm`，则调用 `_init_vlm()` 加载 VLM 并创建 `DualSystemAgentLoop`。

**predict 路由**：当 `self.agentloop is not None` 时，`predict()` 将观测路由到 agentloop 而非直接调用 VLA。

**DualSystemAgentLoop.run_step() 流程**：

```
obs + task_description (+ memory) --> VLM --> raw_output
                                              |
                                   解析 subtask (+ 更新 memory)
                                              |
                         obs + subtask --> VLA --> actions
```

### 记忆模块

- **无记忆模式**（`enable_memory: False`）：VLM 输出纯文本 subtask
- **有记忆模式**（`enable_memory: True`）：VLM 输出 JSON `{"subtask": "...", "memory": "..."}`，遵循 MEM 论文公式 `piHL(lt+1, mt+1 | ot, mt, g)`
- **auto-reset 对齐**：当 BEHAVIOR 环境自动重置某个已完成的环境时，`dones` 信号从 env_worker 传播到 rollout worker，仅清除对应批次索引的记忆，防止跨 episode 记忆泄漏

### 轨迹日志

`DualSystemEvalRunner` 在 `{log_path}/trajectories/` 下生成：
- `trajectory.jsonl`：每步一条 JSON 记录，包含完整 VLM/VLA 输入输出
- `images/w{worker}_s{step:04d}.jpg`：选定步骤的观测图像
- `k_images` 配置控制每条轨迹保存的图像数（`-1`=全部，`0`=无，默认 `10`）

### VLM 配置参考

```yaml
vlm:
  log_subtasks: True
  enable_memory: False
  k_images: 10

  sampling_params:
    do_sample: True
    temperature: 1.0
    top_p: 0.95
    top_k: 20
    repetition_penalty: 1.0
    max_new_tokens: 512

  model:
    model_type: "qwen2.5_vl_embodied"   # 或 "qwen3_vl_embodied"
    model_path: /path/to/model
    precision: "bf16"
    is_lora: False
    min_pixels: 3136
    max_pixels: 12845056
```

### 支持的 VLM 模型

| 模型 | config `model_type` | 说明 |
|------|-------------------|------|
| Qwen2.5-VL | `qwen2.5_vl_embodied` | 使用 `process_vision_info` 两步推理；依赖 `qwen_vl_utils` |
| Qwen3-VL-Thinking | `qwen3_vl_embodied` | thinking 始终开启，`<think>` token 被剥离；需要 `transformers>=4.57` |

---

## BEHAVIOR-1K 评估

### 评估模式对比

| 方面 | 独立 VLA | VLM+VLA（双系统） |
|------|---------|-----------------|
| 配置 | `behavior_ppo_openpi_pi05.yaml` | `behavior_ppo_openpi_agentic.yaml` |
| 模型数量 | 1（OpenPI pi0.5） | 2（Qwen VLM + OpenPI pi0.5） |
| 推理速度 | 快（约 100 ms/步） | 较慢（约 500 ms/步） |
| 可解释性 | 低 | 高（subtask 有日志） |
| 记忆模块 | 无 | 可选 MEM 模块 |
| 轨迹日志 | 仅指标 | 完整输入输出 + 图像 |

### 启动命令速查

```bash
# 独立 VLA 评估
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05

# 双系统 agentic 评估（自动检测 config 名中的 "agentic"）
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic

# 切换任务
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=picking_up_trash

# 禁用记忆
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.enable_memory=False

# 切换到 Qwen3-VL-Thinking
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.model.model_type=qwen3_vl_embodied \
  vlm.model.model_path=/path/to/Qwen3-VL-4B-Thinking
```

---

## 向后兼容性

| 场景 | 行为 |
|------|------|
| `full_pi05: false`（默认） | 完全等同残血版 pi0.5，零影响 |
| `full_pi05: true, forward_mode: "vla"` | 满血版但仅做 flow matching，与残血版结果一致 |
| pi0 模型（非 pi0.5） | 不受影响，不走满血版路径 |

---

## 详细文档索引

### pi05_doc/ 目录

| 文档 | 内容 |
|------|------|
| `01-architecture.md` | 模型继承 + 三种模式 + 注意力 mask |
| `02-sft-guide.md` | SFT 三种模式启动指南 |
| `03-code-walkthrough.md` | 代码路径走读 |
| `04-sft-eval.md` | VLM-only SFT eval 流程（baseline、accuracy、QA 打印、FSDP 注意事项） |
| `05-behavior-vla-sft.md` | BEHAVIOR VLA SFT 实现（数据处理、norm stats、代码改动） |
| `06-behavior-b1k-alignment.md` | BEHAVIOR B1K openpi-comet 对齐详解 |
| `07-norm-stats.md` | Norm stats 生成方法 |

### dual_system/ 目录

| 文档 | 内容 |
|------|------|
| `README.md` | BEHAVIOR-1K 评估总览、快速入门、启动命令 |
| `vla_eval.md` | 独立 VLA 评估指南 |
| `vlm_vla_eval.md` | VLM+VLA 双系统评估指南 |
| `behavior_config.md` | 完整配置参考（50 个任务、场景初始化模式） |
| `data.md` | 数据参考（task instances、challenge demos、目录结构） |

### 其他

| 文档 | 内容 |
|------|------|
| `PPO.md` | PPO 全流程深度解析（rollout 数据收集到训练） |
