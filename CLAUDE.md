# RLinf Pi0.5 Full Implementation Guide

## 项目背景

RLinf 是一个将强化学习（RL）应用到具身智能（Embodied AI）的框架。本仓库 `RLinf_pi05` 在 RLinf 基础上支持了 Physical Intelligence 的 pi0/pi0.5 模型。

### pi0 vs pi0.5 核心区别

| 特性 | pi0 | pi0.5 (残血版) | pi0.5 (满血版) |
|------|-----|---------------|---------------|
| State 输入 | 连续向量 (suffix) | 离散化到 token (prefix) | 同残血版 |
| 时间注入 | concat + MLP | adaRMSNorm | 同残血版 |
| Action 输出 | Flow matching | Flow matching | Flow matching |
| CoT 文本输出 | - | - | 支持 |
| VLM CE loss | - | - | 支持 |
| KV cache 推理 | - | - | 支持 |

**满血版 pi0.5 的核心增量**：让 VLM 先输出一段 CoT 推理文本（subtask），然后 action expert 再基于这个上下文输出 action。

### 相关仓库

- `/mnt/public/xzxuan/repos/openpi` — pi 官方仓库，实现了残血版 pi0.5（仅 flow matching，无 CoT）
- `/mnt/public/xzxuan/repos/vla_lib` — 第三方完整实现，包含完整 pi0.5（CoT + action + KV cache）
- `/mnt/public/xzxuan/repos/RLinf_pi05` — **本仓库**，在 RLinf 框架中实现完整 pi0.5

## 核心改动概览

在残血版 pi0.5 基础上新增了以下功能：

1. **VLM 文本输出**：通过 `lm_head` 输出 logits，支持 CE loss 训练
2. **三种训练模式**：VLM-only / VLA-only / VLM+VLA，通过 config `forward_mode` 控制
3. **自回归语言生成**：`generate_language()` 使用 StaticKVCache 高效生成
4. **先想再做推理**：`sample_with_reasoning()` 先生成 CoT 文本，再采样 action
5. **统一 SFT 接口**：一个 `sft_forward()` + 一个 `fsdp_vla_sft_worker.py` 处理全部模式

### 配置开关

```yaml
openpi:
  full_pi05: true      # true=满血版, false=残血版
  forward_mode: "vla"   # "vla" | "vlm" | "vlm_vla"
```

## 关键文件

| 文件 | 说明 |
|------|------|
| `rlinf/models/embodiment/openpi/openpi_full_pi05_model.py` | **核心** — 满血版 pi0.5 模型子类，含 `forward()` GENERATE_LANGUAGE 路由 |
| `rlinf/models/embodiment/openpi/static_kv_cache.py` | StaticKVCache + left_to_right_align |
| `rlinf/models/embodiment/openpi/openpi_action_model.py` | 父类（残血版），新增 config 字段 |
| `rlinf/models/embodiment/openpi/__init__.py` | `get_model()` 路由逻辑 |
| `rlinf/models/embodiment/openpi/dataconfig/cot_transform.py` | CoT 数据变换（VLM+VLA 模式） |
| `rlinf/models/embodiment/base_policy.py` | `ForwardType` 枚举（增加 `GENERATE_LANGUAGE`） |
| `rlinf/data/datasets/pi05_vlm_dataset.py` | Robo2VLM -> pi0.5 Observation 数据集，支持 eval_mode + 3-tuple |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | **统一** SFT worker（处理训练 + eval） |
| `rlinf/runners/sft_runner.py` | SFT runner，含 `val_at_step_0` 训练前 baseline eval 钩子 |

详细技术文档见 `pi05_doc/` 目录：
- `01-architecture.md`：模型继承 + 三种模式 + 注意力 mask
- `02-sft-guide.md`：SFT 三种模式启动指南
- `03-code-walkthrough.md`：代码路径走读
- `04-sft-eval.md`：VLM-only SFT eval 流程（baseline、accuracy、QA 打印、FSDP 注意事项）
- `05-behavior-vla-sft.md`：BEHAVIOR 数据集 VLA SFT 实现（数据处理、norm stats、代码改动）

---

## SFT 训练

### 统一设计

所有 SFT 模式使用**统一的入口**和**统一的数据结构**：

| 组件 | 统一入口 |
|------|---------|
| 启动脚本 | `examples/sft/run_vla_sft.sh <config_name>` |
| 训练 worker | `rlinf/workers/sft/fsdp_vla_sft_worker.py` |
| 模型 forward | `openpi_full_pi05_model.py: sft_forward()` |

**模式切换**：通过 config 中 `forward_mode` 字段控制：

| forward_mode | 数据源 | 监督信号 | 数据加载器 |
|-------------|--------|---------|-----------|
| `"vlm"` | Robo2VLM (parquet) | 文本 (CE loss) | `Pi05VLMDataset` |
| `"vla"` | LeRobot | action (flow matching) | openpi data loader |
| `"vlm_vla"` | LeRobot + CoT | 文本 + action | openpi + CoT transform |

**统一数据格式**：所有模式产生 `(observation, actions)` 二元组：

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

**模式自动检测**（`sft_forward` 内部）：
- `actions=None` -> `_forward_vlm()` (CE loss)
- `actions` + `loss_mask` 全 False -> `_forward_vla_full(ce=False)` (flow matching)
- `actions` + `loss_mask` 有 True -> `_forward_vla_full(ce=True)` (CE + flow matching)

---

### 模式 1：VLM-only SFT（Robo2VLM 数据集）

在 A100 上启动：

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_vlm
```

**配置文件**：`examples/sft/config/robotwin_sft_openpi_pi05_vlm.yaml`

**关键参数**：

```yaml
runner:
  val_check_interval: 200       # 每 N 步 eval 一次（>0 启用）
  save_interval: 2000           # 必须能被 val_check_interval 整除
  val_at_step_0: True           # 训练前先 eval 一次（untrained baseline）
  print_eval_samples: 5         # 每次 eval rank 0 打印 K 条 QA 对

data:
  type: vlm
  dataset_name: "pi05_robo2vlm"
  train_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/data"
  val_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/eval_data"   # eval 数据
  prompt_key: "question"
  choice_key: "choices"
  answer_key: "correct_answer"
  image_keys: ["image"]
  max_token_len: 200
  num_workers: 4

actor:
  micro_batch_size: 4
  eval_batch_size: 4              # eval per-rank batch
  global_batch_size: 256
  model:
    precision: null                # pi0.5 各层精度不同
    model_path: "/mnt/public/xzxuan/models/pi05_base_pytorch"  # 不需要 asset
    # num_action_chunks (10), action_dim (7), openpi.config_name ("pi05_libero")
    # 都从 model/pi0_5.yaml 默认值继承，不显式声明也能跑
    openpi:
      full_pi05: True              # 启用满血版
      forward_mode: "vlm"          # VLM-only 模式
      num_images_in_input: 1       # Robo2VLM 只有 1 张图（默认 2）
  fsdp_config:
    sharding_strategy: "no_shard"
    use_orig_params: True          
    gradient_checkpointing: False  # pi0.5 必须关闭
```

**注意事项**：
- VLM-only 模式**不需要** norm_stats/asset，可以使用不带 asset 的 base 模型
- `gradient_checkpointing` 必须设为 `False`
- `precision` 必须设为 `null`

**Eval 流程详细说明**：见 `pi05_doc/04-sft-eval.md`。关键能力：
- step 0 baseline：训练前 eval 一次，看 untrained 模型表现
- 每 `val_check_interval` 步触发一次 eval，准确率写入 TensorBoard `eval/eval_accuracy`
- rank 0 打印 K 条 QA 对（仅文字，包括 question / choices / gold letter / model raw output / 提取的 letter）

### 模式 2：VLA-only SFT（LeRobot 数据集）

**RoboTwin 数据**（残血版或满血版）：

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05
```

**BEHAVIOR 数据**（满血版，task-0000）：

```bash
bash examples/sft/run_vla_sft.sh behavior_pi05_vla
```

> 使用前需先运行数据准备脚本，详见 `pi05_doc/05-behavior-vla-sft.md`。

如果要在其他 LeRobot 数据上用满血版：

```yaml
openpi:
  full_pi05: True
  forward_mode: "vla"    # VLA-only，与残血版行为一致
```

### 模式 3：VLM+VLA SFT（未来扩展）

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_vlm_vla
```

需要带有 `cot_text` 标注的 LeRobot 数据集。

---

## PPO 训练（带 CoT 推理）

```bash
bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05_full
```

**设计**：CoT 仅用于推理（生成 subtask 文本 → 采样 action），PPO 只训练 action，不训练文本。


---

## 向后兼容性

| 场景 | 行为 |
|------|------|
| `full_pi05: false`（默认） | 完全等同残血版 pi0.5，零影响 |
| `full_pi05: true, forward_mode: "vla"` | 满血版但仅做 flow matching，与残血版结果一致 |
| pi0 模型（非 pi0.5） | 不受影响，不走满血版路径 |
