# Pi0.5 SFT 训练指南

## 统一设计

所有 SFT 模式共享：

1. **统一启动脚本**：`examples/sft/run_vla_sft.sh <config_name>`
2. **统一训练 worker**：`rlinf/workers/sft/fsdp_vla_sft_worker.py`
3. **统一模型入口**：`OpenPi05FullForRLActionPrediction.sft_forward()`
4. **统一数据格式**：`(observation_dict, actions_or_None, meta_dict)` 3-tuple
   - `meta_dict` 在训练时是 `{}`，在 eval 时携带原始问题/答案文本
   - openpi LeRobot 数据加载器仍是 2-tuple，worker 自动兼容

通过 config 中 `forward_mode` 字段切换模式。

## 统一数据格式

每个 batch 是 3-tuple `(observation, actions, meta)`（VLM 数据集）或 2-tuple `(observation, actions)`（openpi 数据加载器）：

```python
# 统一的 observation 结构
observation = {
    "image": {
        "image_0": ndarray[B, 224, 224, 3],    # float32 [-1, 1]
        "image_1": ndarray[B, 224, 224, 3],    # 可选，不用的设 mask=False
    },
    "image_mask": {
        "image_0": ndarray[B],                  # bool
        "image_1": ndarray[B],
    },
    "state": ndarray[B, state_dim],             # float32
    "tokenized_prompt": ndarray[B, max_token_len],      # int32
    "tokenized_prompt_mask": ndarray[B, max_token_len],  # bool
    "token_ar_mask": ndarray[B, max_token_len],          # int32 (0=双向, 1=因果)
    "token_loss_mask": ndarray[B, max_token_len],        # bool
    "token_kv_cache_mask": ndarray[B, max_token_len],    # bool
}

# VLM-only: actions=None
# VLA-only: actions=ndarray[B, action_horizon, action_dim]
# VLM+VLA: actions=ndarray[B, action_horizon, action_dim]
```

**模式由数据内容决定**：

| 数据内容 | 模式 | forward 路径 | 损失函数 |
|---------|------|-------------|---------|
| `actions=None` + `loss_mask` 有 True | VLM-only | `_forward_vlm()` | CE loss |
| `actions` 有值 + `loss_mask` 全 False | VLA-only | `_forward_vla_full(ce=False)` | Flow matching |
| `actions` 有值 + `loss_mask` 有 True | VLM+VLA | `_forward_vla_full(ce=True)` | CE + flow matching |

## 模式 1：VLM-only SFT

### 用途

训练 pi0.5 的 VLM 部分学习视觉问答，用于后续 CoT 推理能力。

### 数据集

Robo2VLM（parquet 格式）：

```
data/Robo2VLM/data/
├── train-00000-of-00262.parquet
├── train-00001-of-00262.parquet
└── ...

字段：question(str), choices(str), correct_answer(int), image(bytes)
```

### 数据处理流程

```
Robo2VLM parquet
    ↓ Pi05VLMDataset.__getitem__()
    ├── image → resize 224x224, normalize to [-1,1]
    ├── question+answer → PaliGemma tokenize
    ├── token_ar_mask: prompt=0(双向), answer+EOS=1(因果)
    ├── token_loss_mask: answer+EOS=True
    └── token_kv_cache_mask: all True except EOS
    ↓ pi05_vlm_collate_fn
    ↓ (observation_dict, None)
    ↓ fsdp_vla_sft_worker.get_train_model_output()
    ├── 提取 token_kv_cache_mask（不在 Observation 中）
    ├── observation → GPU tensors
    └── model(ForwardType.SFT, data={obs, actions=None, kv_mask})
    ↓ sft_forward()
    ├── Observation.from_dict(observation)
    ├── _preprocess_observation_full() → 8-tuple
    ├── has_actions=False → _forward_vlm()
    │   ├── embed_prefix_with_ar_mask (因果注意力)
    │   ├── PaliGemma forward (无 action expert)
    │   └── _compute_ce_loss (next-token prediction)
    └── return {"loss": ..., "language_loss": ..., "language_token_acc": ...}
```

### 启动命令

```bash
bash examples/sft/run_vla_sft.sh robotwin_sft_openpi_pi05_vlm
```

### 关键配置

```yaml
# examples/sft/config/robotwin_sft_openpi_pi05_vlm.yaml
runner:
  val_check_interval: 200       # 每 N 步 eval 一次（>0 启用）
  save_interval: 2000
  val_at_step_0: True           # 训练前 baseline eval
  print_eval_samples: 5         # rank 0 每次 eval 打印 K 条 QA

data:
  type: vlm
  dataset_name: "pi05_robo2vlm"
  train_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/data"
  val_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/eval_data"
  prompt_key: "question"
  choice_key: "choices"
  answer_key: "correct_answer"
  image_keys: ["image"]
  max_token_len: 200
  num_workers: 4

actor:
  micro_batch_size: 4
  eval_batch_size: 4
  global_batch_size: 256
  model:
    precision: null
    model_path: "/mnt/public/xzxuan/models/pi05_base_pytorch"
    openpi:
      full_pi05: True
      forward_mode: "vlm"
      num_images_in_input: 1   # Robo2VLM 1 张图（默认 2）
  fsdp_config:
    sharding_strategy: "no_shard"
    use_orig_params: True       # 满血版必须 True，见 CLAUDE.md
    gradient_checkpointing: False
```

### Eval 流程

每 `val_check_interval` 步触发一次 eval（`val_at_step_0: True` 时训练前也跑一次 baseline）。

- **指标**：MCQ 字母准确率（`generate_language` 自回归生成 → 解码 → 提取 `[A-F]` → 与 gold 比较）
- **打印**：rank 0 在终端打印 `print_eval_samples` 条 QA 对（question / choices / gold / pred / raw）
- **TensorBoard**：`eval/eval_accuracy`

详细流程见 [04-sft-eval.md](./04-sft-eval.md)。

### 不需要 asset/norm_stats

VLM-only 模式不涉及 action 归一化。`get_model()` 检测到 `full_pi05=True, forward_mode="vlm"` 时自动跳过 norm_stats 加载。因此可以使用不带 asset 的 base 模型。

## 模式 2：VLA-only SFT

### 用途

训练 action 预测（flow matching），与残血版 pi0.5 完全一致。

### 数据集

LeRobot 格式（HuggingFace Datasets），通过 openpi 数据加载器。

### 数据处理流程

```
LeRobot dataset
    ↓ openpi.training.data_loader.create_data_loader()
    ├── RepackTransform → Normalize → TokenizePrompt → ResizeImages
    └── yield (Observation, actions_tensor)
    ↓ fsdp_vla_sft_worker.get_train_model_output()
    └── model(ForwardType.SFT, data={obs, actions})
    ↓ sft_forward()
    ├── _preprocess_observation_full()
    ├── token_loss_mask.any()=False → compute_ce_loss=False
    ├── _forward_vla_full(compute_ce_loss=False)
    │   ├── 采样 noise, time → x_t = t*noise + (1-t)*actions
    │   ├── _compute_velocity_with_prefix_out() → joint forward
    │   └── flow_loss = MSE(u_t, v_t)
    └── return {"loss": flow_loss, "action_loss": flow_loss}
```

### 启动命令

```bash
# 残血版（原有配置不变）
bash run_vla_sft.sh robotwin_sft_openpi_pi05

# 满血版（forward_mode=vla 时行为一致）
# 在 yaml 中加 full_pi05: True, forward_mode: "vla"
```

### 需要 asset/norm_stats

VLA 模式需要 action 归一化参数。使用带 asset 的模型路径（如 `/mnt/public/daibo/models/pi05_b1kpt50_pt`）。

## 模式 3：VLM+VLA SFT（未来）

同时训练文本推理和 action 预测。需要带有 `cot_text` 标注的数据集。

```bash
bash run_vla_sft.sh robotwin_sft_openpi_pi05_vlm_vla
```

Loss = `language_loss_weight * CE_loss + action_loss_weight * flow_loss`

## 监控指标

| 指标 | VLM-only | VLA-only | VLM+VLA |
|------|---------|---------|---------|
| `loss` | CE loss | flow loss | weighted sum |
| `language_loss` | CE loss | - | CE loss |
| `action_loss` | - | flow loss | flow loss |
| `language_token_acc` | token acc | - | token acc |

## 常见问题

### Q: 启动报错 "Asset id is required to load norm stats"
检查是否设置了 `full_pi05: True` + `forward_mode: "vlm"`。VLM-only 模式会跳过 norm_stats。如果用 VLA 或 VLM+VLA 模式，需要带 asset 的模型。

### Q: gradient_checkpointing 必须关闭吗？
是。pi0.5 不同层精度不同，gradient checkpointing 会导致精度错误。

### Q: precision 必须设为 null 吗？
是。让模型自行管理各层精度。

### Q: 满血版和残血版可以共用 checkpoint 吗？
可以。满血版是子类，权重完全兼容。lm_head 已在 PaliGemma 中，无需额外权重。

### Q: `use_orig_params` 必须 True 吗？
满血版必须 True。原因：pi0.5 各层 dtype 不同（`precision: null`），`use_orig_params=False` 会要求 flat_param 内 dtype 一致 → 直接报错。

### Q: eval 期间报 `_is_root` 断言？
不要直接调 `model.generate_language(...)` 或子模块 `forward()` —— 会绕过外层 FSDP 触发 lazy_init 冲突。正确方式：`model(forward_type=ForwardType.GENERATE_LANGUAGE, ...)`。详见 [04-sft-eval.md](./04-sft-eval.md)。
