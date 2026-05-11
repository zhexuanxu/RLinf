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

## 模式 1：VLM-only SFT（BEHAVIOR 技能预测）

### 用途

训练 VLM 根据当前画面预测正在执行的技能（skill）。目前支持两个模型：

| 方案 | 模型 | 启动脚本 | 数据格式 |
|------|------|---------|---------|
| pi0.5 VLM | PaliGemma (pi0.5 内置) | `run_vla_sft.sh` | openpi Observation |
| Qwen2.5-VL | Qwen2.5-VL-3B-Instruct | `run_vlm_sft.sh` | SftDatasetItem + chat template |

### 数据集

BEHAVIOR-1K task-0000（200 个 episode），通过 head camera MP4 视频 + annotation JSON 构建。

**数据来源**：`/mnt/public/xzxuan/data/2025-challenge-demos/`

**4 个固定技能标签**（所有 episode 共享）：

| skill_idx | 标签 | 典型帧范围 |
|-----------|------|-----------|
| 0 | `move to radio` | 0~265 |
| 1 | `pick up radio from coffee table` | 265~1162 |
| 2 | `press radio` | 1162~1434 |
| 3 | `place radio on coffee table` | 1434~1776 |

**帧→标签映射**：每个 annotation JSON 的 `skill_annotation[i].frame_duration = [start, end]`，用 `bisect` 将帧索引映射到对应的 skill_idx。

**采样策略**：每 10 帧采样 1 帧（`frame_stride=10`），~36K 样本。Episode 划分：180 train / 20 eval。

### 数据处理流程（两种路径共享 `_BehaviorSkillIndex`）

```
annotations/task-0000/episode_*.json
    ↓ _BehaviorSkillIndex._build_index()
    ├── 解析 skill_annotation → frame_duration
    ├── 对每个 skill，每隔 10 帧采样一个 (episode_id, frame_idx, skill_idx)
    └── 生成 flat sample 列表

videos/task-0000/observation.images.rgb.head/episode_*.mp4
    ↓ _BehaviorSkillIndex.load_frame(episode_id, frame_idx)
    ├── cv2.VideoCapture → seek → read → BGR→RGB → PIL.Image
    └── 1-video LRU 缓存
```

**pi0.5 路径**（`BehaviorSkillPi05Dataset`）：
```
PIL.Image → resize 224×224, normalize [-1,1]
prompt = "Task: Turn on the radio...\nSkill: "
answer = "pick up radio from coffee table"
→ PaliGemma tokenize → observation_dict
→ pi05_vlm_collate_fn → (observation, None, meta)
→ sft_forward() → _forward_vlm() → CE loss
```

**Qwen2.5-VL 路径**（`BehaviorSkillQwenDataset`）：
```
PIL.Image (原始分辨率)
→ Qwen processor + chat template:
  System: (可选)
  User: [image] "Task: Turn on the radio... What skill is being performed?"
  Assistant: "pick up radio from coffee table"
→ SftDatasetItem (input_ids, attention_mask, label_mask, pixel_values)
→ sft_collate_fn → CE loss on answer tokens
```

### 启动命令

**pi0.5 VLM-only**（4 GPU）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_sft
```

**Qwen2.5-VL**（4 GPU）：

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 bash examples/sft/run_vlm_sft.sh behavior_qwen2_5_vlm_sft
```

### 关键配置

**pi0.5**（`examples/sft/config/behavior_pi05_vlm_sft.yaml`）：

```yaml
data:
  dataset_name: "behavior_skill_pi05"
  train_data_paths: "/mnt/public/xzxuan/data/2025-challenge-demos"
  val_data_paths: "/mnt/public/xzxuan/data/2025-challenge-demos"
  max_token_len: 200
  frame_stride: 10
  eval_episode_count: 20

actor:
  model:
    precision: null
    model_path: "/mnt/public/xzxuan/models/pi05_base_pytorch"
    openpi:
      full_pi05: True
      forward_mode: "vlm"
      num_images_in_input: 1
  fsdp_config:
    sharding_strategy: "no_shard"
    use_orig_params: True
    gradient_checkpointing: False
```

**Qwen2.5-VL**（`examples/sft/config/behavior_qwen2_5_vlm_sft.yaml`）：

```yaml
data:
  dataset_name: "behavior_skill_sft"
  train_data_paths: "/mnt/public/xzxuan/data/2025-challenge-demos"
  val_data_paths: "/mnt/public/xzxuan/data/2025-challenge-demos"
  max_prompt_length: 512
  frame_stride: 10
  eval_episode_count: 20

actor:
  model:
    model_type: "qwen2.5_vl"
    model_path: "/mnt/public/xzxuan/models/Qwen2.5-VL-3B-Instruct"
  fsdp_config:
    sharding_strategy: "full_shard"
    mixed_precision:
      param_dtype: bf16
```

### Eval 指标

- **pi0.5**：技能标签精确匹配率（`generate_language` 生成 → 与 gold skill label 比较）
- **Qwen2.5-VL**：生成文本与 gold skill label 匹配
- **TensorBoard**：`eval/eval_accuracy`

### 不需要 asset/norm_stats

VLM-only 模式不涉及 action 归一化。两种路径都不加载 norm_stats。

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
