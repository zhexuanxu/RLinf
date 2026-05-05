# BEHAVIOR VLA-only SFT 实现文档

本文档记录在 RLinf 框架中为满血版 pi0.5 接入 BEHAVIOR-1K 数据集进行 VLA-only SFT 的完整实现过程，包括原始数据处理、归一化统计量计算、代码适配、以及训练配置。

---

## 1. 背景与目标

### 1.1 任务目标

在满血版 pi0.5 上实现 VLA-only SFT，使用 BEHAVIOR-1K challenge demos 数据集中 task-0000（turning_on_radio）的 200 条 episode 进行训练。

### 1.2 关键认知

满血版 pi0.5 在 `forward_mode="vla"` 模式下，与残血版 pi0.5 在行为上完全一致——只做 flow matching，不输出 CoT 文本。因此本次工作的核心不在模型侧，而在**数据管线适配**。

### 1.3 数据来源

| 项 | 值 |
|----|-----|
| 原始数据 | `/mnt/public/xzxuan/data/2025-challenge-demos-short/` |
| 数据格式 | LeRobot v2.1（parquet + MP4 视频） |
| 使用任务 | task-0000（turning_on_radio），200 条 episode，约 43 万帧 |
| 机器人 | R1Pro（双臂人形，23 维 action，256 维 proprioception） |
| 摄像头 | 3 个 RGB 摄像头：head (720x720)、left_wrist (480x480)、right_wrist (480x480) |

---

## 2. 需要解决的问题

在接入 BEHAVIOR 数据集的过程中，遇到了以下问题：

### 2.1 LeRobot episode 索引 bug

BEHAVIOR 数据集的 episode 索引不是从 0 开始的连续整数，而是跨任务的全局索引（如 10, 20, 30, ..., 3000）。LeRobot v2.1 的 `_get_query_indices` 方法将 parquet 中的 `episode_index` 原始值直接作为数组下标访问 `episode_data_index` 张量，导致 `IndexError`。

**解决方案**：编写数据重索引脚本，将 task-0000 的 200 条 episode 提取出来，重新编号为 0-199。

### 2.2 数据键名不匹配

原有的 `behavior_dataconfig.py` 中的 `RepackTransform` 映射的键名与 BEHAVIOR LeRobot v2.1 实际的键名不一致：

| 原有映射（错误） | 数据集实际键名 |
|-----------------|--------------|
| `image` | `observation.images.rgb.head` |
| `wrist_image`（堆叠格式） | `observation.images.rgb.left_wrist` / `right_wrist`（分离格式） |
| `state` | `observation.state` |
| `actions`（复数） | `action`（单数） |

### 2.3 Norm stats 缺失

pi0.5 base 模型（`pi05_base_pytorch`）目录下没有 BEHAVIOR 数据对应的归一化统计量。需要从 task-0000 数据中计算。

### 2.4 视频解码后端

环境中安装了 torchcodec 但运行时缺少 FFmpeg 动态库，导致视频解码失败。需要回退到 pyav。

### 2.5 视频时间戳容差

LeRobot 默认的 `tolerance_s=0.0001`（0.1ms）对于 BEHAVIOR 的 H.265 视频太严格。虽然 parquet 中的时间戳精确到 1/30s，但 pyav 解码 H.265 长视频时，实际解出的帧 PTS 与请求时间戳之间存在漂移，在长 episode 尾部（如 91.5s 处）可达 133ms（4 帧），触发 `AssertionError`。最终将容差设为 1.0s 彻底解决。

---

## 3. 原始数据处理

### 3.1 数据重索引脚本

**脚本路径**：`toolkits/behavior/prepare_behavior_task0000.py`

该脚本从原始数据集中提取 task-0000 的 200 条 episode，重新编号为连续的 0-199，输出一个独立的 LeRobot v2.1 格式数据集。

#### 处理流程

```
原始数据集 (2025-challenge-demos-short)
  │
  ├── 读取 meta/episodes.jsonl
  ├── 按任务描述筛选 task-0000 的 200 条 episode
  │   （匹配文本 "Turn on the radio receiver..."）
  │
  ├── 对每条 episode：
  │   ├── 读取原始 parquet 文件
  │   ├── 将 episode_index 列重写为新的连续索引（0-199）
  │   ├── 写入新目录 data/chunk-000/episode_NNNNNN.parquet
  │   └── 将 3 个 RGB 视频文件创建符号链接到新目录
  │
  ├── 生成新的 meta/info.json（更新 total_episodes、features 等）
  ├── 生成新的 meta/tasks.jsonl（仅包含 task-0000）
  ├── 生成新的 meta/episodes.jsonl（连续索引）
  └── 从原始数据过滤生成 meta/episodes_stats.jsonl
```

#### 运行方法

```bash
source /mnt/public/xzxuan/.venv_pi_371/bin/activate
python toolkits/behavior/prepare_behavior_task0000.py
```

#### 输出

| 项 | 值 |
|----|-----|
| 输出路径 | `/mnt/public/xzxuan/data/behavior-task0000-reindexed/` |
| Episode 数 | 200 |
| 总帧数 | 429,928 |
| 视频文件 | 符号链接到原始数据（不复制，节省空间） |

#### 输出目录结构

```
behavior-task0000-reindexed/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.rgb.head/
        │   ├── episode_000000.mp4 -> (symlink)
        │   └── ...
        ├── observation.images.rgb.left_wrist/
        └── observation.images.rgb.right_wrist/
```

### 3.2 为什么需要重索引

LeRobot v2.1 内部通过 `episode_data_index["from"][ep_idx]` 查找每条 episode 的帧范围。这里 `ep_idx` 取自 parquet 中的 `episode_index` 列值，而 `episode_data_index` 是按加载顺序构建的位置索引。当 `episode_index` 不从 0 开始连续编号时（如 BEHAVIOR 的 10, 20, 30...），两者不匹配，导致越界。

重索引后 `episode_index` 为 0, 1, 2, ..., 199，与位置索引一致。

---

## 4. Norm Stats 计算

### 4.1 计算脚本

**脚本路径**：`toolkits/behavior/compute_behavior_norm_stats.py`

### 4.2 计算流程

```
遍历重索引数据集的所有 parquet 文件
  │
  ├── 对每一帧：
  │   ├── 读取 observation.state（256 维原始 proprioception）
  │   ├── 调用 extract_state_from_proprio() 提取 23 维策略状态
  │   ├── 零填充到 32 维（模型维度）
  │   ├── 读取 action（23 维）
  │   └── 零填充到 32 维
  │
  ├── 汇总所有帧，计算每个维度的：
  │   ├── mean（均值）
  │   ├── std（标准差）
  │   ├── q01（第 1 百分位数）
  │   └── q99（第 99 百分位数）
  │
  └── 保存为 norm_stats.json
```

### 4.3 状态提取细节

原始 proprioception 为 256 维 R1Pro 全状态向量。通过 `extract_state_from_proprio()` 提取 23 维策略相关子集：

| 成分 | 索引范围 | 维度 |
|------|---------|------|
| base_qvel（底盘速度） | [253:256] | 3 |
| trunk_qpos（躯干关节） | [236:240] | 4 |
| arm_left_qpos（左臂关节） | [158:165] | 7 |
| arm_right_qpos（右臂关节） | [197:204] | 7 |
| left_gripper_width（左夹爪开合） | sum([193:195]) | 1 |
| right_gripper_width（右夹爪开合） | sum([232:234]) | 1 |
| **合计** | | **23** |

提取后的 23 维状态和 23 维动作分别零填充到 32 维（pi0.5 模型的统一动作维度），然后计算统计量。

### 4.4 为什么填充到 32 维

pi0.5 模型内部的 action_dim 固定为 32。数据管线中的 `PadStatesAndActions` transform 会将实际的 23 维状态/动作零填充到 32 维后再送入模型。norm stats 的维度必须与填充后一致，因此在计算时就按 32 维处理。填充维度的统计量为全零（mean=0, std=0, q01=0, q99=0），归一化时这些维度不受影响。

### 4.5 运行方法

```bash
source /mnt/public/xzxuan/.venv_pi_371/bin/activate
python toolkits/behavior/compute_behavior_norm_stats.py
```

### 4.6 输出

```
输出路径: /mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json
```

文件格式：

```json
{
  "norm_stats": {
    "state": {
      "mean": [... 32 个值 ...],
      "std":  [... 32 个值 ...],
      "q01":  [... 32 个值 ...],
      "q99":  [... 32 个值 ...]
    },
    "actions": {
      "mean": [... 32 个值 ...],
      "std":  [... 32 个值 ...],
      "q01":  [... 32 个值 ...],
      "q99":  [... 32 个值 ...]
    }
  }
}
```

### 4.7 归一化方式

训练时使用 **quantile normalization**（分位数归一化）：

```
normalized = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
```

将数据映射到大约 [-1, 1] 范围。这是 pi0.5 对 BEHAVIOR 数据的标准归一化方式（通过配置中 `use_quantile_norm=True` 控制）。

### 4.8 Norm stats 路径如何被找到

模型加载时有两处读取 norm stats：

1. **数据加载器创建时**：`DataConfigFactory.create_base_config()._load_norm_stats(assets_dir, asset_id)` 在 `{assets_dir}/{asset_id}/norm_stats.json` 查找
2. **模型初始化时**：`_checkpoints.load_norm_stats(checkpoint_dir, asset_id)` 在 `{checkpoint_dir}/{asset_id}/norm_stats.json` 查找

通过配置中 `assets_dir="/mnt/public/xzxuan/models/pi05_base_pytorch"` 和 `asset_id="physical-intelligence/behavior"`，两处都指向同一路径。

---

## 5. 代码改动说明

### 5.1 behavior_dataconfig.py — 键名映射修正

**文件**：`rlinf/models/embodiment/openpi/dataconfig/behavior_dataconfig.py`

`RepackTransform` 的结构是 `{输出键: 输入键}`，即"从数据中查找 `输入键` 的值，赋给 `输出键`"。

修改前后对比：

```python
# 修改前（旧键名，不匹配 BEHAVIOR v2.1）
{
    "observation/image": "image",
    "observation/wrist_image": "wrist_image",
    "observation/state": "state",
    "actions": "actions",
    "prompt": "prompt",
}

# 修改后（匹配 BEHAVIOR LeRobot v2.1 实际键名）
{
    "observation/image": "observation.images.rgb.head",
    "observation/left_wrist_image": "observation.images.rgb.left_wrist",
    "observation/right_wrist_image": "observation.images.rgb.right_wrist",
    "observation/state": "observation.state",
    "actions": "action",       # 注意：数据集中是单数 "action"
    "prompt": "prompt",
}
```

另外在返回的 `DataConfig` 中增加了 `action_sequence_keys=("action",)`。这控制 `create_torch_dataset` 中 `delta_timestamps` 的键名，确保 LeRobot 用 `"action"` 而非默认的 `"actions"` 来构建 action chunk。

### 5.2 behavior_policy.py — 支持分离的腕部图像

**文件**：`rlinf/models/embodiment/openpi/policies/behavior_policy.py`

BEHAVIOR 数据集以分离的键提供左右腕部图像（`observation.images.rgb.left_wrist` 和 `right_wrist`），而非旧代码预期的堆叠格式。

修改为同时兼容两种格式：

```python
# 旧代码（只支持堆叠格式）
wrist_image = _parse_image(data["observation/wrist_image"])  # [2, h, w, c]
left_wrist = wrist_image[0, ...]
right_wrist = wrist_image[1, ...]

# 新代码（兼容两种格式）
if "observation/wrist_image" in data:
    # 旧的堆叠格式
    wrist_image = _parse_image(data["observation/wrist_image"])
    left_wrist = wrist_image[0, ...]
    right_wrist = wrist_image[1, ...]
else:
    # BEHAVIOR v2.1 分离格式
    left_wrist = _parse_image(data["observation/left_wrist_image"])
    right_wrist = _parse_image(data["observation/right_wrist_image"])
```

### 5.3 dataconfig/\_\_init\_\_.py — 注册 pi05_behavior_local 配置

**文件**：`rlinf/models/embodiment/openpi/dataconfig/__init__.py`

新增一条 `TrainConfig` 注册项：

```python
TrainConfig(
    name="pi05_behavior_local",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=32),
    data=LeRobotBehaviorDataConfig(
        repo_id="/mnt/public/xzxuan/data/behavior-task0000-reindexed",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="/mnt/public/xzxuan/models/pi05_base_pytorch",
            asset_id="physical-intelligence/behavior",
        ),
        extra_delta_transform=False,
        extract_state_from_proprio=True,
        use_all_wrist_images=True,
        use_quantile_norm=True,
    ),
    pytorch_weight_path="/mnt/public/xzxuan/models/pi05_base_pytorch",
    num_train_steps=30_000,
),
```

关键参数说明：

| 参数 | 值 | 说明 |
|------|-----|------|
| `repo_id` | 本地绝对路径 | LeRobot 接受绝对路径作为 repo_id，会直接从该路径加载 |
| `asset_id` | `"physical-intelligence/behavior"` | norm stats 查找路径的子目录名 |
| `action_horizon` | 32 | B1K 标准 action chunk 长度 |
| `extract_state_from_proprio` | True | 从 256 维 proprio 提取 23 维策略状态 |
| `use_all_wrist_images` | True | 启用左右腕部图像（3 个摄像头全用） |
| `use_quantile_norm` | True | 使用分位数归一化（非 z-score） |
| `prompt_from_task` | True | 从 LeRobot 的 tasks 元数据自动提取文本指令 |

### 5.4 fsdp_vla_sft_worker.py — 视频后端修复

**文件**：`rlinf/workers/sft/fsdp_vla_sft_worker.py`

在 `build_dataloader` 方法中添加两处 monkey-patch：

**1) pyav 回退**：torchcodec 虽然可以 import，但运行时缺少 FFmpeg 动态库。通过实际尝试导入 `VideoDecoder` 来判断是否可用，失败则回退到 pyav：

```python
def _pyav_fallback():
    try:
        from torchcodec.decoders import VideoDecoder
        return _orig_codec()
    except Exception:
        return "pyav"
```

**2) 时间戳容差放宽**：将 LeRobotDataset 的默认 `tolerance_s` 从 0.0001s 提高到 1.0s。BEHAVIOR 的 H.265 视频在 pyav 解码时，长 episode 尾部的时间戳漂移可达 133ms（4 帧），0.04s 不够，1.0s 可以覆盖所有情况：

```python
def _init_with_tolerance(self_ds, *args, **kwargs):
    kwargs.setdefault("tolerance_s", 1.0)
    _orig_lrd_init(self_ds, *args, **kwargs)
```

---

## 6. 训练配置

### 6.1 配置文件

**路径**：`examples/sft/config/behavior_pi05_vla.yaml`

### 6.2 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `config_name` | `"pi05_behavior_local"` | 使用新注册的 BEHAVIOR 本地数据配置 |
| `full_pi05` | `True` | 使用满血版 pi0.5 模型 |
| `forward_mode` | `"vla"` | VLA-only 模式（只做 flow matching） |
| `model_path` | `pi05_base_pytorch` | 基础 pi0.5 PyTorch 权重 |
| `num_action_chunks` | 32 | B1K action horizon |
| `action_dim` | 23 | R1Pro 动作空间维度 |
| `num_images_in_input` | 3 | head + left_wrist + right_wrist |
| `micro_batch_size` | 2 | 每 GPU batch size |
| `global_batch_size` | 4 | 总 batch size（2 GPU x 2） |
| `sharding_strategy` | `"no_shard"` | 不分片，每张 GPU 放完整模型 |
| `gradient_checkpointing` | False | pi0.5 不支持 gradient checkpointing |
| `precision` | null | pi0.5 各层精度不同，不统一设置 |
| GPU 分配 | `actor,env,rollout: 0-1` | 使用 GPU 0 和 1 |

### 6.3 启动命令

```bash
source /mnt/public/xzxuan/.venv_pi_371/bin/activate
bash examples/sft/run_vla_sft.sh behavior_pi05_vla
```

### 6.4 训练表现

首次成功运行的指标：

| 指标 | 值 |
|------|-----|
| Step 1 loss | 0.249 |
| Step 10 loss | 0.383 |
| 稳态 loss 范围 | 0.2 - 0.35 |
| 每步耗时 | ~0.7s（首步 ~25s 含初始化） |
| Action loss | 0.15 - 0.50 |
| Grad norm | 2 - 10 |

---

## 7. 完整数据流

从原始数据到模型训练的完整数据流：

```
                        ┌─────────────────────────────────┐
                        │  原始 BEHAVIOR 数据集              │
                        │  2025-challenge-demos-short/      │
                        │  (12 tasks, 10000 eps, 非连续索引) │
                        └───────────────┬─────────────────┘
                                        │
                        prepare_behavior_task0000.py
                                        │
                        ┌───────────────▼─────────────────┐
                        │  重索引数据集                      │
                        │  behavior-task0000-reindexed/     │
                        │  (1 task, 200 eps, 连续索引 0-199) │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  LeRobotDataset(repo_id, ...)    │
                        │  delta_timestamps={"action": ... }│
                        │  video_backend="pyav"             │
                        │  tolerance_s=0.04                 │
                        └───────────────┬─────────────────┘
                                        │ 返回：
                                        │ {
                                        │   "observation.images.rgb.head": [3,720,720] float32
                                        │   "observation.images.rgb.left_wrist": [3,480,480]
                                        │   "observation.images.rgb.right_wrist": [3,480,480]
                                        │   "observation.state": [256] float32
                                        │   "action": [32, 23] float32  (action chunk)
                                        │   "task_index": 0
                                        │   ...
                                        │ }
                                        │
                        ┌───────────────▼─────────────────┐
                        │  PromptFromLeRobotTask            │
                        │  task_index=0 → "Turn on the ..." │
                        │  添加 "prompt" 字段                │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  RepackTransform                  │
                        │  observation.images.rgb.head      │
                        │    → observation/image             │
                        │  observation.images.rgb.left_wrist │
                        │    → observation/left_wrist_image  │
                        │  observation.state                 │
                        │    → observation/state             │
                        │  action → actions                  │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  BehaviorInputs                   │
                        │  ├─ _parse_image(): CHW→HWC,      │
                        │  │  float32→uint8                  │
                        │  ├─ extract_state_from_proprio():  │
                        │  │  256d → 23d → 截取前32d          │
                        │  └─ 构建 {state, image, image_mask,│
                        │       actions, prompt}             │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  Normalize (quantile)             │
                        │  state: (x-q01)/(q99-q01)*2 - 1  │
                        │  actions: (x-q01)/(q99-q01)*2 - 1 │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  ModelTransformFactory             │
                        │  ├─ ResizeImages: → 224x224        │
                        │  ├─ TokenizePrompt: PaliGemma      │
                        │  └─ PadStatesAndActions: → 32d     │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  DataLoader                       │
                        │  batch = (Observation, Actions)   │
                        │  Actions: [B, 32, 32]             │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  FSDPVlaSftWorker                 │
                        │  .get_train_model_output(batch)   │
                        │  → 转 GPU tensor                   │
                        │  → model(ForwardType.SFT, data)   │
                        └───────────────┬─────────────────┘
                                        │
                        ┌───────────────▼─────────────────┐
                        │  OpenPi05FullForRLActionPrediction │
                        │  .sft_forward()                   │
                        │  ├─ 自动检测 VLA-only 模式          │
                        │  │  (actions≠None, loss_mask=0)   │
                        │  ├─ _forward_vla_full(ce=False)   │
                        │  └─ 返回 flow matching loss        │
                        └───────────────────────────────────┘
```

---

## 8. 文件清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `toolkits/behavior/prepare_behavior_task0000.py` | 数据重索引脚本 |
| `toolkits/behavior/compute_behavior_norm_stats.py` | Norm stats 计算脚本 |
| `examples/sft/config/behavior_pi05_vla.yaml` | 训练配置 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `rlinf/models/embodiment/openpi/dataconfig/behavior_dataconfig.py` | RepackTransform 键名修正 + 添加 `action_sequence_keys` |
| `rlinf/models/embodiment/openpi/policies/behavior_policy.py` | BehaviorInputs 兼容分离/堆叠两种腕部图像格式 |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | 注册 `pi05_behavior_local` 配置 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | pyav 回退 + 视频时间戳容差放宽 |

### 生成的数据/资源

| 路径 | 说明 |
|------|------|
| `/mnt/public/xzxuan/data/behavior-task0000-reindexed/` | 重索引后的 task-0000 数据集 |
| `/mnt/public/xzxuan/models/pi05_base_pytorch/physical-intelligence/behavior/norm_stats.json` | Task-0000 归一化统计量 |

---

## 9. 向后兼容性

| 场景 | 影响 |
|------|------|
| 原有 `pi05_behavior` 配置 | 不影响——新增了独立的 `pi05_behavior_local` |
| BehaviorInputs 旧格式 | 兼容——`if/else` 分支同时处理堆叠和分离格式 |
| 其他数据集（ALOHA、LIBERO 等） | 不影响——它们使用各自的 DataConfig 和 Policy |
| pyav 回退 | 仅在 torchcodec 运行时失败时触发，不影响正常环境 |
| 时间戳容差 | 使用 `setdefault`，不覆盖已显式设置的值 |

---

## 10. 如果要适配其他任务

如果需要在其他 BEHAVIOR 任务上训练（如 task-0001），需要：

1. 修改 `prepare_behavior_task0000.py` 中的 `TASK_TEXT` 匹配目标任务描述
2. 重新运行数据重索引和 norm stats 计算
3. 更新 `__init__.py` 中 `pi05_behavior_local` 的 `repo_id` 指向新数据集
4. 或者创建新的 config 注册项（如 `pi05_behavior_task0001`）

如果要使用全部 50 个任务的数据，需要解决 LeRobot 的非连续索引问题——可以对全量数据做类似的重索引处理。

---

## 11. 参考

- **openpi-comet**（`/mnt/public/xzxuan/repos/openpi-comet`）：BEHAVIOR 数据适配的参考实现
- **pi05-b1kpt50-cs32**（`/mnt/public/xzxuan/models/pi05-b1kpt50-cs32`）：预训练 BEHAVIOR 模型，其 assets 下有全量 norm stats 可参考
- **残血版 SFT 文档**：`docs/source-zh/rst_source/examples/embodied/sft_openpi.rst`
