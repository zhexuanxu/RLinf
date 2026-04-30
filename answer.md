# 四个问题的回答

## Q1: 如何安装 FFmpeg 使 torchcodec 正常工作？

### 问题诊断

当前环境的状态：

| 组件 | 当前版本 | torchcodec 需要 |
|------|---------|----------------|
| FFmpeg (libavutil) | 4.4 (`libavutil.so.56`) | 5/6/7 (`libavutil.so.57/58/59`) |
| PyTorch | 2.5.1+cu124 | torchcodec 0.2.0 需要匹配的 PyTorch ABI |

torchcodec 加载失败有**两个原因**：
1. 系统 FFmpeg 版本太低（4.4），torchcodec 依次尝试 `libavutil.so.59`/`.58`/`.57` 均找不到
2. PyTorch ABI 不兼容：`undefined symbol: _ZNK3c1011StorageImpl27throw_data_ptr_access_errorEv`，说明 torchcodec 0.2.0 编译时对应的 PyTorch 版本与当前 2.5.1 不匹配

### 解决方案

**方案 A（推荐）：不管 torchcodec，继续用 pyav**

当前代码已通过 monkey-patch 自动回退到 pyav，训练正常运行。pyav 对于 SFT 训练完全够用，无需修复 torchcodec。

**方案 B：升级 FFmpeg + 重装 torchcodec**

如果确实需要 torchcodec（性能更好），步骤如下：

```bash
source /mnt/public/xzxuan/.venv_pi_371/bin/activate

# 1. 安装 FFmpeg 7（通过 conda-forge，不影响系统 FFmpeg）
conda install -c conda-forge ffmpeg=7

# 2. 安装与当前 PyTorch 2.5.1 匹配的 torchcodec
#    参考 https://github.com/pytorch/torchcodec#installing-torchcodec
#    PyTorch 2.5 对应 torchcodec 0.1.x，不是 0.2.0
pip install torchcodec==0.1.1

# 3. 验证
python -c "from torchcodec.decoders import VideoDecoder; print('OK')"
```

注意：torchcodec 0.2.0 需要 PyTorch >= 2.6。当前环境是 PyTorch 2.5.1，应降级到 torchcodec 0.1.x，或升级 PyTorch。由于升级 PyTorch 可能影响其他依赖，建议使用方案 A。

---

## Q2: TrainConfig 注册项的作用？SFT 专用还是 RL 也用？

### 作用

`TrainConfig` 是 openpi 框架中的**数据管线配置注册表**，**SFT 和 RL 都会用到**。它定义了：

| 字段 | 作用 | SFT 用 | RL 用 |
|------|------|--------|-------|
| `name` | 配置名，通过 YAML 中 `config_name` 引用 | Y | Y |
| `model` | Pi0Config（action_horizon、是否 pi05 等） | Y | Y |
| `data` | DataConfigFactory（RepackTransform、BehaviorInputs/Outputs、norm stats） | Y | Y |
| `pytorch_weight_path` | 模型权重路径 | Y | Y |
| `num_train_steps` | 训练步数 | Y | N |

### SFT 如何使用

在 `fsdp_vla_sft_worker.py` 的 `build_dataloader()` 中：

```python
config = get_openpi_config("pi05_behavior_local", model_path=..., batch_size=...)
data_loader = openpi_data_loader.create_data_loader(config, ...)
```

TrainConfig 的**全部字段**都被用到：`data` 字段创建数据加载器（RepackTransform + BehaviorInputs + Normalize + 模型 transforms），`model` 字段确定 action_horizon 用于构建 delta_timestamps。

### RL 如何使用

在 `rlinf/models/embodiment/openpi/__init__.py` 的 `get_model()` 中：

```python
actor_train_config = get_openpi_config("pi05_behavior", model_path=cfg.model_path)
data_config = actor_train_config.data.create(actor_train_config.assets_dirs, actor_model_config)
norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
model.setup_wrappers(
    transforms=[
        *transforms.Group().inputs,          # 空的 repack（RL 不用 RepackTransform）
        *data_config.data_transforms.inputs,  # BehaviorInputs
        transforms.Normalize(norm_stats, ...),
        *data_config.model_transforms.inputs,
    ],
    output_transforms=[
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, ...),
        *data_config.data_transforms.outputs,  # BehaviorOutputs
    ],
)
```

RL 用到了：
- `data_config.data_transforms`（BehaviorInputs/Outputs）—— 作为推理时的输入/输出 transform
- `data_config.norm_stats` / `asset_id` —— 加载归一化统计量
- `data_config.model_transforms` —— tokenizer、padding 等
- `data_config.use_quantile_norm` —— 归一化方式

RL **不用**：
- `data_config.repack_transforms` —— 被空 `transforms.Group()` 替代
- `data_config.action_sequence_keys` —— 只在 SFT 数据加载器中使用

### RL 配置引用

RL 的 YAML 配置（`behavior_ppo_openpi_pi05.yaml` 第 123 行）：

```yaml
openpi:
  config_name: "pi05_behavior"    # 指向原始注册项
```

而 SFT 配置（`behavior_pi05_vla.yaml`）：

```yaml
openpi:
  config_name: "pi05_behavior_local"  # 指向新注册项
```

两者引用不同的 TrainConfig 注册项，互不影响。

---

## Q3: 修改 behavior_dataconfig.py 和 behavior_policy.py 会不会影响 RL？

### 结论：不会影响。修改是向后兼容的。

### 逐项分析

#### 改动 1：behavior_dataconfig.py 的 RepackTransform 键名

```python
# 改前
{"observation/image": "image", "observation/wrist_image": "wrist_image", ...}
# 改后
{"observation/image": "observation.images.rgb.head", "observation/left_wrist_image": "observation.images.rgb.left_wrist", ...}
```

**不影响 RL。** 原因：RL 推理路径中，`get_model()` 在第 119 行显式创建了空的 `repack_transforms = transforms.Group()`，然后用这个空 group 的 `.inputs` 设置 wrapper。behavior_dataconfig.py 中定义的 RepackTransform **只在 SFT 数据加载管线中生效**（`openpi/training/data_loader.py` → `transform_dataset()`），不会进入 RL 推理路径。

代码证据（`rlinf/models/embodiment/openpi/__init__.py:119-129`）：

```python
repack_transforms = transforms.Group()  # ← 空的，不是 behavior_dataconfig 的
model.setup_wrappers(
    transforms=[
        *repack_transforms.inputs,            # ← 空列表
        transforms.InjectDefaultPrompt(...),
        *data_config.data_transforms.inputs,  # ← BehaviorInputs（下面分析）
        transforms.Normalize(...),
        *data_config.model_transforms.inputs,
    ],
    ...
)
```

#### 改动 2：behavior_dataconfig.py 添加 `action_sequence_keys=("action",)`

**不影响 RL。** `action_sequence_keys` 只在 `openpi/training/data_loader.py` 的 `create_torch_dataset()` 中被读取，用于构建 `delta_timestamps`。RL 推理路径的 `get_model()` 从未读取此字段。

#### 改动 3：behavior_policy.py 的 BehaviorInputs 支持分离腕部图像

```python
# 改前
wrist_image = _parse_image(data["observation/wrist_image"])  # 只支持堆叠格式

# 改后
if "observation/wrist_image" in data:       # RL 走这里（堆叠格式）
    wrist_image = _parse_image(data["observation/wrist_image"])
    left_wrist = wrist_image[0, ...]
    right_wrist = wrist_image[1, ...]
else:                                        # SFT 走这里（分离格式）
    left_wrist = _parse_image(data["observation/left_wrist_image"])
    right_wrist = _parse_image(data["observation/right_wrist_image"])
```

**不影响 RL。** RL 环境通过 `obs_processor`（`openpi_action_model.py:500-501`）提供的 key 是 `"observation/wrist_image"`（堆叠格式）：

```python
# openpi_action_model.py:500-501
if env_obs["wrist_images"] is not None:
    processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
```

所以 `"observation/wrist_image" in data` 为 `True`，走的是 **if 分支（旧逻辑）**，行为与修改前完全一致。

新增的 else 分支（分离格式）只在 SFT 数据管线中触发，因为 SFT 的 RepackTransform 把 LeRobot 的 `observation.images.rgb.left_wrist` 映射为 `observation/left_wrist_image`，不会出现 `observation/wrist_image` key。

### 总结

| 改动 | RL 是否使用 | 是否影响 RL | 原因 |
|------|-----------|-----------|------|
| RepackTransform 键名 | 否 | 否 | RL 用空 Group()，RepackTransform 只在 SFT 数据加载中使用 |
| `action_sequence_keys` | 否 | 否 | 只在 SFT 的 `create_torch_dataset` 中读取 |
| BehaviorInputs 分离腕部图像 | 是 | 否 | RL 提供 `observation/wrist_image`，命中 if 分支（旧逻辑） |

---

## Q4: openpi-comet 如何解决视频时间戳漂移问题？

### 结论：openpi-comet 没有增大 tolerance，而是**完全绕过了基于时间戳的视频帧寻址**。

### openpi-comet 的方案：Chunk Streaming 模式

openpi-comet 的 `BehaviorLeRobotDataset`（`src/behavior/learning/datas/dataset.py`）有两种模式：

| 模式 | 是否默认 | 视频加载方式 | 是否受 tolerance 影响 |
|------|---------|------------|---------------------|
| Chunk Streaming（`chunk_streaming_using_keyframe=True`） | **默认** | 自定义 VideoLoader 逐帧迭代 | **否** |
| 标准模式（`chunk_streaming_using_keyframe=False`） | 非默认 | LeRobotDataset._query_videos() | 是 |

在默认的 Chunk Streaming 模式下：

1. **将 episode 切分为 250 帧的 chunk**（对应 H.265 的 GOP 大小）
2. **用自定义 VideoLoader 逐帧加载**（`OBS_LOADER_MAP` 中的 `RGBVideoLoader` 等），这些 loader 基于 PyAV 的帧迭代器，按顺序读取帧，**不做时间戳寻址**
3. **完全不调用** LeRobotDataset 的 `_query_videos()` 方法，因此 `decode_video_frames()` 中的 tolerance 断言永远不会触发

相关代码（`dataset.py:395-428`）：

```python
# Chunk streaming 模式的帧加载
self.obs_loaders[vid_key] = iter(
    OBS_LOADER_MAP[vid_key.split(".")[2]](    # RGBVideoLoader / DepthVideoLoader
        data_path=self.root,
        task_id=task_id,
        camera_id=vid_key.split(".")[-1],
        demo_id=f"{ep_idx:08d}",
        start_idx=...,                        # chunk 起始帧
        batch_size=1,
        stride=1,                             # 逐帧迭代
    )
)
# ...
for key in self.meta.video_keys:
    item[key] = next(self.obs_loaders[key])[0]  # 直接 next()，无时间戳寻址
```

### 我的方案与 openpi-comet 的对比

| 方面 | openpi-comet | RLinf (我的方案) |
|------|-------------|----------------|
| 视频加载 | 自定义 VideoLoader 逐帧迭代 | LeRobot 标准 `_query_videos()` + pyav |
| tolerance 处理 | 不需要（绕过了时间戳寻址） | 放宽到 1.0s |
| 额外依赖 | omnigibson 的 VideoLoader 库 | 无 |
| 复杂度 | 高（自定义 Dataset、迭代器、chunk 管理） | 低（monkey-patch 一行） |

openpi-comet 的方案更精确（逐帧无误差），但需要依赖 omnigibson 库并实现自定义 Dataset 类。我的方案更简单，代价是最多 1 秒内的帧偏移（30 帧），对 SFT 训练质量影响可忽略。如果未来需要更精确的帧对齐，可以考虑移植 openpi-comet 的 chunk streaming 机制。
