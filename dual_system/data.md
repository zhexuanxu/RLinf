# BEHAVIOR-1K 数据参考手册

本文档介绍 RLinf BEHAVIOR 评估流水线使用的两类主要数据源。

## 概述

| 数据集 | 位置 | 用途 |
|--------|------|------|
| 任务实例 | `$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances/` | 预生成的场景初始化配置（物体摆放、机器人位姿），用于评估 |
| 挑战赛演示 | `/path/to/2025-challenge-demos/` | 人类遥操作演示数据，用于 VLA 模仿学习训练 |

**任务实例**由模拟器在运行时加载，用于搭建评估场景。
**挑战赛演示**为离线轨迹数据集，用于通过监督学习训练 VLA 模型。

---

## 任务实例（`2025-challenge-task-instances/`）

为 BEHAVIOR-1K 的 50 个挑战任务预生成的场景配置。下载方式：

```bash
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

### 目录结构

```
2025-challenge-task-instances/
├── README.md
├── metadata/
│   ├── test_instances.csv              # 官方测试实例 ID（每任务 20 个）
│   ├── episodes.jsonl                  # Episode 级别统计（长度、位移等）
│   ├── B50_task_misc.csv               # 每任务元数据（所需房间、就绪标志）
│   └── B50_object_instance_ID.csv      # 任务相关物体名称映射
│
└── scenes/
    ├── house_double_floor_lower/       # 大多数任务使用的主场景
    ├── house_double_floor_upper/
    └── house_single_floor/
```

### 场景目录结构

每个场景目录（例如 `house_double_floor_lower/`）包含：

```
house_double_floor_lower/
└── json/
    ├── <scene>_task_<activity>_0_0_template.json               # 默认实例（id=0），完整场景快照
    ├── <scene>_task_<activity>_0_0_template-partial_rooms.json  # 同上，但仅加载任务相关房间
    └── <scene>_task_<activity>_instances/                       # 预生成实例目录
        ├── <scene>_task_<activity>_0_0_template-tro_state.json
        ├── <scene>_task_<activity>_0_1_template-tro_state.json
        ├── ...
        └── <scene>_task_<activity>_0_300_template-tro_state.json
```

### 文件命名规则

template 和 tro_state 文件均遵循相同的命名格式：

```
<scene_model>_task_<activity_name>_<definition_id>_<instance_id>_template[-tro_state].json
```

| 组成部分 | 示例 | 含义 |
|----------|------|------|
| `scene_model` | `house_double_floor_lower` | 该任务使用的 3D 场景 |
| `activity_name` | `turning_on_radio` | BEHAVIOR-1K 任务名称 |
| `definition_id` | `0` | BDDL 问题定义变体（50 个挑战任务均为 0） |
| `instance_id` | `242` | 特定的物体摆放/机器人位姿配置 |
| 后缀 | `_template.json` 或 `_template-tro_state.json` | 文件格式（见下方说明） |

### 文件格式

| 格式 | 后缀 | 大小 | 内容 | 用途 |
|------|------|------|------|------|
| **template** | `_template.json` | 较大 | 完整场景：所有物体的 `metadata`、`init_info`、`objects_info`、`state` | 初始场景构建（OmniGibson 从零构建场景时必需） |
| **tro_state** | `_template-tro_state.json` | 较小 | 仅任务相关物体 + 可选的 `robot_poses` | 在已加载场景上进行轻量级状态替换 |

**重要说明**：`tro_state` 文件不能用于从零初始化场景。它只存储 3-5 个任务相关物体（例如 `radio_receiver.n.01_1`、`table.n.02_1`）。初始场景加载需要完整的 `template.json`，之后 `tro_state` 文件可以就地替换物体摆放。

### `test_instances.csv`

每行列出一个任务的 20 个已验证实例 ID：

```csv
Task ID,Task,Public Test Instance IDs
0,turning_on_radio,"242, 295, 211, 203, 109, 181, 197, ..."
1,picking_up_trash,"196, 67, 155, 106, 161, 245, 171, ..."
```

这 20 个实例从约 300 个预生成实例中筛选，已验证物理有效且可完成。它们是 BEHAVIOR-1K 挑战赛的标准化评估集。

### `activity_definition_id` 与 `activity_instance_id`

这是 BEHAVIOR-1K 中的两个变化层级：

- **`activity_definition_id`**：指向一个 BDDL `problem{N}.bddl` 文件，定义逻辑任务约束（物体类型、空间关系、目标条件）。50 个挑战任务目前均只有 `definition_id=0`。

- **`activity_instance_id`**：某个 definition 的具体物理实例化。BDDL 约束求解器随机化物体摆放、机器人位姿等。每个 ID 产生不同的场景布局，同时满足相同的逻辑约束。

```
definition_id = 0  -->  problem0.bddl（逻辑："客厅桌上有收音机，目标：打开它"）
    ├── instance_id = 0    -->  收音机在位置 A，机器人在位姿 X
    ├── instance_id = 1    -->  收音机在位置 B，机器人在位姿 Y
    ├── ...
    └── instance_id = 300  -->  收音机在位置 N，机器人在位姿 Z
```

---

## 挑战赛演示（`2025-challenge-demos/`）

在 BEHAVIOR-1K 模拟器中录制的人类遥操作演示。人类操作员通过遥操作控制 R1Pro 机器人完成各项任务，完整轨迹（图像、动作、状态）被全程记录。

### 下载

提供完整数据集和轻量版子集：

- **完整版**：`2025-challenge-demos`（50 个任务共约 10,000 个 episode，每任务 200 个）
- **精简版**：`2025-challenge-demos-short`（12 个任务的子集，但元数据仍保留全部 10,000 个 episode 的记录）

两者共享相同的目录结构，数据格式为 LeRobot v2.1。

> **注意**：精简版的 `meta/info.json` 中 `total_episodes` 仍为 10000，`meta/episodes.jsonl` 也包含全部 10000 条记录，但实际只有 12 个任务的 parquet 和视频文件存在。

### 目录结构

```
2025-challenge-demos/
├── .cache/huggingface/                         # HuggingFace 下载缓存
├── meta/
│   ├── info.json                               # 数据集概览（见下方说明）
│   ├── tasks.jsonl                             # 任务索引 -> 任务名称 -> 自然语言描述
│   ├── episodes.jsonl                          # 逐 episode 统计（长度、位移）
│   ├── episodes_stats.jsonl                    # 逐 episode 动作分布统计
│   └── episodes/
│       └── task-{NNNN}/
│           └── episode_{NNNNNNNN}.json         # 该 episode 的完整 OmniGibson 配置快照
│
├── data/
│   └── task-{NNNN}/                            # 按任务索引分组（task-0000 = turning_on_radio）
│       └── episode_{NNNNNNNN}.parquet          # 轨迹数据（图像、动作、状态、时间戳）
│
├── annotations/
│   └── task-{NNNN}/
│       └── episode_{NNNNNNNN}.json             # 人工标注的技能分解
│
└── videos/
    └── task-{NNNN}/
        ├── observation.images.rgb.head/        # 头部相机视频（逐 episode）
        ├── observation.images.rgb.left_wrist/  # 左腕相机视频（逐 episode）
        ├── observation.images.rgb.right_wrist/ # 右腕相机视频（逐 episode）
        ├── observation.images.depth.*/         # 深度图视频
        ├── observation.images.seg_*/           # 实例分割视频
        └── episode_{NNNNNNNN}.mp4              # 综合概览视频
```

### `meta/info.json` 关键字段

```json
{
    "robot_type": "R1Pro",
    "total_episodes": 10000,
    "total_frames": 119094660,
    "total_tasks": 50,
    "fps": 30,
    "features": {
        "observation.images.rgb.head":       { "shape": [720, 720, 3] },
        "observation.images.rgb.left_wrist": { "shape": [480, 480, 3] },
        "observation.images.rgb.right_wrist":{ "shape": [480, 480, 3] },
        "observation.images.depth.*":        { "shape": [H, W, 3], "is_depth_map": true },
        "observation.images.seg_instance_id.*": { "shape": [H, W, 3] },
        "action":                            { "shape": [23], "dtype": "float32" },
        "observation.state":                 { "shape": [256], "dtype": "float32" },
        "observation.cam_rel_poses":         { "shape": [21], "dtype": "float32" }
    }
}
```

### `data/` -- 轨迹 Parquet 文件

每个 `.parquet` 文件包含一个完整的 episode。Parquet 列仅包含**非视频**特征（action、observation.state、timestamps 等），图像帧存储在 `videos/` 下的 MP4 视频中，通过 LeRobot 的 video backend 按 timestamp 索引解码。

Parquet 中的主要列：

| 列名 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `index` | int64 | (1,) | 帧在全数据集中的全局索引 |
| `episode_index` | int64 | (1,) | episode 编号（跨任务的全局索引，非从 0 开始） |
| `task_index` | int64 | (1,) | 所属任务编号（task-0000 -> 0） |
| `timestamp` | float64 | (1,) | 时间戳（秒），以 1/30s 递增 |
| `observation.state` | float32 | (256,) | R1Pro 全 proprioception 向量 |
| `observation.cam_rel_poses` | float32 | (21,) | 3 个摄像头相对位姿（3 x 7D） |
| `observation.task_info` | float32 | (46,) | 任务相关状态信息 |
| `action` | float32 | (23,) | 23 维动作向量 |

> **注意**：Episode 索引不是连续的（如 task-0000 的 episode_index 为 10, 20, 30, ..., 3000），这导致 LeRobot v2.1 的 `_get_query_indices` 索引越界。如需用于训练，应使用 `toolkits/behavior/prepare_behavior_task0000.py` 进行重索引。

### `annotations/` -- 技能分解

每个 episode 由人工标注者分解为顺序技能序列：

```json
{
    "task_name": "turning on radio",
    "skill_annotation": [
        {
            "skill_idx": 0,
            "skill_description": ["move to"],
            "object_id": [["radio_89"]],
            "frame_duration": [0, 265],
            "skill_type": ["navigation"]
        },
        {
            "skill_idx": 1,
            "skill_description": ["pick up from"],
            "object_id": [["radio_89", "coffee_table_koagbh_0"]],
            "frame_duration": [265, 1162],
            "skill_type": ["manipulation"]
        }
    ]
}
```

### 任务编号规则

`task-{NNNN}` 目录名对应任务索引（零填充至 4 位）：

| 目录 | 任务索引 | 任务名称 |
|------|----------|----------|
| `task-0000` | 0 | `turning_on_radio` |
| `task-0001` | 1 | `picking_up_trash` |
| `task-0006` | 6 | `hiding_Easter_eggs` |
| ... | ... | ... |

精简版中可能并非所有任务都存在。完整版包含全部 50 个任务。

### 动作维度说明

R1Pro 的 23 维动作向量结构（与 proprioception 中的关节对应）：

| 成分 | 维度 | 说明 |
|------|------|------|
| base velocity | 3 | 底盘平移/旋转速度 |
| trunk joints | 4 | 躯干关节位置 |
| left arm joints | 7 | 左臂 7 自由度关节 |
| right arm joints | 7 | 右臂 7 自由度关节 |
| left gripper | 1 | 左夹爪开合 |
| right gripper | 1 | 右夹爪开合 |
| **合计** | **23** | |

### 使用的场景实例

所有演示 episode 使用 `activity_instance_id=0`（默认场景 template）。精简版中每任务的 200 个 episode 仅在人类操作员的执行轨迹上有所不同，场景初始化完全相同。

---

## VLA 训练数据使用

关于如何将挑战赛演示数据用于 VLA SFT 训练（包括数据重索引、norm stats 计算、代码适配），详见 `pi05_doc/05-behavior-vla-sft.md`。

---

## 两类数据集的关系

```
任务实例（模拟器资源）                        挑战赛演示（训练数据）
======================================       ======================================
3D 场景、物体模型、机器人模型         <--     在这些场景中录制
每任务 300 个预生成实例               --      所有演示使用 instance_id=0
                                             
使用时机：评估运行时                          使用时机：VLA 模型训练
用途：搭建多样化评估场景                      用途：模仿学习的监督信号
格式：JSON（由 OmniGibson 加载）              格式：Parquet + MP4（由数据加载器读取）
```
