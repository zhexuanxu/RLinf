# BEHAVIOR-1K Data Reference

This document describes the two main data sources used by RLinf's BEHAVIOR evaluation pipeline.

## Overview

| Dataset | Location | Purpose |
|---------|----------|---------|
| Task Instances | `$OMNIGIBSON_DATA_PATH/2025-challenge-task-instances/` | Pre-generated scene initializations for evaluation (object placements, robot poses) |
| Challenge Demos | `/path/to/2025-challenge-demos/` | Human teleoperation demonstrations for VLA training (imitation learning) |

The **task instances** are loaded by the simulator at runtime to set up evaluation scenes.
The **challenge demos** are offline trajectory datasets used to train VLA models via supervised learning.

---

## Task Instances (`2025-challenge-task-instances/`)

Pre-generated scene configurations for BEHAVIOR-1K's 50 challenge tasks. Downloaded via:

```bash
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

### Directory Layout

```
2025-challenge-task-instances/
├── README.md
├── metadata/
│   ├── test_instances.csv              # Official test instance IDs (20 per task)
│   ├── episodes.jsonl                  # Episode-level statistics (length, displacement, etc.)
│   ├── B50_task_misc.csv               # Per-task metadata (required rooms, readiness flags)
│   └── B50_object_instance_ID.csv      # Task-relevant object name mapping
│
└── scenes/
    ├── house_double_floor_lower/       # Primary scene for most tasks
    ├── house_double_floor_upper/
    └── house_single_floor/
```

### Scene Directory Structure

Each scene directory (e.g., `house_double_floor_lower/`) contains:

```
house_double_floor_lower/
└── json/
    ├── <scene>_task_<activity>_0_0_template.json               # Default instance (id=0), full scene snapshot
    ├── <scene>_task_<activity>_0_0_template-partial_rooms.json  # Same, but only task-relevant rooms loaded
    └── <scene>_task_<activity>_instances/                       # Pre-generated instances directory
        ├── <scene>_task_<activity>_0_0_template-tro_state.json
        ├── <scene>_task_<activity>_0_1_template-tro_state.json
        ├── ...
        └── <scene>_task_<activity>_0_300_template-tro_state.json
```

### Filename Convention

Both template and tro_state files follow the same naming pattern:

```
<scene_model>_task_<activity_name>_<definition_id>_<instance_id>_template[-tro_state].json
```

| Component | Example | Meaning |
|-----------|---------|---------|
| `scene_model` | `house_double_floor_lower` | The 3D scene used for this task |
| `activity_name` | `turning_on_radio` | BEHAVIOR-1K task name |
| `definition_id` | `0` | BDDL problem definition variant (always 0 for the 50 challenge tasks) |
| `instance_id` | `242` | Specific object placement / robot pose configuration |
| suffix | `_template.json` or `_template-tro_state.json` | File format (see below) |

### File Formats

| Format | Suffix | Size | Contents | Use Case |
|--------|--------|------|----------|----------|
| **template** | `_template.json` | Large | Full scene: `metadata`, `init_info`, `objects_info`, `state` for ALL objects | Initial scene construction (required by OmniGibson to build the scene from scratch) |
| **tro_state** | `_template-tro_state.json` | Small | Only task-relevant objects + optional `robot_poses` | Lightweight state replacement on an already-loaded scene |

**Important**: A `tro_state` file cannot be used to initialize a scene from scratch. It only stores 3-5 task-relevant objects (e.g., `radio_receiver.n.01_1`, `table.n.02_1`). The full `template.json` is needed for the initial scene load, after which `tro_state` files can swap object placements in-place.

### `test_instances.csv`

Each row lists 20 validated instance IDs for one task:

```csv
Task ID,Task,Public Test Instance IDs
0,turning_on_radio,"242, 295, 211, 203, 109, 181, 197, ..."
1,picking_up_trash,"196, 67, 155, 106, 161, 245, 171, ..."
```

These 20 instances per task are selected from the ~300 pre-generated instances and have been verified to be physically valid and completable. They are used as the standardized evaluation set for the BEHAVIOR-1K challenge.

### `activity_definition_id` vs `activity_instance_id`

These are two levels of variation in BEHAVIOR-1K:

- **`activity_definition_id`**: Points to a BDDL `problem{N}.bddl` file that defines the logical task constraints (which object types, spatial relations, goal conditions). All 50 challenge tasks currently have only `definition_id=0`.

- **`activity_instance_id`**: A specific physical instantiation of a definition. The BDDL constraint solver randomizes object placement, robot pose, etc. Each ID produces a different scene layout while satisfying the same logical constraints.

```
definition_id = 0  -->  problem0.bddl (logical: "radio on table in living room, goal: toggle on")
    ├── instance_id = 0    -->  radio at position A, robot at pose X
    ├── instance_id = 1    -->  radio at position B, robot at pose Y
    ├── ...
    └── instance_id = 300  -->  radio at position N, robot at pose Z
```

---

## Challenge Demos (`2025-challenge-demos/`)

Human teleoperation demonstrations recorded in the BEHAVIOR-1K simulator. A human operator controls the R1Pro robot via teleoperation to complete each task, and the full trajectory (images, actions, states) is recorded.

### Download

The full dataset and a lightweight subset are available:

- **Full**: `2025-challenge-demos` (~10,000 episodes across 50 tasks)
- **Short**: `2025-challenge-demos-short` (subset of ~2,400 episodes across 12 tasks)

Both share the same directory layout.

### Directory Layout

```
2025-challenge-demos/
├── .cache/huggingface/                         # HuggingFace download cache
├── meta/
│   ├── info.json                               # Dataset overview (see below)
│   ├── tasks.jsonl                             # Task index -> task name -> natural language description
│   ├── episodes.jsonl                          # Per-episode statistics (length, displacement)
│   ├── episodes_stats.jsonl                    # Per-episode action distribution statistics
│   └── episodes/
│       └── task-{NNNN}/
│           └── episode_{NNNNNNNN}.json         # Full OmniGibson config snapshot for this episode
│
├── data/
│   └── task-{NNNN}/                            # Grouped by task index (task-0000 = turning_on_radio)
│       └── episode_{NNNNNNNN}.parquet          # Trajectory data (images, actions, states, timestamps)
│
├── annotations/
│   └── task-{NNNN}/
│       └── episode_{NNNNNNNN}.json             # Human-annotated skill decomposition
│
└── videos/
    └── task-{NNNN}/
        ├── observation.images.rgb.head/        # Head camera video per episode
        ├── observation.images.rgb.left_wrist/  # Left wrist camera video per episode
        ├── observation.images.rgb.right_wrist/ # Right wrist camera video per episode
        ├── observation.images.depth.*/         # Depth map videos
        ├── observation.images.seg_*/           # Instance segmentation videos
        └── episode_{NNNNNNNN}.mp4              # Composite overview video
```

### `meta/info.json` Key Fields

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

### `data/` — Trajectory Parquet Files

Each `.parquet` file contains one complete episode with columns matching the features in `info.json`. Video frames are stored separately in `videos/` and referenced by frame index.

### `annotations/` — Skill Decomposition

Each episode is decomposed into sequential skills by human annotators:

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

### Task Numbering Convention

The `task-{NNNN}` directory name corresponds to the task index (zero-padded to 4 digits):

| Directory | Task Index | Task Name |
|-----------|-----------|-----------|
| `task-0000` | 0 | `turning_on_radio` |
| `task-0001` | 1 | `picking_up_trash` |
| `task-0006` | 6 | `hiding_Easter_eggs` |
| ... | ... | ... |

Not all tasks may be present in the short version. The full dataset contains all 50 tasks.

### Scene Instance Used

All demonstration episodes use `activity_instance_id=0` (the default scene template). The 200 episodes per task in the short version differ only in the human operator's execution trajectory, not in the scene initialization.

---

## Relationship Between the Two Datasets

```
Task Instances (simulator assets)              Challenge Demos (training data)
======================================         ======================================
3D scenes, object models, robot models  <--    Recorded inside these scenes
300 pre-generated instances per task     --     All demos use instance_id=0
                                               
Used at: evaluation runtime                    Used at: VLA model training
Purpose: set up diverse eval scenes            Purpose: imitation learning supervision
Format: JSON (loaded by OmniGibson)            Format: Parquet + MP4 (read by dataloaders)
```
