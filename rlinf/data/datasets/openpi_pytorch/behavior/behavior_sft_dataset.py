# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained BEHAVIOR-1K streaming LeRobot dataset for pi05 SFT.

Ported from ``rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py`` with
two changes relative to the old module:

* the installed-``openpi`` import is removed (the only transform base used here,
  :class:`DataTransformFn`, is taken from the vendored
  ``openpi_pytorch.policies.behavior_policy`` module instead);
* the streaming chunk partition is made *rank-aware* so that, under
  ``torchrun``/``DistributedDataParallel``, each rank streams a disjoint slice of
  the keyframe chunks (see :meth:`BehaviorSftDataset.__getitem__`).

The video/stat utilities (``hf_transform_to_torch``, ``aggregate_stats``,
``decode_video_frames`` and the per-camera video loaders) are the real ones from
``omnigibson.learning.utils`` rather than local copies. They are imported lazily
(inside the methods that use them, via :func:`_omnigibson_utils`) so that merely
importing this module never pulls OmniGibson — config validation must not trigger
any scene/asset load.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    STATS_PATH,
    TASKS_PATH,
    backward_compatible_episodes_stats,
    cast_stats_to_numpy,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    is_valid_version,
    load_info,
    load_json,
    load_jsonlines,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec
from torch.utils.data import Dataset, get_worker_info

# The vendored, self-contained transform base (replaces openpi.transforms.DataTransformFn).
from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    entry_to_subtask_text,
)
from rlinf.data.datasets.openpi_pytorch.behavior.skill_segments import (
    SkillSegments,
    build_skill_segments,
    resolve_frame_subtask,
)
from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    DataTransformFn,
)

logger = logging.getLogger("BehaviorSftDataset")

# ---------------------------------------------------------------------------
# Inlined constants (from omnigibson.learning.utils.eval_utils)
# ---------------------------------------------------------------------------

ROBOT_CAMERA_NAMES = {
    "R1Pro": {
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1::robot_r1:zed_link:Camera:0",
    },
}

TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_INDICES_TO_NAMES = {v: k for k, v in TASK_NAMES_TO_INDICES.items()}

ANNOTATIONS_PATH = "annotations"


# ---------------------------------------------------------------------------
# Lazy OmniGibson utility access
# ---------------------------------------------------------------------------


def _install_lerobot_compat_shim():
    """Alias ``lerobot.datasets.compute_stats`` to the pinned LeRobot's
    ``lerobot.common.datasets.compute_stats`` so OmniGibson's ``lerobot_utils`` (which
    imports ``_assert_type_and_shape`` from the new LeRobot layout) loads unchanged.

    This imports only LeRobot — never OmniGibson — so running it at module import does
    not regress the lazy-OmniGibson contract. It must be installed at import time
    (below) because spawned (``num_workers>0``) DataLoader workers unpickle this
    dataset's cached OmniGibson helper references, which re-imports
    ``omnigibson.learning.utils.lerobot_utils`` before any method runs; without the
    shim already in place that import raises ``ModuleNotFoundError: lerobot.datasets``.
    """
    import sys

    if "lerobot.datasets.compute_stats" in sys.modules:
        return
    try:
        import lerobot.datasets.compute_stats  # noqa: F401
    except ModuleNotFoundError:
        import types

        from lerobot.common.datasets.compute_stats import _assert_type_and_shape

        pkg = sys.modules.setdefault(
            "lerobot.datasets", types.ModuleType("lerobot.datasets")
        )
        shim = types.ModuleType("lerobot.datasets.compute_stats")
        shim._assert_type_and_shape = _assert_type_and_shape
        sys.modules["lerobot.datasets.compute_stats"] = shim
        pkg.compute_stats = shim


# Install the LeRobot compat shim at import time (LeRobot-only, no OmniGibson) so
# spawned DataLoader workers can unpickle this dataset's OmniGibson helper references.
_install_lerobot_compat_shim()


def _omnigibson_utils():
    """Lazily import the OmniGibson video/stat utilities used by this dataset.

    Imported on first call (never at module import) so that loading this module
    — e.g. during config validation — does not pull OmniGibson or trigger any
    scene/asset load. Returns ``(hf_transform_to_torch, aggregate_stats,
    decode_video_frames, OBS_LOADER_MAP)``.
    """
    from omnigibson.learning.utils.lerobot_utils import (
        aggregate_stats,
        decode_video_frames,
        hf_transform_to_torch,
    )
    from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP

    return hf_transform_to_torch, aggregate_stats, decode_video_frames, OBS_LOADER_MAP


# ---------------------------------------------------------------------------
# PromptFromLeRobotItem transform
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotItem(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset item's ``task`` field."""

    def __call__(self, data: dict) -> dict:
        return {**data, "prompt": data.pop("task")}


# ---------------------------------------------------------------------------
# BehaviorSftDatasetMetadata
# ---------------------------------------------------------------------------


class BehaviorSftDatasetMetadata(LeRobotDatasetMetadata):
    """LeRobot metadata extended with BEHAVIOR task filtering and skill annotations."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
    ):
        self.task_name_candidates = (
            set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        )
        self.modalities = set(modalities) if modalities else {"rgb"}
        self.camera_names = (
            set(cameras) if cameras else {"head", "left_wrist", "right_wrist"}
        )
        assert self.modalities.issubset({"rgb", "depth", "seg_instance_id"})
        assert self.camera_names.issubset(ROBOT_CAMERA_NAMES["R1Pro"])

        self.repo_id = repo_id
        self.revision = revision or CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(
                allow_patterns="meta/**", ignore_patterns="meta/episodes/**"
            )
            self.load_metadata()

    def load_metadata(self):
        """Load info, filtered tasks/episodes, annotations, and stats."""
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(
            self.root
        )
        valid_task_indices = [
            idx
            for idx, name in self.task_names.items()
            if name in self.task_name_candidates
        ]
        self.task_names = {self.task_names[idx] for idx in valid_task_indices}
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        self.annotations = self.load_annotations(self.root)
        import packaging.version

        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(
                self.stats, self.episodes
            )
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            _, aggregate_stats, _, _ = _omnigibson_utils()
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def load_tasks(self, local_dir: Path):
        """Load the ``tasks.jsonl`` mapping (index -> task text / task name)."""
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {
            item["task_index"]: item["task_name"]
            for item in sorted(tasks, key=lambda x: x["task_index"])
        }
        tasks_dict = {
            item["task_index"]: item["task"]
            for item in sorted(tasks, key=lambda x: x["task_index"])
        }
        task_to_task_index = {task: idx for idx, task in tasks_dict.items()}
        return tasks_dict, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path):
        """Load episodes belonging to the selected tasks."""
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path):
        """Load aggregate dataset stats (legacy <v2.1 datasets)."""
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path):
        """Load per-episode stats for the selected episodes."""
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def load_annotations(self, local_dir: Path):
        """Load per-episode skill annotations (the subtask label source)."""
        annotations_dir = local_dir / ANNOTATIONS_PATH
        if not annotations_dir.exists():
            return {}
        task_list = [
            task_id for task_id in annotations_dir.iterdir() if task_id.is_dir()
        ]
        return {
            int(episode.stem[8:]): load_json(episode)
            for task_id in task_list
            if int(task_id.name[5:]) in self.tasks
            for episode in sorted(task_id.iterdir())
        }

    def get_annotation_path(self, ep_index: int) -> Path:
        """Resolve the annotation file path for ``ep_index``."""
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(
            episode_chunk=ep_chunk, episode_index=ep_index
        )
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        """Resolve the meta-info file path for ``ep_index``."""
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(
            episode_chunk=ep_chunk, episode_index=ep_index
        )
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Template path for episode annotation files (from ``info``)."""
        return self.info.get("annotation_path")

    @property
    def metainfo_path(self) -> str | None:
        """Template path for episode meta-info files (from ``info``)."""
        return self.info.get("metainfo_path")

    @property
    def features(self) -> dict[str, dict]:
        """Image features filtered to the selected cameras/modalities."""
        features = {}
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


# ---------------------------------------------------------------------------
# BehaviorSftDataset
# ---------------------------------------------------------------------------


def partition_chunk_indices(
    num_chunks: int,
    *,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    """Return the chunk indices a single ``(rank, worker)`` pair streams.

    The distributed ``rank`` is folded into the per-worker stride so that every
    ``(rank, worker)`` pair receives a disjoint, non-overlapping set of chunk
    indices whose union over all ranks and workers covers every chunk. This is
    what lets the streaming dataset shard distinct data per rank under
    ``torchrun``/DDP: a ``DistributedSampler`` cannot, because the dataset
    ignores the ``idx`` it is handed and streams chunks internally.
    """
    global_worker_id = rank * num_workers + worker_id
    stride = world_size * num_workers
    return list(range(global_worker_id, num_chunks, stride))


class BehaviorSftDataset(LeRobotDataset):
    """Streaming BEHAVIOR-1K dataset for pi05 SFT (task filtering + chunk streaming).

    The dataset streams contiguous keyframe *chunks* of each episode rather than
    returning the frame at a random ``idx``: ``__getitem__`` ignores ``idx`` and
    advances an internal streaming cursor. Chunks are partitioned across data
    loader workers (and, in this port, across distributed ranks) so that every
    consumer sees a disjoint stream — see :meth:`__getitem__`.

    The text attached to each streamed frame is controlled by
    ``fine_grained_level``:

    * ``0`` — one text item: ``item["task"]`` is the episode's main-task text
      (the model prompt). Every frame is used.
    * ``1`` — two text items: ``item["task"]`` (still the main task, the model
      prompt) plus ``item["response"]``, the subtask label supervising the VLM.
      The label is resolved deterministically from the frame's own episode
      ``skill_annotation`` via :mod:`.skill_segments` (which frame belongs to
      which skill window) and :func:`.skill_language.entry_to_subtask_text`
      (that window's ``skill_description`` + ``object_id`` rendered as natural
      language). Because the text comes from each episode's own annotation, the
      dataset is correct across tasks whose skill sequences vary per episode —
      no fixed per-task label list is needed or used. Frames the resolver maps
      to no subtask (outside ``valid_duration``, gaps with ``enable_gap=False``,
      trailing gaps) are skipped by the streaming cursor.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # Custom arguments
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,
        train_rgb_type: str = "regular",
        return_seg_instance: bool = False,
        skill_list: list[str] | None = None,
        subtask_labels: dict[int, str] | None = None,
        enable_gap: bool = True,
        dist_rank: int | None = None,
        dist_world_size: int | None = None,
    ):
        import packaging.version

        # Weighted skill sampling was removed together with the orchestrator
        # machinery (its only label source); only the no-op values survive so
        # stale configs fail loudly instead of silently changing the data mix.
        if skill_list not in (None, ["all"]):
            raise ValueError(
                "BehaviorSftDataset no longer supports weighted skill sampling; "
                f"skill_list must be None or ['all'], got {skill_list!r}."
            )
        if fine_grained_level not in (0, 1):
            raise ValueError(
                f"fine_grained_level must be 0 or 1, got {fine_grained_level!r}."
            )
        # Subtask labels are resolved per frame from that frame's own episode
        # annotation (see `_attach_text`), so no per-task `subtask_labels` list is
        # required at level 1. The parameter is retained for API compatibility but
        # is not consulted for label text.
        if fine_grained_level == 1 and not chunk_streaming_using_keyframe:
            raise ValueError(
                "fine_grained_level=1 skips unlabeled frames and therefore "
                "requires chunk streaming (chunk_streaming_using_keyframe=True)."
            )

        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = (
            Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        )
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.fine_grained_level = int(fine_grained_level)
        self.subtask_labels = subtask_labels
        # Gap frames (between two skill windows) belong to the NEXT skill when
        # `enable_gap` is set, and are skipped from training otherwise. See
        # `skill_segments.resolve_frame_subtask` for the full edge rules.
        self.enable_gap = enable_gap
        # Explicit distributed identity captured in the MAIN process. The streaming
        # chunk partition is rank-aware, but DataLoader workers are SPAWNED (fresh
        # interpreters that do NOT inherit ``torch.distributed``), so reading
        # ``dist.get_rank()`` inside a worker returns 0 and every rank would replicate
        # rank 0's partition. Storing rank/world_size here (pickled into the worker)
        # lets ``_select_streaming_chunk`` partition by the correct per-rank id; we fall
        # back to ``torch.distributed`` only when these are not provided.
        self._dist_rank = dist_rank
        self._dist_world_size = dist_world_size
        # Real OmniGibson video/stat utilities, imported lazily here (never at module
        # import) and cached so the streaming hot path and `load_hf_dataset` can reuse
        # them without re-importing. See `_omnigibson_utils`.
        (
            self._hf_transform_to_torch,
            self._aggregate_stats,
            self._decode_video_frames,
            self._obs_loader_map,
        ) = _omnigibson_utils()

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.seed = seed
        if modalities is None:
            modalities = ["rgb"]
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = (
            set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        )
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]

        self.meta = BehaviorSftDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )

        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["episode_index"] // 1e4 in self.meta.tasks:
                epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [
                    epi_by_task[task_id][i]
                    for i in episodes
                    if i < len(epi_by_task[task_id])
                ]
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])

        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        self.shuffle = shuffle
        if self._chunk_streaming_using_keyframe:
            self.chunks = self._get_keyframe_chunk_indices()
            # The per-(rank, worker) chunk slice is resolved lazily on first
            # access (inside the worker process) by `_select_streaming_chunk`;
            # `shuffle` decides whether that slice is shuffled and entered at a
            # random chunk, or walked in order from the first chunk.
            self.current_streaming_chunk_idx = None
            self.current_streaming_frame_idx = None
            self.obs_loaders = {}
            self._should_obs_loaders_reload = True

        self.episode_data_index_pos = {
            ep_idx: i for i, ep_idx in enumerate(self.episodes)
        }
        logger.info("Total episodes: %d", len(self.episodes))

        if self.episodes is not None and self.meta._version >= packaging.version.parse(
            "v2.1"
        ):
            episodes_stats = [
                self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes
            ]
            self.stats = self._aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), (
                    f"Missing file: {self.root / fpath}"
                )
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )

        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {
                k: t.numpy() for k, t in self.episode_data_index.items()
            }
            check_timestamps_sync(
                timestamps,
                episode_indices,
                ep_data_index_np,
                self.fps,
                self.tolerance_s,
            )

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        # Subtask supervision resolves every frame through the deterministic
        # skill-segment table; an episode without a (valid) skill annotation
        # cannot be labeled, so it aborts construction instead of training on
        # silently mislabeled frames. The per-window ``skill_idx`` is bounded by
        # that episode's own number of skill windows (indices are dense and
        # contiguous 0..N-1 within an episode), so the subtask text is resolved
        # from the episode's own annotation and is correct even for tasks whose
        # skill sequence varies across episodes.
        self._skill_segments: dict[int, SkillSegments] = {}
        # Cache of converted subtask text keyed by (episode_id, skill_idx): one
        # skill window spans many frames, so the conversion is done once per
        # window rather than per frame.
        self._subtask_text_cache: dict[tuple[int, int], str] = {}
        if self.fine_grained_level == 1:
            for ep_id in self.episodes:
                annotation = self.meta.annotations.get(ep_id)
                if annotation is None:
                    raise ValueError(
                        f"episode {ep_id}: fine_grained_level=1 requires a skill "
                        "annotation under annotations/, but none was loaded."
                    )
                num_windows = len(annotation.get("skill_annotation") or [])
                self._skill_segments[ep_id] = build_skill_segments(
                    annotation, num_windows, episode_id=ep_id
                )

        self.omnigibson_mapping = {
            ep_idx: defaultdict(dict) for ep_idx in self.episodes
        }

    # -------------------------------------------------------------------------

    def get_episodes_file_paths(self) -> list[str]:
        """Return the data/meta/video file paths for the selected episodes."""
        episodes = (
            self.episodes
            if self.episodes is not None
            else list(self.meta.episodes.keys())
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        metainfo_path = getattr(self.meta, "metainfo_path", None)
        if metainfo_path:
            fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """Download only the selected tasks' data (and videos) from the hub."""
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(
                self.task_indices
            ):
                ignore_patterns.append(f"**/task-{task:04d}/**")
        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(
            allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )

    def pull_from_repo(self, allow_patterns=None, ignore_patterns=None):
        """Snapshot-download the dataset repo into ``self.root``."""
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max(1, os.cpu_count() - 2),
        )

    def load_hf_dataset(self):
        """Load the parquet frames for the selected episodes as a HF dataset."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx))
                for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(self._hf_transform_to_torch)
        return hf_dataset

    def _select_streaming_chunk(self) -> None:
        """Pick this ``(rank, worker)``'s active chunk set and starting frame.

        Reads the distributed ``rank`` / ``world_size`` and the data-loader
        ``worker_id`` / ``num_workers``, folds them into the keyframe-chunk
        partition (via :func:`partition_chunk_indices`), shuffles the resulting
        chunks with a per-``(rank, worker)`` seed, and sets the streaming cursor
        to the start of the chosen chunk. Folding the rank in is what makes each
        distributed rank stream a disjoint set of chunks (a ``DistributedSampler``
        cannot, since this dataset ignores ``idx``).
        """
        # Prefer the explicit rank/world_size captured at construction (in the main
        # process); spawned DataLoader workers cannot read ``torch.distributed``, so
        # without these every rank would replicate rank 0's chunk partition.
        if self._dist_rank is not None and self._dist_world_size is not None:
            rank, world_size = self._dist_rank, self._dist_world_size
        else:
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        global_worker_id = rank * num_workers + worker_id
        if not hasattr(self, "_active_chunks") or self._active_chunks is None:
            indices = partition_chunk_indices(
                len(self.chunks),
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            worker_chunks = [self.chunks[i] for i in indices]
            if self.shuffle:
                rng = np.random.default_rng(self.seed + global_worker_id)
                rng.shuffle(worker_chunks)
            self._active_chunks = worker_chunks
        if self.shuffle:
            rng = np.random.default_rng(self.seed + global_worker_id)
            self.current_streaming_chunk_idx = rng.integers(
                0, len(self._active_chunks)
            ).item()
        else:
            self.current_streaming_chunk_idx = 0
        self.current_streaming_frame_idx = self._active_chunks[
            self.current_streaming_chunk_idx
        ][0]

    def __getitem__(self, idx) -> dict:
        """Return the next streamed frame (the ``idx`` argument is ignored).

        In streaming mode the dataset maintains a per-consumer cursor over a
        disjoint slice of the keyframe chunks. The slice is selected so that:

        * each data loader worker on a rank streams a different set of chunks, and
        * each distributed rank streams a different set of chunks.

        Because the streaming dataset partitions itself here, a
        ``DistributedSampler`` has no effect on it (the sampler only reorders
        ``idx`` values, which are ignored). Folding the distributed rank into the
        chunk stride is therefore required to avoid every rank seeing identical
        data under ``torchrun``/DDP.

        At ``fine_grained_level=1``, frames the segment resolver maps to no
        subtask are skipped by advancing the cursor in a loop (gaps can span
        hundreds of consecutive frames, so this must not recurse).
        """
        if not self._chunk_streaming_using_keyframe:
            item = super().__getitem__(idx)
            self._attach_text(item)
            return item

        # Streaming mode
        if self.current_streaming_chunk_idx is None:
            self._select_streaming_chunk()

        while True:
            if (
                self.current_streaming_frame_idx
                >= self._active_chunks[self.current_streaming_chunk_idx][1]
            ):
                self.current_streaming_chunk_idx += 1
                if self.current_streaming_chunk_idx >= len(self._active_chunks):
                    self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self._active_chunks[
                    self.current_streaming_chunk_idx
                ][0]
                self._should_obs_loaders_reload = True

            item = self.hf_dataset[self.current_streaming_frame_idx]
            if "observation.task_info" in item:
                item.pop("observation.task_info")
            ep_idx = item["episode_index"].item()

            if self._should_obs_loaders_reload:
                for loader in self.obs_loaders.values():
                    loader.close()
                self.obs_loaders = {}
                self.current_streaming_episode_idx = ep_idx
                for vid_key in self.meta.video_keys:
                    kwargs = {}
                    task_id = item["task_index"].item()
                    if "rgb" in vid_key:
                        kwargs["train_rgb_type"] = self.train_rgb_type
                    loader_cls = self._obs_loader_map.get(vid_key.split(".")[2])
                    if loader_cls is None:
                        continue
                    self.obs_loaders[vid_key] = iter(
                        loader_cls(
                            data_path=self.root,
                            task_id=task_id,
                            camera_id=vid_key.split(".")[-1],
                            demo_id=f"{ep_idx:08d}",
                            start_idx=self._active_chunks[
                                self.current_streaming_chunk_idx
                            ][2],
                            start_idx_is_keyframe=False,
                            batch_size=1,
                            stride=1,
                            **kwargs,
                        )
                    )
                self._should_obs_loaders_reload = False

            # Subtask resolution decides whether this frame is used BEFORE any
            # per-frame decode work; skipped frames still consume one frame from
            # every video loader so the loaders stay in lockstep with the cursor.
            subtask_index = None
            if self.fine_grained_level == 1:
                frame_index = round(item["timestamp"].item() * self.fps)
                subtask_index = resolve_frame_subtask(
                    self._skill_segments[ep_idx], frame_index, self.enable_gap
                )
                if subtask_index is None:
                    self.current_streaming_frame_idx += 1
                    for key in self.obs_loaders:
                        next(self.obs_loaders[key])[0]
                    continue

            if self.delta_indices is not None:
                query_indices, padding = self._get_query_indices(
                    self.current_streaming_frame_idx, ep_idx
                )
                query_result = self._query_hf_dataset(query_indices)
                item = {**item, **padding}
                for key, val in query_result.items():
                    item[key] = val

            for key in self.obs_loaders:
                item[key] = next(self.obs_loaders[key])[0]

            if self.image_transforms is not None:
                image_keys = self.meta.camera_keys
                for cam in image_keys:
                    item[cam] = self.image_transforms(item[cam])

            self._attach_text(item, subtask_index)
            self.current_streaming_frame_idx += 1
            return item

    def _attach_text(self, item: dict, subtask_index: int | None = None) -> None:
        """Attach the per-frame text fields to ``item``.

        Always sets ``item["task"]``, the episode's main-task text (the model
        prompt at every level). At ``fine_grained_level=1`` it additionally sets
        ``item["response"]``, the subtask label supervising the VLM, resolved
        from the frame's own episode annotation. The non-streaming path passes
        ``subtask_index=None`` and resolves it here.
        """
        ep_idx = item["episode_index"].item()
        item["task"] = self.meta.episodes[ep_idx]["tasks"][0]
        if self.fine_grained_level != 1:
            return
        if subtask_index is None:
            frame_index = round(item["timestamp"].item() * self.fps)
            subtask_index = resolve_frame_subtask(
                self._skill_segments[ep_idx], frame_index, self.enable_gap
            )
            if subtask_index is None:
                raise ValueError(
                    f"episode {ep_idx} frame {frame_index} has no subtask label; "
                    "unlabeled frames are only skippable in streaming mode."
                )
        item["response"] = self._resolve_subtask_text(ep_idx, subtask_index)

    def _resolve_subtask_text(self, ep_idx: int, skill_idx: int) -> str:
        """Return the natural-language subtask for one skill window of an episode.

        The window's ``skill_description`` + ``object_id`` are converted by
        :func:`.skill_language.entry_to_subtask_text`. Results are cached per
        ``(ep_idx, skill_idx)`` because a window spans many frames. A missing
        annotation or an out-of-range ``skill_idx`` fails loudly rather than
        emitting a silently mislabeled frame.
        """
        cache_key = (ep_idx, skill_idx)
        cached = self._subtask_text_cache.get(cache_key)
        if cached is not None:
            return cached

        annotation = self.meta.annotations.get(ep_idx)
        if annotation is None:
            raise ValueError(
                f"episode {ep_idx}: no skill annotation loaded, cannot resolve "
                f"subtask text for skill_idx {skill_idx}."
            )
        windows = annotation.get("skill_annotation") or []
        if not 0 <= skill_idx < len(windows):
            raise ValueError(
                f"episode {ep_idx}: skill_idx {skill_idx} is out of range for the "
                f"{len(windows)} skill window(s) in this episode's annotation."
            )
        text = entry_to_subtask_text(
            windows[skill_idx],
            task_name=self.meta.episodes[ep_idx]["tasks"][0],
            episode_id=ep_idx,
            skill_idx=skill_idx,
        )
        self._subtask_text_cache[cache_key] = text
        return text

    def _get_query_indices(self, idx: int, ep_idx: int):
        """Compute action-horizon query indices and per-key padding masks."""
        ep_idx_pos = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx_pos]
        ep_end = self.episode_data_index["to"][ep_idx_pos]
        query_indices = {
            key: [
                max(ep_start.item(), min(ep_end.item() - 1, idx + delta))
                for delta in delta_idx
            ]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": th.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps, ep_idx):
        """Decode the requested video frames for ``ep_idx``."""
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = self._decode_video_frames(
                video_path, query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames.squeeze(0)
        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250):
        """Split each episode into contiguous ``chunk_size``-frame keyframe chunks.

        Returns a list of ``(global_start, global_end, local_start)`` tuples where
        ``global_*`` index into the flat HF dataset and ``local_start`` is the
        in-episode frame offset used to seek the per-camera video loaders.
        """
        episode_lengths = {
            ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()
        }
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks
