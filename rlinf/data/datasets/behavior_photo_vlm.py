# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Photo-based VLM SFT dataset for BEHAVIOR skill prediction (Qwen models).

Reads pre-extracted JPEG frames from disk instead of decoding H.265 video
on-the-fly.

Entry point:
    ``create_behavior_photo_data_loader_qwen`` — factory function called by
    ``FSDPVlmSftWorker.build_dataloader``.
"""

from __future__ import annotations

import bisect
import json
import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from rlinf.agents.dualsystem.prompts import build_vlm_answer, build_vlm_user_text
from rlinf.data.datasets.item import SftDatasetItem

logger = logging.getLogger(__name__)

# Task name -> index mapping
TASK_NAMES_TO_INDICES = {
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

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _load_task_descriptions(data_root: str) -> dict[int, str]:
    """Load task_index -> task_description from tasks.jsonl."""
    tasks_path = Path(data_root) / "meta" / "tasks.jsonl"
    result = {}
    with tasks_path.open() as f:
        for line in f:
            item = json.loads(line)
            result[item["task_index"]] = item["task"]
    return result


def _load_episodes(data_root: str, target_task_ids: set[int]) -> list[dict]:
    """Load episodes.jsonl, filter by task IDs. Returns sorted episode list."""
    episodes_path = Path(data_root) / "meta" / "episodes.jsonl"
    episodes = []
    with episodes_path.open() as f:
        for line in f:
            ep = json.loads(line)
            task_id = int(ep["episode_index"] // 1e4)
            if task_id in target_task_ids:
                episodes.append(ep)
    return sorted(episodes, key=lambda x: x["episode_index"])


def _load_skill_boundaries(
    data_root: str, episode_indices: list[int]
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """Load annotation JSONs and build bisect-compatible skill boundaries."""
    skill_starts: dict[int, list[int]] = {}
    skill_ends: dict[int, list[int]] = {}

    for ep_idx in episode_indices:
        task_id = int(ep_idx // 1e4)
        ann_path = (
            Path(data_root) / "annotations" / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}.json"
        )
        with ann_path.open() as f:
            ann = json.load(f)
        skills = sorted(ann["skill_annotation"], key=lambda s: s["skill_idx"])
        skill_starts[ep_idx] = [s["frame_duration"][0] for s in skills]
        skill_ends[ep_idx] = [s["frame_duration"][1] for s in skills]

    return skill_starts, skill_ends


def _get_skill_label(
    ep_idx: int,
    frame_index: int,
    skill_starts: dict[int, list[int]],
    skill_ends: dict[int, list[int]],
    skill_labels: dict[int, str],
) -> str:
    """Bisect lookup: frame -> skill label.

    Gap frames are split at the midpoint between adjacent skills.
    """
    start_frames = skill_starts[ep_idx]
    end_frames = skill_ends[ep_idx]

    skill_idx = bisect.bisect_right(start_frames, frame_index) - 1
    skill_idx = max(0, min(skill_idx, len(start_frames) - 1))

    if frame_index < end_frames[skill_idx]:
        return skill_labels[skill_idx]

    if skill_idx + 1 < len(start_frames):
        gap_start = end_frames[skill_idx]
        gap_end = start_frames[skill_idx + 1]
        midpoint = (gap_start + gap_end) // 2
        if frame_index < midpoint:
            return skill_labels[skill_idx]
        else:
            return skill_labels[skill_idx + 1]

    return skill_labels[skill_idx]


def _build_task_skill_labels(
    task_subtasks: dict[str, list[str]],
) -> dict[int, dict[int, str]]:
    """Build per-task skill_labels dicts from task_subtasks config.

    Returns {task_id: {skill_idx: subtask_label, ...}, ...}.
    """
    result: dict[int, dict[int, str]] = {}
    for task_name, subtasks in task_subtasks.items():
        result[TASK_NAMES_TO_INDICES[task_name]] = {
            i: s for i, s in enumerate(subtasks)
        }
    return result


# ---------------------------------------------------------------------------
# Episode split helpers
# ---------------------------------------------------------------------------


def _get_train_eval_episode_indices(
    data_root: str,
    tasks: list[str],
    eval_ratio: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Deterministic episode split: last ``eval_ratio`` fraction = eval."""
    task_name_to_index: dict[str, int] = {}
    tasks_path = Path(data_root) / "meta" / "tasks.jsonl"
    with tasks_path.open() as f:
        for line in f:
            item = json.loads(line)
            task_name_to_index[item["task_name"]] = item["task_index"]

    target_task_ids = {task_name_to_index[t] for t in tasks}

    episodes_path = Path(data_root) / "meta" / "episodes.jsonl"
    ep_count = 0
    with episodes_path.open() as f:
        for line in f:
            item = json.loads(line)
            if int(item["episode_index"] // 1e4) in target_task_ids:
                ep_count += 1

    if ep_count == 0:
        return [], []

    all_indices = list(range(ep_count))
    n_eval = max(1, int(ep_count * eval_ratio))
    return all_indices[: ep_count - n_eval], all_indices[ep_count - n_eval :]


def _get_agentic_episode_split(
    sft_data_dir: str,
    video_data_root: str,
    task_ids: set[int],
    eval_ratio: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Deterministic agentic episode split.

    Expects ``sft_data_dir/task-XXXX/episode_*.json`` layout.
    """
    sft_eps: list[int] = []
    sft_path = Path(sft_data_dir)
    for task_dir in sorted(sft_path.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
            continue
        tid = int(task_dir.name.replace("task-", ""))
        if tid not in task_ids:
            continue
        for f in sorted(task_dir.iterdir()):
            if f.suffix == ".json":
                sft_eps.append(int(f.stem.replace("episode_", "")))
    sft_eps.sort()

    episodes_path = Path(video_data_root) / "meta" / "episodes.jsonl"
    all_task_eps: list[int] = []
    with episodes_path.open() as f:
        for line in f:
            item = json.loads(line)
            if int(item["episode_index"] // 1e4) in task_ids:
                all_task_eps.append(item["episode_index"])
    all_task_eps.sort()
    ep_to_pos = {ep: i for i, ep in enumerate(all_task_eps)}

    sft_positions = sorted(ep_to_pos[ep] for ep in sft_eps if ep in ep_to_pos)

    n_eval = max(1, int(len(sft_positions) * eval_ratio))
    return sft_positions[: len(sft_positions) - n_eval], sft_positions[len(sft_positions) - n_eval :]


# ---------------------------------------------------------------------------
# Qwen encoding helper
# ---------------------------------------------------------------------------


def _encode_qwen_sft_item(
    pil_img: Image.Image,
    user_text: str,
    answer_text: str,
    subtask_str: str,
    processor,
    eval_mode: bool,
) -> SftDatasetItem:
    """Encode one sample into an SftDatasetItem via Qwen processor.

    Uses ``process_vision_info`` from ``qwen_vl_utils`` to align image
    processing with the VLM inference policies (Qwen2.5-VL / Qwen3-VL).
    """
    from qwen_vl_utils import process_vision_info

    image_content = [
        {"type": "image", "image": pil_img},
        {"type": "text", "text": user_text},
    ]
    prompt_messages: list[dict] = [{"role": "user", "content": image_content}]

    with_answer = [
        (
            *prompt_messages,
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        )
    ]
    without_answer = [tuple(prompt_messages)]

    def _encode(messages):
        rendered = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=rendered,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            return_tensors="pt",
            padding=True,
        )

    prompt_inputs = _encode(with_answer)
    label_inputs = _encode(without_answer)

    if eval_mode:
        input_ids = label_inputs.pop("input_ids")
        attention_mask = label_inputs.pop("attention_mask")
        label_mask = attention_mask
        multi_modal = label_inputs
    else:
        input_ids = prompt_inputs.pop("input_ids")
        attention_mask = prompt_inputs.pop("attention_mask")
        label_mask = label_inputs.pop("attention_mask")
        multi_modal = prompt_inputs

    if isinstance(input_ids, torch.Tensor):
        if input_ids.dim() == 2 and input_ids.size(0) == 1:
            input_ids = input_ids.squeeze(0)
        input_ids = input_ids.to(dtype=torch.long)
    else:
        input_ids = torch.tensor(input_ids, dtype=torch.long)

    return SftDatasetItem(
        prompt=input_ids,
        length=int(input_ids.numel()),
        idx=0,
        image_data=[pil_img],
        answer=subtask_str,
        prompt_text=user_text,
        attention_mask=attention_mask,
        label_mask=label_mask,
        meta=None,
        multi_modal_inputs={k: v for k, v in multi_modal.items()},
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


class BehaviorPhotoSkillDataset(Dataset):
    """All-frames dataset for ``behavior_skill_sft`` (annotation-based skill labels)."""

    def __init__(
        self,
        photo_root: str,
        data_root: str,
        task_names: list[str],
        episode_indices: list[int] | None,
        processor,
        tokenizer,
        eval_mode: bool = False,
        enable_reasoning: bool = False,
        enable_memory: bool = False,
        simple_skill: bool = True,
        task_subtasks: dict[str, list[str]] | None = None,
        skill_library: list[str] | None = None,
        aug_option: str = "none",
    ):
        self.photo_root = photo_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode
        self.enable_reasoning = enable_reasoning
        self.enable_memory = enable_memory
        self.simple_skill = simple_skill
        self.task_subtasks = task_subtasks or {}
        self.skill_library = skill_library or []
        self.aug_option = aug_option

        self._task_skill_labels = _build_task_skill_labels(self.task_subtasks)

        target_task_ids = {TASK_NAMES_TO_INDICES[t] for t in task_names}
        self.task_descriptions = _load_task_descriptions(data_root)
        all_episodes = _load_episodes(data_root, target_task_ids)

        if episode_indices is not None:
            self.episodes = [all_episodes[i] for i in episode_indices]
        else:
            self.episodes = all_episodes

        self.ep_indices = [ep["episode_index"] for ep in self.episodes]
        self.ep_lengths = [ep["length"] for ep in self.episodes]
        self.cum_lengths = np.cumsum([0] + self.ep_lengths).tolist()
        self.total_frames = self.cum_lengths[-1]

        self.skill_starts, self.skill_ends = _load_skill_boundaries(
            data_root, self.ep_indices
        )

        logger.info(
            "BehaviorPhotoSkillDataset: %d episodes, %d total frames",
            len(self.episodes), self.total_frames,
        )

    def __len__(self) -> int:
        return self.total_frames

    def __getitem__(self, idx: int) -> SftDatasetItem:
        ep_pos = bisect.bisect_right(self.cum_lengths, idx) - 1
        frame_number = idx - self.cum_lengths[ep_pos]
        ep_idx = self.ep_indices[ep_pos]
        task_id = int(ep_idx // 1e4)

        img_path = os.path.join(
            self.photo_root,
            f"task-{task_id:04d}",
            f"episode_{ep_idx:08d}",
            f"frame_{frame_number:06d}.jpg",
        )
        pil_img = Image.open(img_path).convert("RGB")

        task_desc = self.task_descriptions[task_id]
        task_name = TASK_INDICES_TO_NAMES[task_id]
        skill_labels = self._task_skill_labels[task_id]
        skill_label = _get_skill_label(
            ep_idx, frame_number, self.skill_starts, self.skill_ends, skill_labels
        )

        user_text = build_vlm_user_text(
            task_description=task_desc,
            memory="",
            enable_memory=self.enable_memory,
            skill_aug=self.skill_library,
            subtask_aug=self.task_subtasks.get(task_name),
            aug_option=self.aug_option,
        )
        answer_text = build_vlm_answer(
            reasoning="",
            memory="",
            subtask=skill_label,
            enable_reasoning=self.enable_reasoning,
            enable_memory=self.enable_memory,
        )

        return _encode_qwen_sft_item(
            pil_img, user_text, answer_text, skill_label,
            self.processor, self.eval_mode,
        )


class BehaviorPhotoAgenticDataset(Dataset):
    """SFT-JSON-frame dataset for ``behavior_agentic_sft``.

    Expects ``sft_data_dir/task-XXXX/episode_*.json`` layout.
    """

    def __init__(
        self,
        photo_root: str,
        data_root: str,
        sft_data_dir: str,
        task_names: list[str],
        episode_indices: list[int] | None,
        processor,
        tokenizer,
        eval_mode: bool = False,
        enable_reasoning: bool = False,
        enable_memory: bool = False,
        simple_skill: bool = True,
        task_subtasks: dict[str, list[str]] | None = None,
        skill_library: list[str] | None = None,
        aug_option: str = "none",
    ):
        self.photo_root = photo_root
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode
        self.enable_reasoning = enable_reasoning
        self.enable_memory = enable_memory
        self.simple_skill = simple_skill
        self.task_subtasks = task_subtasks or {}
        self.skill_library = skill_library or []
        self.aug_option = aug_option

        self._task_skill_labels = _build_task_skill_labels(self.task_subtasks)
        self.task_descriptions = _load_task_descriptions(data_root)

        target_task_ids = {TASK_NAMES_TO_INDICES[t] for t in task_names}
        all_episodes = _load_episodes(data_root, target_task_ids)
        if episode_indices is not None:
            valid_ep_idxs = {all_episodes[i]["episode_index"] for i in episode_indices}
        else:
            valid_ep_idxs = {ep["episode_index"] for ep in all_episodes}

        if self.simple_skill:
            self.skill_starts, self.skill_ends = _load_skill_boundaries(
                data_root, list(valid_ep_idxs)
            )
        else:
            self.skill_starts, self.skill_ends = {}, {}

        # Load SFT JSON files: sft_data_dir/task-XXXX/episode_*.json
        self.samples: list[dict] = []
        sft_path = Path(sft_data_dir)
        for task_dir in sorted(sft_path.iterdir()):
            if not task_dir.is_dir() or not task_dir.name.startswith("task-"):
                continue
            tid = int(task_dir.name.replace("task-", ""))
            if tid not in target_task_ids:
                continue
            task_name = TASK_INDICES_TO_NAMES[tid]
            skill_labels = self._task_skill_labels.get(tid, {})

            for fpath in sorted(task_dir.iterdir()):
                if fpath.suffix != ".json":
                    continue
                ep_idx = int(fpath.stem.replace("episode_", ""))
                if ep_idx not in valid_ep_idxs:
                    continue

                with fpath.open() as f:
                    data = json.load(f)
                for r in data["results"]:
                    if not r["result_used"]:
                        continue
                    old_mem = r["request_input"]["old_memory"]
                    frame_number = r["frame_number"]

                    if self.simple_skill and skill_labels and ep_idx in self.skill_starts:
                        subtask_label = _get_skill_label(
                            ep_idx, frame_number,
                            self.skill_starts, self.skill_ends, skill_labels,
                        )
                    else:
                        subtask_label = r["model_response"]["subtask"]

                    self.samples.append({
                        "ep_idx": ep_idx,
                        "task_id": tid,
                        "task_name": task_name,
                        "frame_number": frame_number,
                        "old_memory": old_mem,
                        "reasoning": r["model_response"]["reasoning"],
                        "new_memory": r["model_response"]["new_memory"],
                        "subtask": r["model_response"]["subtask"],
                        "subtask_label": subtask_label,
                    })

        logger.info(
            "BehaviorPhotoAgenticDataset: %d samples from %d episodes",
            len(self.samples),
            len({s["ep_idx"] for s in self.samples}),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> SftDatasetItem:
        sample = self.samples[idx]
        ep_idx = sample["ep_idx"]
        frame_number = sample["frame_number"]
        task_id = sample["task_id"]
        task_name = sample["task_name"]

        img_path = os.path.join(
            self.photo_root,
            f"task-{task_id:04d}",
            f"episode_{ep_idx:08d}",
            f"frame_{frame_number:06d}.jpg",
        )
        pil_img = Image.open(img_path).convert("RGB")

        subtask = sample["subtask_label"] if self.simple_skill else sample["subtask"]

        old_memory = sample["old_memory"]
        old_memory_str = old_memory.get("Progress", "") if old_memory else ""
        new_memory = sample["new_memory"]
        new_memory_str = new_memory.get("Progress", "") if new_memory else ""

        task_desc = self.task_descriptions[task_id]
        user_text = build_vlm_user_text(
            task_description=task_desc,
            memory=old_memory_str,
            enable_memory=self.enable_memory,
            skill_aug=self.skill_library,
            subtask_aug=self.task_subtasks.get(task_name),
            aug_option=self.aug_option,
        )
        answer_text = build_vlm_answer(
            reasoning=sample["reasoning"],
            memory=new_memory_str,
            subtask=subtask,
            enable_reasoning=self.enable_reasoning,
            enable_memory=self.enable_memory,
        )

        return _encode_qwen_sft_item(
            pil_img, user_text, answer_text, subtask,
            self.processor, self.eval_mode,
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_behavior_photo_data_loader_qwen(
    data_root: str,
    *,
    tasks: list[str],
    tolerance_s: float = 1e-4,
    processor,
    tokenizer,
    eval_mode: bool = False,
    eval_ratio: float = 0.1,
    eval_split: str | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    enable_reasoning: bool = False,
    enable_memory: bool = False,
    simple_skill: bool = True,
    sft_data_dir: str | None = None,
    task_subtasks: dict[str, list[str]] | None = None,
    skill_library: list[str] | None = None,
    aug_option: str = "none",
) -> DataLoader:
    """Create a DataLoader for Qwen VLM SFT on BEHAVIOR photo data."""
    from rlinf.data.datasets import sft_collate_fn

    photo_root = os.path.join(data_root, "photos")
    needs_sft_json = enable_reasoning or enable_memory or not simple_skill

    if needs_sft_json:
        assert sft_data_dir is not None, (
            "sft_data_dir is required when enable_reasoning, enable_memory, "
            "or simple_skill=False"
        )

        target_task_ids = {TASK_NAMES_TO_INDICES[t] for t in tasks}
        train_positions, eval_positions = _get_agentic_episode_split(
            sft_data_dir, data_root, task_ids=target_task_ids, eval_ratio=eval_ratio,
        )

        if eval_split == "all":
            ep_positions = sorted(set(train_positions + eval_positions))
        elif eval_split == "train":
            ep_positions = train_positions
        elif eval_split is not None:
            ep_positions = eval_positions
        else:
            ep_positions = eval_positions if eval_mode else train_positions

        logger.info(
            "Agentic episode split: %d train, %d eval — using %s (%d episodes)",
            len(train_positions), len(eval_positions),
            eval_split or ("eval" if eval_mode else "train"), len(ep_positions),
        )

        dataset = BehaviorPhotoAgenticDataset(
            photo_root=photo_root,
            data_root=data_root,
            sft_data_dir=sft_data_dir,
            task_names=tasks,
            episode_indices=ep_positions,
            processor=processor,
            tokenizer=tokenizer,
            eval_mode=eval_mode,
            enable_reasoning=enable_reasoning,
            enable_memory=enable_memory,
            simple_skill=simple_skill,
            task_subtasks=task_subtasks,
            skill_library=skill_library,
            aug_option=aug_option,
        )
    else:
        train_indices, eval_indices = _get_train_eval_episode_indices(
            data_root, tasks, eval_ratio=eval_ratio,
        )

        if eval_split == "all":
            ep_indices = sorted(set(train_indices + eval_indices))
        elif eval_split == "train":
            ep_indices = train_indices
        elif eval_split is not None:
            ep_indices = eval_indices
        else:
            ep_indices = eval_indices if eval_mode else train_indices

        logger.info(
            "Episode split (Qwen photo): %d train, %d eval — using %s (%d episodes)",
            len(train_indices), len(eval_indices),
            eval_split or ("eval" if eval_mode else "train"), len(ep_indices),
        )

        dataset = BehaviorPhotoSkillDataset(
            photo_root=photo_root,
            data_root=data_root,
            task_names=tasks,
            episode_indices=ep_indices,
            processor=processor,
            tokenizer=tokenizer,
            eval_mode=eval_mode,
            enable_reasoning=enable_reasoning,
            enable_memory=enable_memory,
            simple_skill=simple_skill,
            task_subtasks=task_subtasks,
            skill_library=skill_library,
            aug_option=aug_option,
        )

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=not eval_mode,
            drop_last=not eval_mode,
        )

    effective_workers = 0 if eval_mode else num_workers
    mp_context = multiprocessing.get_context("spawn") if effective_workers > 0 else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and not eval_mode),
        num_workers=effective_workers,
        multiprocessing_context=mp_context,
        persistent_workers=effective_workers > 0,
        collate_fn=sft_collate_fn,
        drop_last=not eval_mode,
    )

    logger.info(
        "Built behavior VLM Qwen photo %s%s dataloader: %d samples, batch=%d, workers=%d",
        "agentic " if needs_sft_json else "",
        "eval" if eval_mode else "train",
        len(dataset),
        batch_size,
        effective_workers,
    )
    return loader
