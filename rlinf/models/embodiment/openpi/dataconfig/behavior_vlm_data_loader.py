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

"""VLM-specific data loader for BEHAVIOR skill prediction (pi0.5 path).

Reuses ``BehaviorLeRobotDataset`` (the same backbone as the VLA pipeline) but
applies VLM-specific transforms instead of VLA transforms.  No norm_stats are
loaded — this is purely VLM (CE loss on text tokens).

Entry point:
- ``create_behavior_vlm_data_loader`` — pi0.5 VLM-only path

For Qwen VLM SFT, see ``rlinf.data.datasets.behavior_photo_vlm``.
"""

from __future__ import annotations

import json
import logging
import multiprocessing

import einops
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rlinf.models.embodiment.openpi.dataconfig.behavior_dataset import (
    BehaviorLeRobotDataset,
    PromptFromLeRobotItem,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pi0.5 VLM transform
# ---------------------------------------------------------------------------


class BehaviorSkillVLMTransform:
    """Convert BehaviorLeRobotDataset item → pi0.5 VLM observation dict.

    Output is a 3-tuple ``(observation_dict, None, meta_dict)`` matching
    ``Pi05VLMDataset`` format and compatible with ``pi05_vlm_collate_fn``.
    """

    def __init__(
        self,
        tokenizer,
        max_token_len: int = 200,
        image_size: int = 224,
        num_images: int = 1,
        eval_mode: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.image_size = image_size
        self.num_images = num_images
        self.eval_mode = eval_mode
        self._eos_token_id = 1  # PaliGemma default

    def __call__(self, data: dict) -> tuple[dict, None, dict]:
        # --- Image: [3, H, W] float [0,1] → [H, W, 3] float [-1, 1] ---
        img_tensor = data.get(
            "observation.images.rgb.head",
            data.get("observation/egocentric_camera"),
        )
        img = np.asarray(img_tensor)
        if np.issubdtype(img.dtype, np.floating):
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            img_uint8 = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            img_uint8 = img
        pil_img = Image.fromarray(img_uint8).resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )
        img_array = np.array(pil_img, dtype=np.float32) / 255.0 * 2.0 - 1.0

        # --- Text ---
        prompt_text = f"Task: {data.get('prompt', '')}\nSkill: "
        skill_label = data.get("skill_label", "")

        max_len = self.max_token_len
        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)

        if self.eval_mode:
            if len(prompt_tokens) > max_len:
                prompt_tokens = prompt_tokens[:max_len]
            prompt_len = len(prompt_tokens)

            tokens = np.zeros(max_len, dtype=np.int32)
            tokens[:prompt_len] = prompt_tokens
            token_mask = np.zeros(max_len, dtype=bool)
            token_mask[:prompt_len] = True
            ar_mask = np.zeros(max_len, dtype=np.int32)
            loss_mask = np.zeros(max_len, dtype=bool)
            kv_cache_mask = token_mask.copy()

            obs = self._build_observation(
                img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask
            )
            meta = {
                "skill_label": skill_label,
                "prompt_text": prompt_text,
            }
            return obs, None, meta

        # Train mode
        answer_tokens = self.tokenizer.encode(skill_label, add_bos=False)
        prompt_len = len(prompt_tokens)
        answer_len = len(answer_tokens)

        if prompt_len > max_len - 2:
            prompt_tokens = prompt_tokens[: max_len - 2]
            prompt_len = len(prompt_tokens)

        available = max_len - prompt_len - 1
        if available < answer_len:
            answer_tokens = answer_tokens[:available]
            answer_len = len(answer_tokens)

        total_len = prompt_len + answer_len + 1

        tokens = np.zeros(max_len, dtype=np.int32)
        tokens[:prompt_len] = prompt_tokens
        tokens[prompt_len: prompt_len + answer_len] = answer_tokens
        tokens[prompt_len + answer_len] = self._eos_token_id

        token_mask = np.zeros(max_len, dtype=bool)
        token_mask[:total_len] = True

        ar_mask = np.zeros(max_len, dtype=np.int32)
        ar_mask[prompt_len:total_len] = 1

        loss_mask = np.zeros(max_len, dtype=bool)
        loss_mask[prompt_len:total_len] = True

        kv_cache_mask = token_mask.copy()
        kv_cache_mask[prompt_len + answer_len] = False

        obs = self._build_observation(
            img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask
        )
        return obs, None, {}

    def _build_observation(self, img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask):
        image_dict = {"image_0": img_array}
        image_mask_dict = {"image_0": True}
        for i in range(1, self.num_images):
            image_dict[f"image_{i}"] = np.zeros_like(img_array)
            image_mask_dict[f"image_{i}"] = False
        return {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": np.zeros(1, dtype=np.float32),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "token_kv_cache_mask": kv_cache_mask,
        }


# ---------------------------------------------------------------------------
# Wrapper dataset (used by pi0.5 path)
# ---------------------------------------------------------------------------


class _TransformedVLMDataset(Dataset):
    """Wraps BehaviorLeRobotDataset + a chain of transforms."""

    def __init__(self, base_dataset: BehaviorLeRobotDataset, transforms: list):
        self._base = base_dataset
        self._transforms = transforms

    def __len__(self):
        return len(self._base)

    def __getitem__(self, idx):
        item = self._base[idx]
        for t in self._transforms:
            item = t(item)
        return item


# ---------------------------------------------------------------------------
# Collate helpers
# ---------------------------------------------------------------------------


def _pi05_vlm_collate_fn(batch):
    """Collate for pi0.5 VLM 3-tuples: (obs_dict, None, meta_dict)."""
    observations, _, metas = zip(*batch)
    batched = {"image": {}, "image_mask": {}}
    first = observations[0]
    for key in first["image"]:
        batched["image"][key] = np.stack([obs["image"][key] for obs in observations])
        batched["image_mask"][key] = np.array([obs["image_mask"][key] for obs in observations])
    for key in ("state", "tokenized_prompt", "tokenized_prompt_mask", "token_ar_mask", "token_loss_mask", "token_kv_cache_mask"):
        batched[key] = np.stack([obs[key] for obs in observations])
    return batched, None, list(metas)


# ---------------------------------------------------------------------------
# Data loader factory (pi0.5 only)
# ---------------------------------------------------------------------------


def _get_train_eval_episode_indices(
    data_root: str,
    tasks: list[str] | None,
    eval_ratio: float = 0.1,
) -> tuple[list[int], list[int]]:
    """Split episodes into train/eval by index (episode-level, deterministic).

    Returns 0-based index lists into the per-task episode list.
    """
    from pathlib import Path

    episodes_path = Path(data_root) / "meta" / "episodes.jsonl"
    task_name_to_index = {}
    tasks_path = Path(data_root) / "meta" / "tasks.jsonl"
    with tasks_path.open() as f:
        for line in f:
            item = json.loads(line)
            task_name_to_index[item["task_name"]] = item["task_index"]

    target_task_ids = set()
    if tasks:
        for t in tasks:
            if t in task_name_to_index:
                target_task_ids.add(task_name_to_index[t])
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
    train_indices = all_indices[: ep_count - n_eval]
    eval_indices = all_indices[ep_count - n_eval :]
    return train_indices, eval_indices


def _create_base_dataset(
    data_root: str,
    tasks: list[str] | None,
    tolerance_s: float,
    seed: int,
    shuffle: bool,
    episode_indices: list[int] | None = None,
):
    """Create a BehaviorLeRobotDataset configured for VLM (no delta_timestamps)."""
    skill_labels = {
        0: "move to radio",
        1: "pick up radio from coffee table",
        2: "press radio",
        3: "place radio on coffee table",
    }

    return BehaviorLeRobotDataset(
        repo_id="behavior-1k/2025-challenge-demos",
        root=data_root,
        tolerance_s=tolerance_s,
        tasks=tasks or None,
        episodes=episode_indices,
        modalities=["rgb"],
        local_only=True,
        delta_timestamps=None,
        chunk_streaming_using_keyframe=False,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=0,
        skill_labels=skill_labels,
    )


def create_behavior_vlm_data_loader(
    data_root: str,
    *,
    tasks: list[str] | None = None,
    tolerance_s: float = 1e-4,
    max_token_len: int = 200,
    num_images: int = 1,
    eval_mode: bool = False,
    eval_ratio: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
):
    """Create a PyTorch DataLoader for pi0.5 VLM-only SFT on BEHAVIOR data."""
    import openpi.shared.download as download
    import sentencepiece

    tok_path = download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    )
    with tok_path.open("rb") as f:
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    train_indices, eval_indices = _get_train_eval_episode_indices(
        data_root, tasks, eval_ratio=eval_ratio,
    )
    ep_indices = eval_indices if eval_mode else train_indices
    logger.info(
        "Episode split: %d train, %d eval — using %s (%d episodes)",
        len(train_indices), len(eval_indices),
        "eval" if eval_mode else "train", len(ep_indices),
    )

    base_dataset = _create_base_dataset(
        data_root, tasks, tolerance_s, seed,
        shuffle=not eval_mode,
        episode_indices=ep_indices,
    )

    vlm_transform = BehaviorSkillVLMTransform(
        tokenizer=tokenizer,
        max_token_len=max_token_len,
        image_size=224,
        num_images=num_images,
        eval_mode=eval_mode,
    )

    dataset = _TransformedVLMDataset(
        base_dataset, [PromptFromLeRobotItem(), vlm_transform]
    )

    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=not eval_mode,
            drop_last=True,
        )

    effective_workers = 0 if eval_mode else num_workers
    mp_context = multiprocessing.get_context("spawn") if effective_workers > 0 else None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None and not eval_mode),
        num_workers=effective_workers,
        multiprocessing_context=mp_context,
        persistent_workers=effective_workers > 0,
        collate_fn=_pi05_vlm_collate_fn,
        drop_last=True,
    )

    logger.info(
        "Built behavior VLM pi0.5 %s dataloader: %d samples, batch=%d, workers=%d",
        "eval" if eval_mode else "train",
        len(base_dataset),
        batch_size,
        effective_workers,
    )
    return loader, tokenizer
