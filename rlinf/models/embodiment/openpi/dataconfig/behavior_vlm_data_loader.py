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

"""VLM-specific data loader for BEHAVIOR skill prediction.

Reuses ``BehaviorLeRobotDataset`` (the same backbone as the VLA pipeline) but
applies VLM-specific transforms instead of VLA transforms.  No norm_stats are
loaded — this is purely VLM (CE loss on text tokens).

Entry points:
- ``create_behavior_vlm_data_loader`` — pi0.5 VLM-only path
- ``create_behavior_vlm_data_loader_qwen`` — Qwen2.5-VL path (skill or agentic)
"""

from __future__ import annotations

import json
import logging
import multiprocessing
import os

import einops
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rlinf.agents.dualsystem.prompts import (
    DEFAULT_VLM_PROMPT,
    MEMORY_VLM_PROMPT,
    format_agentic_answer,
)
from rlinf.models.embodiment.openpi.dataconfig.behavior_dataset import (
    BehaviorLeRobotDataset,
    PromptFromLeRobotItem,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pi0.5 VLM transform (unchanged — different tokenizer / output format)
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
# Unified Qwen VLM transform (handles both skill-only and agentic modes)
# ---------------------------------------------------------------------------


class BehaviorQwenTransform:
    """Convert BehaviorLeRobotDataset item → Qwen ``SftDatasetItem``.

    When ``enable_memory=False`` (default): uses ``DEFAULT_VLM_PROMPT`` and
    predicts a skill label.

    When ``enable_memory=True``: uses ``MEMORY_VLM_PROMPT`` and predicts
    reasoning + new memory + subtask.

    When ``enable_memory=True`` and ``use_simple_skill=True``: uses
    ``MEMORY_VLM_PROMPT`` for input (with old memory) but the answer is
    the simple skill label (one of the 4 skills), not the full agentic output.

    Prompt templates are imported from ``rlinf.agents.dualsystem.prompts``
    (the single source of truth shared with dual-system eval).
    """

    def __init__(
        self,
        processor,
        tokenizer,
        eval_mode: bool = False,
        system_prompt: str | None = None,
        enable_memory: bool = False,
        use_simple_skill: bool = False,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.eval_mode = eval_mode
        self.system_prompt = system_prompt
        self.enable_memory = enable_memory
        self.use_simple_skill = use_simple_skill

    def _extract_image(self, data: dict) -> Image.Image:
        """Extract image tensor from data and convert to PIL RGB."""
        img_tensor = data.get(
            "observation.images.rgb.head",
            data.get("observation/egocentric_camera"),
        )
        img = np.asarray(img_tensor)
        if np.issubdtype(img.dtype, np.floating):
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
            img = (img * 255).clip(0, 255).astype(np.uint8)
        else:
            if img.shape[0] == 3:
                img = einops.rearrange(img, "c h w -> h w c")
        return Image.fromarray(img).convert("RGB")

    def __call__(self, data: dict):
        from rlinf.data.datasets.item import SftDatasetItem

        pil_img = self._extract_image(data)
        prompt_text = data.get("prompt", "")

        # --- Build user text and answer text ---
        if self.enable_memory:
            old_memory = data.get("old_memory", {})
            progress = old_memory.get("Progress", "") if old_memory else ""
            world_state = old_memory.get("World state", "") if old_memory else ""
            user_text = MEMORY_VLM_PROMPT.format(
                task_description=prompt_text,
                progress=progress,
                world_state=world_state,
            )
            new_memory = data.get("new_memory", {})
            # use_simple_skill: subtask comes from the 4 skill labels;
            # otherwise from the agentic dataset's detailed subtask.
            subtask = (
                data.get("skill_label", "")
                if self.use_simple_skill
                else data.get("subtask", "")
            )
            answer_text = format_agentic_answer(
                reasoning=data.get("reasoning", ""),
                new_progress=new_memory.get("Progress", ""),
                new_world_state=new_memory.get("World state", ""),
                subtask=subtask,
            )
        else:
            user_text = DEFAULT_VLM_PROMPT.format(task_description=prompt_text)
            answer_text = data.get("skill_label", "")

        # --- Build chat messages ---
        image_context = [{"type": "image", "image": pil_img}, {"type": "text", "text": user_text}]
        prompt_messages = []
        if self.system_prompt:
            prompt_messages.append({"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
        prompt_messages.append({"role": "user", "content": image_context})

        with_answer = [(*prompt_messages, {"role": "assistant", "content": [{"type": "text", "text": answer_text}]})]
        without_answer = [tuple(prompt_messages)]

        prompt_inputs = self._encode(with_answer, [pil_img])
        label_inputs = self._encode(without_answer, [pil_img])

        if self.eval_mode:
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
            answer=answer_text,
            prompt_text=user_text,
            attention_mask=attention_mask,
            label_mask=label_mask,
            meta=None,
            multi_modal_inputs={k: v for k, v in multi_modal.items()},
        )

    def _encode(self, messages, images):
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.processor(text=rendered, images=images, return_tensors="pt", padding=True)


# ---------------------------------------------------------------------------
# Wrapper datasets
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


class _AgenticVLMDataset(Dataset):
    """Wraps BehaviorLeRobotDataset, only exposes frames present in the
    agentic SFT JSON files, and attaches old_memory/reasoning/new_memory/subtask."""

    def __init__(self, base_dataset: BehaviorLeRobotDataset, sft_data_dir: str, transforms: list):
        self._base = base_dataset
        self._transforms = transforms

        # Build ep_idx → hf_dataset start offset mapping
        self._ep_offset = {}
        for i, ep_idx in enumerate(base_dataset.episodes):
            self._ep_offset[ep_idx] = base_dataset.episode_data_index["from"][i].item()

        # Load SFT JSON files, collect result_used=True data points
        self.samples = []
        for fname in sorted(os.listdir(sft_data_dir)):
            if not fname.endswith(".json"):
                continue
            ep_idx = int(fname.replace("episode_", "").replace(".json", ""))
            if ep_idx not in self._ep_offset:
                continue  # not in current split
            with open(os.path.join(sft_data_dir, fname)) as f:
                data = json.load(f)
            for r in data["results"]:
                if not r["result_used"]:
                    continue
                old_mem = r["request_input"]["old_memory"]
                if old_mem and isinstance(old_mem, str):
                    old_mem = json.loads(old_mem)
                else:
                    old_mem = {}
                self.samples.append({
                    "ep_idx": ep_idx,
                    "frame_number": r["frame_number"],
                    "old_memory": old_mem,
                    "reasoning": r["model_response"]["reasoning"],
                    "new_memory": r["model_response"]["new_memory"],
                    "subtask": r["model_response"]["subtask"],
                })

        logger.info(
            "AgenticVLMDataset: loaded %d samples from %d episodes in %s",
            len(self.samples),
            len({s["ep_idx"] for s in self.samples}),
            sft_data_dir,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        hf_idx = self._ep_offset[sample["ep_idx"]] + sample["frame_number"]

        item = self._base[hf_idx]

        # Attach agentic fields
        item["old_memory"] = sample["old_memory"]
        item["reasoning"] = sample["reasoning"]
        item["new_memory"] = sample["new_memory"]
        item["subtask"] = sample["subtask"]

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
# Data loader factories
# ---------------------------------------------------------------------------


def _get_train_eval_episode_indices(
    data_root: str,
    tasks: list[str] | None,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split episodes into train/eval by index (episode-level, iid).

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

    rng = np.random.default_rng(seed)
    all_indices = list(range(ep_count))
    rng.shuffle(all_indices)
    n_eval = max(1, int(ep_count * eval_ratio))
    eval_indices = sorted(all_indices[:n_eval])
    train_indices = sorted(all_indices[n_eval:])
    return train_indices, eval_indices


def _get_agentic_episode_split(
    sft_data_dir: str,
    video_data_root: str,
    eval_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Split SFT agentic episodes into train/eval.

    Returns two lists of 0-based per-task positions for use with
    BehaviorLeRobotDataset(episodes=...).
    """
    from pathlib import Path

    sft_eps = sorted(
        int(f.replace("episode_", "").replace(".json", ""))
        for f in os.listdir(sft_data_dir)
        if f.endswith(".json")
    )

    episodes_path = Path(video_data_root) / "meta" / "episodes.jsonl"
    all_task_eps = []
    with episodes_path.open() as f:
        for line in f:
            item = json.loads(line)
            if int(item["episode_index"] // 1e4) == 0:  # task-0000
                all_task_eps.append(item["episode_index"])
    all_task_eps.sort()
    ep_to_pos = {ep: i for i, ep in enumerate(all_task_eps)}

    sft_positions = [ep_to_pos[ep] for ep in sft_eps if ep in ep_to_pos]

    rng = np.random.default_rng(seed)
    shuffled = list(sft_positions)
    rng.shuffle(shuffled)
    n_eval = max(1, int(len(shuffled) * eval_ratio))
    eval_positions = sorted(shuffled[:n_eval])
    train_positions = sorted(shuffled[n_eval:])
    return train_positions, eval_positions


def _create_base_dataset(
    data_root: str,
    tasks: list[str] | None,
    tolerance_s: float,
    seed: int,
    shuffle: bool,
    episode_indices: list[int] | None = None,
):
    """Create a BehaviorLeRobotDataset configured for VLM (no delta_timestamps).

    Uses ``chunk_streaming_using_keyframe=False`` to avoid the streaming
    state that doesn't survive Ray actor spawning / DataLoader workers.
    """
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
        data_root, tasks, eval_ratio=eval_ratio, seed=seed,
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


def create_behavior_vlm_data_loader_qwen(
    data_root: str,
    *,
    tasks: list[str] | None = None,
    tolerance_s: float = 1e-4,
    processor,
    tokenizer,
    eval_mode: bool = False,
    eval_ratio: float = 0.1,
    batch_size: int = 4,
    num_workers: int = 4,
    seed: int = 42,
    system_prompt: str | None = None,
    enable_memory: bool = False,
    sft_data_dir: str | None = None,
    use_simple_skill: bool = False,
):
    """Create a PyTorch DataLoader for Qwen2.5-VL SFT on BEHAVIOR data.

    Args:
        enable_memory: When False (default), trains on skill labels using
            ``DEFAULT_VLM_PROMPT``.  When True, trains on agentic data
            (memory + reasoning + subtask) using ``MEMORY_VLM_PROMPT``.
        sft_data_dir: Required when ``enable_memory=True``.  Path to the
            pre-annotated SFT JSON files.
        use_simple_skill: When True (and enable_memory=True), uses the
            memory prompt as input but predicts simple skill labels instead
            of the full agentic output.
    """
    from rlinf.data.datasets import sft_collate_fn

    if enable_memory:
        assert sft_data_dir is not None, "sft_data_dir is required when enable_memory=True"

        train_positions, eval_positions = _get_agentic_episode_split(
            sft_data_dir, data_root, eval_ratio=eval_ratio, seed=seed,
        )
        ep_positions = eval_positions if eval_mode else train_positions
        logger.info(
            "Agentic episode split: %d train, %d eval — using %s (%d episodes)",
            len(train_positions), len(eval_positions),
            "eval" if eval_mode else "train", len(ep_positions),
        )

        # When use_simple_skill=True, we need skill_labels on the base
        # dataset so that each frame gets a skill_label field.
        skill_labels_for_base = None
        if use_simple_skill:
            skill_labels_for_base = {
                0: "move to radio",
                1: "pick up radio from coffee table",
                2: "press radio",
                3: "place radio on coffee table",
            }

        base_dataset = BehaviorLeRobotDataset(
            repo_id="behavior-1k/2025-challenge-demos",
            root=data_root,
            tolerance_s=tolerance_s,
            tasks=["turning_on_radio"],
            episodes=ep_positions,
            modalities=["rgb"],
            local_only=True,
            delta_timestamps=None,
            chunk_streaming_using_keyframe=False,
            shuffle=not eval_mode,
            seed=seed,
            fine_grained_level=0,
            skill_labels=skill_labels_for_base,
        )

        # Always use eval_mode=False for agentic transform so input_ids
        # contain the full sequence (prompt + answer), enabling eval loss
        # computation with the same masking as training.
        qwen_transform = BehaviorQwenTransform(
            processor=processor,
            tokenizer=tokenizer,
            eval_mode=False,
            system_prompt=system_prompt,
            enable_memory=True,
            use_simple_skill=use_simple_skill,
        )

        dataset = _AgenticVLMDataset(
            base_dataset, sft_data_dir, [PromptFromLeRobotItem(), qwen_transform]
        )
    else:
        train_indices, eval_indices = _get_train_eval_episode_indices(
            data_root, tasks, eval_ratio=eval_ratio, seed=seed,
        )
        ep_indices = eval_indices if eval_mode else train_indices
        logger.info(
            "Episode split (Qwen): %d train, %d eval — using %s (%d episodes)",
            len(train_indices), len(eval_indices),
            "eval" if eval_mode else "train", len(ep_indices),
        )

        base_dataset = _create_base_dataset(
            data_root, tasks, tolerance_s, seed,
            shuffle=not eval_mode,
            episode_indices=ep_indices,
        )

        qwen_transform = BehaviorQwenTransform(
            processor=processor,
            tokenizer=tokenizer,
            eval_mode=eval_mode,
            system_prompt=system_prompt,
            enable_memory=False,
        )

        dataset = _TransformedVLMDataset(
            base_dataset, [PromptFromLeRobotItem(), qwen_transform]
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
        collate_fn=sft_collate_fn,
        drop_last=True,
    )

    logger.info(
        "Built behavior VLM Qwen %s%s dataloader: %d samples, batch=%d, workers=%d",
        "agentic " if enable_memory else "",
        "eval" if eval_mode else "train",
        len(dataset),
        batch_size,
        effective_workers,
    )
    return loader
