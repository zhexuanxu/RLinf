#!/usr/bin/env python3
"""Verify that the new photo-based dataloader produces identical outputs to the old video-based one.

Usage:
    python tools/verify_photo_dataloader.py \
        --data-root /mnt/public/xzxuan/data/2025-challenge-demos \
        --sft-data-dir /mnt/public/xzxuan/data/turn_on_radio_fix_path \
        --model-path /mnt/public/xzxuan/models/Qwen2.5-VL-3B-Instruct \
        --num-samples 10
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from transformers import AutoProcessor, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
logger = logging.getLogger("verify")


def compare_items(old_item, new_item, idx: int, mode: str) -> list[str]:
    """Compare two SftDatasetItem objects. Returns list of failure messages."""
    failures = []
    prefix = f"[{mode}] sample {idx}"

    # Compare answer string
    if old_item.answer != new_item.answer:
        failures.append(f"{prefix}: answer mismatch: {old_item.answer!r} vs {new_item.answer!r}")

    # Compare prompt_text string
    if old_item.prompt_text != new_item.prompt_text:
        failures.append(f"{prefix}: prompt_text mismatch:\n  OLD: {old_item.prompt_text!r}\n  NEW: {new_item.prompt_text!r}")

    # Compare prompt (input_ids)
    old_ids = old_item.prompt.flatten()
    new_ids = new_item.prompt.flatten()
    if not torch.equal(old_ids, new_ids):
        failures.append(f"{prefix}: prompt input_ids differ (shapes: {old_ids.shape} vs {new_ids.shape})")

    # Compare attention_mask
    old_am = old_item.attention_mask.flatten() if old_item.attention_mask is not None else None
    new_am = new_item.attention_mask.flatten() if new_item.attention_mask is not None else None
    if old_am is not None and new_am is not None:
        if not torch.equal(old_am, new_am):
            failures.append(f"{prefix}: attention_mask differs")

    # Compare label_mask
    old_lm = old_item.label_mask.flatten() if old_item.label_mask is not None else None
    new_lm = new_item.label_mask.flatten() if new_item.label_mask is not None else None
    if old_lm is not None and new_lm is not None:
        if not torch.equal(old_lm, new_lm):
            failures.append(f"{prefix}: label_mask differs")

    # Compare multi_modal_inputs pixel_values (within tolerance due to JPEG)
    if old_item.multi_modal_inputs and new_item.multi_modal_inputs:
        old_pv = old_item.multi_modal_inputs.get("pixel_values")
        new_pv = new_item.multi_modal_inputs.get("pixel_values")
        if old_pv is not None and new_pv is not None:
            if old_pv.shape != new_pv.shape:
                failures.append(f"{prefix}: pixel_values shape mismatch: {old_pv.shape} vs {new_pv.shape}")
            else:
                max_diff = (old_pv.float() - new_pv.float()).abs().max().item()
                if max_diff > 0.1:
                    failures.append(f"{prefix}: pixel_values max diff = {max_diff:.4f} (threshold: 0.1)")
                else:
                    logger.info(f"{prefix}: pixel_values max diff = {max_diff:.6f} (OK)")

    return failures


def test_skill_mode(data_root, model_path, num_samples):
    """Test behavior_skill_sft mode (no reasoning, no memory, simple skill)."""
    logger.info("=" * 60)
    logger.info("Testing SKILL mode (enable_reasoning=F, enable_memory=F, simple_skill=T)")
    logger.info("=" * 60)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    # Old dataset
    from rlinf.models.embodiment.openpi.dataconfig.behavior_vlm_data_loader import (
        BehaviorQwenTransform,
        PromptFromLeRobotItem,
        _TransformedVLMDataset,
        _create_base_dataset,
        _get_train_eval_episode_indices,
    )

    train_indices, eval_indices = _get_train_eval_episode_indices(
        data_root, ["turning_on_radio"], eval_ratio=0.1,
    )
    old_base = _create_base_dataset(
        data_root, ["turning_on_radio"], 1e-4, 42,
        shuffle=False, episode_indices=train_indices,
    )
    old_transform = BehaviorQwenTransform(
        processor=processor, tokenizer=tokenizer, eval_mode=False,
        enable_reasoning=False, enable_memory=False, simple_skill=True,
    )
    old_dataset = _TransformedVLMDataset(
        old_base, [PromptFromLeRobotItem(), old_transform]
    )

    # New dataset
    from rlinf.data.datasets.behavior_photo_vlm import (
        BehaviorPhotoSkillDataset,
        _get_train_eval_episode_indices as new_get_split,
    )

    new_train_indices, _ = new_get_split(data_root, ["turning_on_radio"], eval_ratio=0.1)
    new_dataset = BehaviorPhotoSkillDataset(
        photo_root=f"{data_root}/photos",
        data_root=data_root,
        task_names=["turning_on_radio"],
        episode_indices=new_train_indices,
        processor=processor, tokenizer=tokenizer,
        eval_mode=False,
        enable_reasoning=False, enable_memory=False, simple_skill=True,
    )

    logger.info(f"Old dataset size: {len(old_dataset)}, New dataset size: {len(new_dataset)}")
    if len(old_dataset) != len(new_dataset):
        logger.error(f"Dataset sizes differ! Old: {len(old_dataset)}, New: {len(new_dataset)}")

    failures = []
    for i in range(min(num_samples, len(old_dataset), len(new_dataset))):
        logger.info(f"Comparing sample {i}...")
        old_item = old_dataset[i]
        new_item = new_dataset[i]
        failures.extend(compare_items(old_item, new_item, i, "skill"))

    return failures


def test_agentic_mode(data_root, sft_data_dir, model_path, num_samples):
    """Test behavior_agentic_sft mode (reasoning + memory + simple_skill)."""
    logger.info("=" * 60)
    logger.info("Testing AGENTIC mode (enable_reasoning=T, enable_memory=T, simple_skill=T)")
    logger.info("=" * 60)

    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    # Old dataset
    from rlinf.models.embodiment.openpi.dataconfig.behavior_vlm_data_loader import (
        BehaviorQwenTransform,
        PromptFromLeRobotItem,
        _AgenticVLMDataset,
        _get_agentic_episode_split,
    )
    from rlinf.models.embodiment.openpi.dataconfig.behavior_dataset import (
        BehaviorLeRobotDataset,
    )

    old_train_positions, _ = _get_agentic_episode_split(
        sft_data_dir, data_root, eval_ratio=0.1,
    )
    skill_labels = {
        0: "move to radio",
        1: "pick up radio from coffee table",
        2: "press radio",
        3: "place radio on coffee table",
    }
    old_base = BehaviorLeRobotDataset(
        repo_id="behavior-1k/2025-challenge-demos",
        root=data_root, tolerance_s=1e-4,
        tasks=["turning_on_radio"],
        episodes=old_train_positions,
        modalities=["rgb"], local_only=True,
        delta_timestamps=None,
        chunk_streaming_using_keyframe=False,
        shuffle=False, seed=42,
        fine_grained_level=0,
        skill_labels=skill_labels,
    )
    old_transform = BehaviorQwenTransform(
        processor=processor, tokenizer=tokenizer, eval_mode=False,
        enable_reasoning=True, enable_memory=True, simple_skill=True,
    )
    old_dataset = _AgenticVLMDataset(
        old_base, sft_data_dir, [PromptFromLeRobotItem(), old_transform]
    )

    # New dataset
    from rlinf.data.datasets.behavior_photo_vlm import (
        BehaviorPhotoAgenticDataset,
        _get_agentic_episode_split as new_get_agentic_split,
    )

    new_train_positions, _ = new_get_agentic_split(
        sft_data_dir, data_root, eval_ratio=0.1,
    )
    new_dataset = BehaviorPhotoAgenticDataset(
        photo_root=f"{data_root}/photos",
        data_root=data_root,
        sft_data_dir=sft_data_dir,
        episode_indices=new_train_positions,
        processor=processor, tokenizer=tokenizer,
        eval_mode=False,
        enable_reasoning=True, enable_memory=True, simple_skill=True,
    )

    logger.info(f"Old dataset size: {len(old_dataset)}, New dataset size: {len(new_dataset)}")
    if len(old_dataset) != len(new_dataset):
        logger.error(f"Dataset sizes differ! Old: {len(old_dataset)}, New: {len(new_dataset)}")

    failures = []
    for i in range(min(num_samples, len(old_dataset), len(new_dataset))):
        logger.info(f"Comparing sample {i}...")
        old_item = old_dataset[i]
        new_item = new_dataset[i]
        failures.extend(compare_items(old_item, new_item, i, "agentic"))

    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sft-data-dir", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--mode", choices=["skill", "agentic", "all"], default="all")
    args = parser.parse_args()

    all_failures = []

    if args.mode in ("skill", "all"):
        failures = test_skill_mode(args.data_root, args.model_path, args.num_samples)
        all_failures.extend(failures)

    if args.mode in ("agentic", "all") and args.sft_data_dir:
        failures = test_agentic_mode(
            args.data_root, args.sft_data_dir, args.model_path, args.num_samples,
        )
        all_failures.extend(failures)

    print("\n" + "=" * 60)
    if all_failures:
        print(f"FAILED: {len(all_failures)} mismatches found:")
        for f in all_failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED!")
        sys.exit(0)


if __name__ == "__main__":
    main()
