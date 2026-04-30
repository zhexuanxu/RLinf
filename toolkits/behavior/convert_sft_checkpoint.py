#!/usr/bin/env python3
"""Convert FSDP SFT checkpoint to base-model-compatible safetensors format.

Handles two issues with FSDP full_state_dict:
1. FSDP unties embed_tokens/lm_head shared weights, saving an extra copy.
   We drop the extra key to match base model format.
2. FSDP may promote LayerNorm/VisionEmbedding params to float32 during training.
   We cast all params to bfloat16 to match base model dtype.

Usage:
    python toolkits/behavior/convert_sft_checkpoint.py \
        --ckpt /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260429-08:26:43/sft_behavior_pi05_vla_comet/checkpoints/global_step_3000/actor/model_state_dict/full_weights.pt \
        --base-model /mnt/public/xzxuan/models/pi05_base_pytorch \
        --output /mnt/public/xzxuan/models/pi05_behavior_sft_step3000
"""

import argparse
import json
import os
import shutil

import safetensors.torch
import torch


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP SFT checkpoint to safetensors")
    parser.add_argument("--ckpt", required=True, help="Path to full_weights.pt")
    parser.add_argument("--base-model", required=True, help="Base model directory (for config.json and dtype reference)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--norm-stats", default=None, help="Path to norm_stats.json (if different from base model)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Load base model to get the reference key set
    print(f"Loading base model keys from {args.base_model}/model.safetensors ...")
    base_keys = set(safetensors.torch.load_file(
        os.path.join(args.base_model, "model.safetensors"), device="cpu"
    ).keys())
    print(f"  Base model: {len(base_keys)} keys")

    # Load SFT checkpoint
    print(f"Loading checkpoint from {args.ckpt} ...")
    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"  Checkpoint: {len(state_dict)} keys")

    # Drop keys not in base model (untied embed_tokens, etc.)
    extra_keys = set(state_dict.keys()) - base_keys
    for k in extra_keys:
        del state_dict[k]
        print(f"  Dropped extra key: {k}")

    # Warn about missing keys
    missing_keys = base_keys - set(state_dict.keys())
    if missing_keys:
        print(f"  WARNING: {len(missing_keys)} keys in base model but not in checkpoint:")
        for k in sorted(missing_keys)[:5]:
            print(f"    {k}")

    # Cast all to bfloat16 (matching base model)
    state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
    print(f"  All {len(state_dict)} tensors cast to bfloat16")

    # Save safetensors
    out_safetensors = os.path.join(args.output, "model.safetensors")
    safetensors.torch.save_file(state_dict, out_safetensors)
    size_gb = os.path.getsize(out_safetensors) / 1e9
    print(f"  Saved {out_safetensors} ({size_gb:.2f} GB)")

    # Copy config.json
    shutil.copy2(os.path.join(args.base_model, "config.json"), os.path.join(args.output, "config.json"))
    print(f"  Copied config.json")

    # Copy norm_stats
    norm_src = args.norm_stats or os.path.join(args.base_model, "physical-intelligence/behavior/norm_stats.json")
    norm_dst_dir = os.path.join(args.output, "physical-intelligence/behavior")
    os.makedirs(norm_dst_dir, exist_ok=True)
    shutil.copy2(norm_src, os.path.join(norm_dst_dir, "norm_stats.json"))
    print(f"  Copied norm_stats.json from {norm_src}")

    print(f"\nDone: {args.output}")


if __name__ == "__main__":
    main()
