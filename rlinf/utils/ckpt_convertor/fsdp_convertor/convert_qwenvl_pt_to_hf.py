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

"""Convert Qwen-VL FSDP checkpoint to HuggingFace safetensors format.

The generic ``convert_pt_to_hf`` converter relies on ``get_model()`` which only
covers embodied model types registered in ``_MODEL_REGISTRY``.  Qwen-VL models
(qwen2.5_vl, qwen3_vl, etc.) are loaded via ``AutoModelForVision2Seq`` in the
FSDP worker fallback path, so they need a dedicated converter.

Supports both Qwen2.5-VL and Qwen3-VL by auto-detecting the target model's
on-disk key format from its safetensors index.

Usage:
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_qwenvl_pt_to_hf \
        --ckpt_path /mnt/public/xzxuan/repos/dualsys/logs/20260522-13:08:24/behavior_qwen3_vlm_sft_agentic/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt \
        --model_path /mnt/public/xzxuan/models/Qwen3-VL-4B-Thinking \
        --save_path /mnt/public/xzxuan/repos/dualsys/logs/20260522-13:08:24/hf
"""

import argparse
import json
import os

import torch
from transformers import AutoConfig, AutoModelForVision2Seq

from .utils import copy_model_config_and_code, save_state_dict_sharded_safetensors


def _load_reference_keys(model_path: str) -> set[str]:
    """Load the set of weight keys from the reference model's safetensors index."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return set(json.load(f)["weight_map"].keys())
    single = os.path.join(model_path, "model.safetensors")
    if os.path.exists(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt") as f:
            return set(f.keys())
    return set()


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen-VL FSDP checkpoint to HuggingFace safetensors"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to full_weights.pt"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base HuggingFace model (for architecture and configs)",
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "bf16", "fp16"],
        help="Torch dtype for loading the model (default: bf16)",
    )
    args = parser.parse_args()

    dtype_map = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}
    torch_dtype = dtype_map[args.dtype]

    # 1. Load model architecture
    print(f"Loading model architecture from {args.model_path} ...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    # 2. Load trained weights
    print(f"Loading checkpoint from {args.ckpt_path} ...")
    state_dict = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)

    # 3. Copy configs from base model (keeps json/py/md identical)
    print(f"Copying configs from {args.model_path} ...")
    copy_model_config_and_code(model_path=args.model_path, save_path=args.save_path)

    # 4. Remap keys from AutoModelForVision2Seq internal format to HF on-disk format.
    #
    #    AutoModelForVision2Seq wraps all models with:
    #      model.language_model.*  and  model.visual.*
    #
    #    But different models have different on-disk formats:
    #      Qwen2.5-VL: model.layers.*, visual.*        (strips language_model + model.visual)
    #      Qwen3-VL:   model.language_model.*, model.visual.*  (keeps as-is)
    #
    #    We auto-detect by inspecting the reference model's safetensors index.
    ref_keys = _load_reference_keys(args.model_path)
    needs_strip_language_model = ref_keys and not any(
        k.startswith("model.language_model.") for k in ref_keys
    )
    needs_strip_visual_prefix = ref_keys and any(
        k.startswith("visual.") for k in ref_keys
    )

    if ref_keys:
        print(
            f"  Key format: strip_language_model={needs_strip_language_model}, "
            f"strip_visual_prefix={needs_strip_visual_prefix}"
        )

    raw_sd = model.state_dict()
    remapped_sd: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[int, str] = {}
    for k, v in raw_sd.items():
        # Deduplicate tied weights (e.g. lm_head.weight == embed_tokens.weight)
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs[ptr] = k

        # Remap key prefixes based on auto-detected format
        new_k = k
        if needs_strip_language_model and k.startswith("model.language_model."):
            new_k = "model." + k[len("model.language_model."):]
        if needs_strip_visual_prefix and k.startswith("model.visual."):
            new_k = "visual." + k[len("model.visual."):]
        remapped_sd[new_k] = v

    skipped = len(raw_sd) - len(remapped_sd)
    if skipped:
        print(f"  Skipped {skipped} tied weight(s)")

    # Validate remapped keys against reference
    if ref_keys:
        unexpected = set(remapped_sd.keys()) - ref_keys
        if unexpected:
            print(f"  WARNING: {len(unexpected)} keys not in reference model: {sorted(unexpected)[:5]}...")
        missing = ref_keys - set(remapped_sd.keys())
        if missing:
            print(f"  WARNING: {len(missing)} reference keys missing from output: {sorted(missing)[:5]}...")

    print(f"Saving safetensors to {args.save_path} ...")
    num_shards, total_size = save_state_dict_sharded_safetensors(
        state_dict=remapped_sd, out_dir=args.save_path
    )

    print(
        f"Done! {num_shards} shard(s), {total_size / 1e9:.2f} GB -> {args.save_path}"
    )


if __name__ == "__main__":
    main()
