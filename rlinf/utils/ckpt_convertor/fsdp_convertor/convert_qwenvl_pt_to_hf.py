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

Usage:
    python -m rlinf.utils.ckpt_convertor.fsdp_convertor.convert_qwenvl_pt_to_hf \
        --ckpt_path /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260517-12:14:02/behavior_qwen2_5_vlm_sft_agentic/checkpoints/global_step_475/actor/model_state_dict/full_weights.pt \
        --model_path /mnt/public/xzxuan/models/Qwen2.5-VL-3B-Instruct \
        --save_path /mnt/public/xzxuan/repos/RLinf_pi05/logs/20260517-12:14:02/behavior_qwen2_5_vlm_sft_agentic/checkpoints/global_step_475/actor/model_state_dict/full_weights.pt
"""

import argparse

import torch
from transformers import AutoConfig, AutoModelForVision2Seq

from .utils import copy_model_config_and_code, save_state_dict_sharded_safetensors


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
    #    AutoModelForVision2Seq wraps sub-models, producing keys like:
    #      model.language_model.layers.0... -> model.layers.0...
    #      model.visual.blocks.0...         -> visual.blocks.0...
    #    We strip the wrapper prefix so the output matches the base model layout.
    raw_sd = model.state_dict()
    remapped_sd: dict[str, torch.Tensor] = {}
    seen_ptrs: dict[int, str] = {}
    for k, v in raw_sd.items():
        # Deduplicate tied weights (e.g. lm_head.weight == embed_tokens.weight)
        ptr = v.data_ptr()
        if ptr in seen_ptrs:
            continue
        seen_ptrs[ptr] = k

        # Remap key prefixes
        if k.startswith("model.language_model."):
            new_k = "model." + k[len("model.language_model."):]
        elif k.startswith("model.visual."):
            new_k = "visual." + k[len("model.visual."):]
        else:
            new_k = k
        remapped_sd[new_k] = v

    skipped = len(raw_sd) - len(remapped_sd)
    if skipped:
        print(f"  Skipped {skipped} tied weight(s)")

    print(f"Saving safetensors to {args.save_path} ...")
    num_shards, total_size = save_state_dict_sharded_safetensors(
        state_dict=remapped_sd, out_dir=args.save_path
    )

    print(
        f"Done! {num_shards} shard(s), {total_size / 1e9:.2f} GB -> {args.save_path}"
    )


if __name__ == "__main__":
    main()
