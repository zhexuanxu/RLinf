#!/usr/bin/env python3
"""Inspect model parameter dtypes before and after FSDP wrapping.

This script resolves the key question: what dtypes do the parameters
actually have during training? Does FSDP flatten mixed-dtype params
into a single FlatParameter, and if so, what dtype does it use?

Usage:
    # Single GPU (no distributed):
    python toolkits/behavior/inspect_fsdp_dtypes.py

    # With torch distributed (simulates training):
    torchrun --nproc_per_node=1 toolkits/behavior/inspect_fsdp_dtypes.py --fsdp
"""
import argparse
import os
import sys
from collections import Counter, defaultdict

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def inspect_before_fsdp():
    """Load model exactly as training does and inspect dtypes."""
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )
    from rlinf.models.embodiment.openpi.openpi_full_pi05_model import (
        OpenPi05FullForRLActionPrediction,
    )
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    import openpi.models.pi0_config as pi0_config
    import safetensors.torch

    ckpt_path = "/mnt/public/xzxuan/models/pi05_base_pytorch"

    # Step 1: Create model (same as get_model in __init__.py)
    cfg_train = get_openpi_config("pi05_behavior_b1k_local", model_path=ckpt_path)
    cfg_model = OpenPi0Config(**cfg_train.model.__dict__)
    cfg_model.__dict__["full_pi05"] = True
    cfg_model.__dict__["forward_mode"] = "vla"

    model = OpenPi05FullForRLActionPrediction(cfg_model)

    # Step 2: Load weights
    state_dict = safetensors.torch.load_file(
        os.path.join(ckpt_path, "model.safetensors"), device="cpu"
    )
    result = model.load_state_dict(state_dict, strict=False)
    print(f"load_state_dict: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")

    # Step 3: Apply to_bfloat16_for_selected_params (same as __init__.py:96)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # Step 4: Inspect dtypes
    print("\n=== AFTER to_bfloat16_for_selected_params('bfloat16') ===")
    dtype_counts = Counter()
    fp32_params = []
    for name, param in model.named_parameters():
        dtype_counts[str(param.dtype)] += 1
        if param.dtype == torch.float32:
            fp32_params.append(name)

    print(f"Dtype distribution: {dict(dtype_counts)}")
    print(f"\nFloat32 parameters ({len(fp32_params)} total):")
    for p in fp32_params[:20]:
        print(f"  {p}")
    if len(fp32_params) > 20:
        print(f"  ... and {len(fp32_params) - 20} more")

    return model, fp32_params


def inspect_with_fsdp(model):
    """Wrap model with FSDP and inspect what happens to dtypes."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy, MixedPrecision

    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://")

    device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
    model = model.to(device)

    print("\n=== BEFORE FSDP wrapping (on GPU) ===")
    dtype_counts = Counter()
    for name, param in model.named_parameters():
        dtype_counts[str(param.dtype)] += 1
    print(f"Dtype distribution: {dict(dtype_counts)}")

    # Same MixedPrecision as training config (all None)
    mp = MixedPrecision(param_dtype=None, reduce_dtype=None, buffer_dtype=None)

    print("\n=== Attempting FSDP wrapping with use_orig_params=False ===")
    try:
        fsdp_model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.NO_SHARD,
            mixed_precision=mp,
            use_orig_params=False,
            device_id=device,
        )
        print("SUCCESS: FSDP wrapped without error")

        # Inspect FlatParameter
        dtype_counts_after = Counter()
        flat_params = []
        for name, param in fsdp_model.named_parameters():
            dtype_counts_after[str(param.dtype)] += 1
            if "flat_param" in name.lower() or param.shape[0] > 100000:
                flat_params.append((name, param.shape, param.dtype))

        print(f"After FSDP dtype distribution: {dict(dtype_counts_after)}")
        if flat_params:
            print(f"FlatParameter-like params:")
            for n, s, d in flat_params[:5]:
                print(f"  {n}: shape={s}, dtype={d}")

    except ValueError as e:
        print(f"FAILED: {e}")
        print("This confirms mixed dtypes crash FSDP with use_orig_params=False")

    print("\n=== Attempting FSDP wrapping with use_orig_params=True ===")
    # Reload model for second attempt
    model2 = model  # Reuse (can't re-wrap already-wrapped model easily)
    # Actually need fresh model - skip this if first attempt modified model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp", action="store_true", help="Test FSDP wrapping")
    args = parser.parse_args()

    model, fp32_params = inspect_before_fsdp()

    if args.fsdp:
        inspect_with_fsdp(model)
    else:
        print("\n=== Summary ===")
        print(f"Model has mixed dtypes: bf16 + {len(fp32_params)} fp32 params")
        print("Run with --fsdp flag to test FSDP behavior:")
        print("  torchrun --nproc_per_node=1 toolkits/behavior/inspect_fsdp_dtypes.py --fsdp")


if __name__ == "__main__":
    main()
