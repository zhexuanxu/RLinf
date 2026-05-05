#!/usr/bin/env python3
"""Verify that full pi0.5 and lite pi0.5 produce identical SFT forward pass outputs
under VLA-only mode.

This script creates both model variants, loads the same checkpoint, and runs
one forward pass with identical noise/time/observation. If the losses differ,
there's a functional difference between the two forward paths.

Usage:
    CUDA_VISIBLE_DEVICES=0 python toolkits/behavior/verify_full_vs_lite_forward.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from openpi.models import model as _model
from rlinf.models.embodiment.openpi.openpi_action_model import (
    OpenPi0Config,
    OpenPi0ForRLActionPrediction,
)
from rlinf.models.embodiment.openpi.openpi_full_pi05_model import (
    OpenPi05FullForRLActionPrediction,
)
import openpi.models.pi0_config as pi0_config
import safetensors.torch

CKPT_PATH = "/mnt/public/xzxuan/models/pi05_base_pytorch"


def create_model(full_pi05: bool):
    """Create model variant."""
    cfg = OpenPi0Config(**pi0_config.Pi0Config(pi05=True, action_horizon=32).__dict__)
    cfg.__dict__["full_pi05"] = full_pi05
    cfg.__dict__["forward_mode"] = "vla"

    if full_pi05:
        model = OpenPi05FullForRLActionPrediction(cfg)
    else:
        model = OpenPi0ForRLActionPrediction(cfg)

    # Load weights (same as training __init__.py)
    state_dict = safetensors.torch.load_file(
        os.path.join(CKPT_PATH, "model.safetensors"), device="cpu"
    )
    model.load_state_dict(state_dict, strict=False)
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    model.eval().cuda()
    return model


def create_observation(batch_size=2, device="cuda"):
    """Create a fixed observation for both models."""
    # Mimic what the data pipeline produces AFTER all transforms:
    # Images: float32 [-1, 1], shape [B, H, W, C]
    # State: float32 [B, 32]
    # Tokenized prompt: int32 [B, 200]
    # Tokenized prompt mask: bool [B, 200]
    torch.manual_seed(42)
    np.random.seed(42)

    # NOTE: Images must be [B, C, H, W] (channel-first) because:
    # - The full model's _preprocess_observation checks shape[-1]==3 for permutation
    # - PI0Pytorch's preprocess_observation_pytorch checks shape[1]==3 for channel detection
    # - Both work correctly with [B, C, H, W] input
    images = {
        "base_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
        "left_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
        "right_wrist_0_rgb": torch.randn(batch_size, 3, 224, 224, device=device) * 0.5,
    }
    image_masks = {
        "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
        "right_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
    state = torch.randn(batch_size, 32, device=device) * 0.1
    # Simple tokenized prompt (pad most of it)
    tokenized_prompt = torch.zeros(batch_size, 200, dtype=torch.int32, device=device)
    tokenized_prompt[:, :10] = torch.randint(100, 5000, (batch_size, 10))
    tokenized_prompt_mask = torch.zeros(batch_size, 200, dtype=torch.bool, device=device)
    tokenized_prompt_mask[:, :10] = True

    obs = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    return obs


def run_lite_forward(model, obs, actions, noise, time):
    """Run the parent PI0Pytorch.forward() — this is what lite sft_forward calls."""
    return model.paligemma_with_expert.paligemma  # just to get the class
    # Actually call the parent's forward directly
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    # The lite model's sft_forward calls: super().forward(observation, actions)
    # Which is PI0Pytorch.forward(observation, actions, noise, time)
    loss = PI0Pytorch.forward(model, obs, actions, noise=noise, time=time)
    return loss


def run_full_forward(model, obs, actions, noise, time):
    """Run the full pi0.5 _forward_vla_full()."""
    from rlinf.models.embodiment.base_policy import ForwardType

    data = {"observation": obs, "actions": actions}
    result = model(forward_type=ForwardType.SFT, data=data)
    return result


def main():
    print("=== Creating models ===")
    lite_model = create_model(full_pi05=False)
    full_model = create_model(full_pi05=True)
    print(f"Lite model class: {type(lite_model).__name__}")
    print(f"Full model class: {type(full_model).__name__}")

    # Verify weights are identical
    lite_sd = {k: v for k, v in lite_model.state_dict().items()}
    full_sd = {k: v for k, v in full_model.state_dict().items()}
    common = set(lite_sd.keys()) & set(full_sd.keys())
    print(f"Common keys: {len(common)}, Lite-only: {len(set(lite_sd)-common)}, Full-only: {len(set(full_sd)-common)}")
    weight_diff = max(
        (lite_sd[k].float() - full_sd[k].float()).abs().max().item()
        for k in list(common)[:100]
    )
    print(f"Max weight diff (sample 100 keys): {weight_diff}")

    print("\n=== Creating observation and actions ===")
    obs = create_observation(batch_size=2)
    actions = torch.randn(2, 32, 32, device="cuda") * 0.3  # [B, action_horizon, action_dim]
    # Fix noise and time for reproducibility
    noise = torch.randn_like(actions)
    time = torch.tensor([0.5, 0.7], device="cuda")

    # Both models use PI0Pytorch.forward() internally. The difference is:
    # - Lite: sft_forward() → super().forward(observation, actions) → PI0Pytorch.forward()
    # - Full: sft_forward() → _forward_vla_full() → _compute_velocity_with_prefix_out()
    #
    # We call them via the SFT worker interface (forward_type=SFT)

    from rlinf.models.embodiment.base_policy import ForwardType

    print("\n=== Running LITE model (sft_forward → PI0Pytorch.forward) ===")
    with torch.no_grad():
        # Monkey-patch noise/time for reproducibility on BOTH models
        orig_lite_noise = lite_model.sample_noise
        orig_lite_time = lite_model.sample_time
        lite_model.sample_noise = lambda shape, device: noise.clone()
        lite_model.sample_time = lambda bsize, device: time.clone()

        data = {"observation": obs, "actions": actions}
        lite_result = lite_model(forward_type=ForwardType.SFT, data=data)

        lite_model.sample_noise = orig_lite_noise
        lite_model.sample_time = orig_lite_time

    # Lite returns a raw tensor (from PI0Pytorch.forward)
    if isinstance(lite_result, dict):
        lite_loss_scalar = lite_result["loss"].item()
        print(f"  Loss: {lite_loss_scalar:.8f}")
    else:
        lite_loss_scalar = lite_result.mean().item()
        print(f"  Loss tensor shape: {lite_result.shape}")
        print(f"  Loss mean: {lite_loss_scalar:.8f}")

    print("\n=== Running FULL model (sft_forward → _forward_vla_full) ===")
    with torch.no_grad():
        orig_full_noise = full_model.sample_noise
        orig_full_time = full_model.sample_time
        full_model.sample_noise = lambda shape, device: noise.clone()
        full_model.sample_time = lambda bsize, device: time.clone()

        data = {"observation": obs, "actions": actions}
        full_result = full_model(forward_type=ForwardType.SFT, data=data)

        full_model.sample_noise = orig_full_noise
        full_model.sample_time = orig_full_time

    full_loss_scalar = full_result["loss"].item()
    full_action_loss = full_result["action_loss"].item()
    print(f"  Loss (total): {full_loss_scalar:.8f}")
    print(f"  Action loss: {full_action_loss:.8f}")

    print("\n=== COMPARISON ===")
    diff = abs(lite_loss_scalar - full_loss_scalar)
    print(f"  Lite loss:  {lite_loss_scalar:.8f}")
    print(f"  Full loss:  {full_loss_scalar:.8f}")
    print(f"  Difference: {diff:.2e}")

    if diff < 1e-5:
        print("\n  ✓ PASS: Full and lite produce IDENTICAL losses (diff < 1e-5)")
    elif diff < 1e-3:
        print(f"\n  ~ CLOSE: Losses differ by {diff:.2e} (likely floating-point precision)")
    else:
        print(f"\n  ✗ FAIL: Losses differ significantly by {diff:.2e}")
        print("  The full and lite forward paths are NOT equivalent!")


if __name__ == "__main__":
    main()
