# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""FP32 flow paths and side-effect-free Euler sampling."""

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


def time_bucket(tau: Tensor, buckets: int = 1000) -> Tensor:
    if tau.dtype != torch.float32:
        raise TypeError("Flow time must remain FP32 before discretization")
    if not ((tau >= 0) & (tau < 1)).all():
        raise ValueError("Flow time must be in [0, 1)")
    return (tau * buckets).long()


def sample_path(actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Independent Beta(1.5,1) time and Gaussian noise for every anchor."""
    with torch.autocast(actions.device.type, enabled=False):
        actions = actions.float()
        # Inverse CDF of Beta(a,1) avoids CPU sampling / dtype promotion.
        u = torch.rand(
            actions.shape[:-2], device=actions.device, dtype=torch.float32
        ).pow(1 / 1.5)
        tau = 0.999 * (1 - u)
        noise = torch.randn_like(actions)
        xt = (1 - tau[..., None, None]) * noise + tau[..., None, None] * actions
        return xt, actions - noise, time_bucket(tau)


def sequence_loss(
    prediction: Tensor,
    target: Tensor,
    action_mask: Tensor,
    anchor_valid: Tensor,
    supervision: Tensor,
    denominator: Tensor,
) -> Tensor:
    """Per-sequence anchor mean, then batch mean; denominator spans all segments."""
    with torch.autocast(prediction.device.type, enabled=False):
        weight = action_mask.float()
        per_anchor = ((prediction.float() - target.float()).square() * weight).sum(
            (-1, -2)
        ) / weight.sum((-1, -2)).clamp_min(1)
        return (
            (per_anchor * anchor_valid * supervision).sum(-1) / denominator.clamp_min(1)
        ).mean()


def euler_sample(
    velocity: Callable[[Tensor, Tensor, Any], tuple[Tensor, Any]],
    noise: Tensor,
    incoming: Any,
    steps: int = 4,
    write_source: str = "final_prediction",
    write_index: int = -1,
) -> tuple[Tensor, Any]:
    """All branches read incoming; return ONE selected candidate for the caller.

    This function never commits state. The additional final-prediction forward
    cannot change the already integrated action returned to the environment.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    if write_source not in {"final_prediction", "denoise_step"}:
        raise ValueError(f"Unknown memory write source: {write_source}")
    index = steps - 1 if write_index == -1 else write_index
    if write_source == "denoise_step" and not 0 <= index < steps:
        raise ValueError("Memory write index outside the denoising trajectory")
    actions = noise.float()
    selected = None
    for i in range(steps):
        tau = torch.full(
            actions.shape[:1], i / steps, device=actions.device, dtype=torch.float32
        )
        pred, candidate = velocity(actions, time_bucket(tau), incoming)
        actions = actions + pred.float() / steps
        if write_source == "denoise_step" and i == index:
            selected = candidate
    if write_source == "final_prediction":
        bucket = torch.full(
            actions.shape[:1], 999, device=actions.device, dtype=torch.long
        )
        _, selected = velocity(actions, bucket, incoming)
    return actions, selected
