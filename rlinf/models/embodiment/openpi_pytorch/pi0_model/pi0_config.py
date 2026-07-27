# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pi0 config for PyTorch, aligned with JAX models/pi0_config.py."""

from __future__ import annotations

import dataclasses

import torch

from . import gemma, model, pointnet


@dataclasses.dataclass
class Pi0Config(model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: gemma.Variant = "gemma_2b"
    action_expert_variant: gemma.Variant = "gemma_300m"
    pointnet_variant: pointnet.Variant = "pcd"

    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 48

    pi05: bool = False
    discrete_state_input: bool | None = None
    pcd: bool = False

    # VLM token output. "vla" is the action-only behavior; "vlm_vla" lets the
    # PaliGemma backbone emit subtask tokens before the action expert denoises
    # actions.
    mode: str = "vla"
    # SFT combines the language and flow-matching losses with these weights.
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # Prevent the flow-matching loss from updating the VLM prefix while still
    # allowing the language loss to train it.
    stop_gradient_to_vlm: bool = False
    # Subtask generation settings (temperature 0 means greedy decoding).
    max_new_tokens: int = 24
    language_temperature: float = 0.0

    def __post_init__(self):
        if self.pi05 and self.max_token_len == 48:
            object.__setattr__(self, "max_token_len", 200)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.mode not in ("vla", "vlm_vla"):
            raise ValueError(f"mode must be 'vla' or 'vlm_vla', got {self.mode!r}")
        if self.mode == "vlm_vla" and not self.pi05:
            raise ValueError("vlm_vla mode requires the pi05 configuration")
        if self.language_loss_weight < 0 or self.action_loss_weight < 0:
            raise ValueError("language and action loss weights must be non-negative")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.language_temperature < 0:
            raise ValueError("language_temperature must be non-negative")

    def create(self, **kwargs) -> model.BaseModel:
        from .pi0 import Pi0

        return Pi0(self)

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
        token_ar_mask = None
        token_loss_mask = None
        token_kv_cache_mask = None
        if self.mode == "vlm_vla":
            token_ar_mask = torch.zeros(
                batch_size, self.max_token_len, dtype=torch.bool
            )
            token_ar_mask[:, -1] = True
            token_loss_mask = torch.zeros_like(token_ar_mask)
            token_loss_mask[:, -1] = True
            token_kv_cache_mask = torch.ones_like(token_ar_mask)
            token_kv_cache_mask[:, -1] = False
        return model.Observation(
            images={
                "base_0_rgb": image,
                "left_wrist_0_rgb": image,
                "right_wrist_0_rgb": image,
            },
            image_masks={
                "base_0_rgb": image_mask,
                "left_wrist_0_rgb": image_mask,
                "right_wrist_0_rgb": image_mask,
            },
            state=torch.ones(batch_size, self.action_dim),
            tokenized_prompt=torch.ones(
                batch_size, self.max_token_len, dtype=torch.long
            ),
            tokenized_prompt_mask=torch.ones(
                batch_size, self.max_token_len, dtype=torch.bool
            ),
            token_ar_mask=token_ar_mask,
            token_loss_mask=token_loss_mask,
            token_kv_cache_mask=token_kv_cache_mask,
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
