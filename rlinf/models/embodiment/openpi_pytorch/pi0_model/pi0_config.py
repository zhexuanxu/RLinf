# Copyright (c) 2025, RLinf contributors.
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
    # PaliGemma backbone emit subtask tokens: SFT adds a CE loss over the
    # response + EOS, and eval generates the subtask before denoising actions.
    mode: str = "vla"
    # SFT loss combination: total = language_loss_weight * ce_loss
    #                             + action_loss_weight * flow_loss.
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    # Detach the prefix K/V seen by the action expert so the flow-matching
    # gradient reaches only the action expert while the CE loss still trains
    # the full VLM (knowledge insulation).
    stop_gradient_to_vlm: bool = False
    # Subtask generation budget and sampling temperature (0 = greedy).
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

    def create(self, **kwargs) -> model.BaseModel:
        from .pi0 import Pi0

        return Pi0(self)

    def fake_obs(self, batch_size: int = 1) -> model.Observation:
        image = torch.ones(batch_size, *model.IMAGE_RESOLUTION, 3)
        image_mask = torch.ones(batch_size, dtype=torch.bool)
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
            pcd_xyz=torch.ones(batch_size, 16, 2025, 3) if self.pcd else None,
        )

    def fake_act(self, batch_size: int = 1) -> torch.Tensor:
        return torch.ones(batch_size, self.action_horizon, self.action_dim)
