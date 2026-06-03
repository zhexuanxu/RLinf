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

"""Entry point for the self-contained PyTorch OpenPI 0.5 BEHAVIOR model.

This model focuses on exactly two responsibilities: eval action sampling
(implemented here) and SFT loss (reserved for a future training implementation).
Its high-level interface mirrors the old ``OpenPi0ForRLActionPrediction`` so the eval
rollout worker can call it unchanged via the OpenPI dispatch path:

    actions, result = model.predict_action_batch(env_obs=env_obs, mode="eval")

``actions`` has shape ``[B, action_chunk, action_env_dim]`` (e.g. ``[B, 32, 23]``).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi_pytorch.processing import BehaviorEvalProcessor
from rlinf.models.embodiment.openpi_pytorch.utils.pi0 import Pi0


class OpenPiPytorchActionModel(nn.Module):
    """Wrap the vendored ``Pi0`` model with BEHAVIOR eval pre/post-processing."""

    def __init__(
        self,
        pi0_model: Pi0,
        processor: BehaviorEvalProcessor,
        *,
        num_steps: int,
        action_chunk: int,
        action_env_dim: int,
    ):
        super().__init__()
        self.model = pi0_model
        self.processor = processor
        self.num_steps = num_steps
        self.action_chunk = action_chunk
        self.action_env_dim = action_env_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions for a batch of observations (eval / action generation)."""
        observation = self.processor.build_observation(env_obs, self.device)
        model_actions = self.model.sample_actions(observation, num_steps=self.num_steps)
        actions = self.processor.postprocess_actions(model_actions).to(self.device)

        batch = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(batch, -1).contiguous(),
                "model_action": model_actions.reshape(batch, -1).contiguous(),
            },
        }
        return actions, result

    # --- Reserved for a future training implementation (SFT); not implemented here. ---
    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError(
            "SFT loss computation is not implemented for this eval-only model; "
            "it is reserved for a future training implementation."
        )

    def sft_forward(self, *args, **kwargs):
        raise NotImplementedError(
            "SFT training is not implemented for this eval-only model; it is "
            "reserved for a future training implementation."
        )
