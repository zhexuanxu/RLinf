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

"""VLM models for embodied dual-system agentloop."""

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load a VLM model for the dual-system embodied agentloop.

    Supported model types:
        - ``qwen2.5_vl_embodied`` — Qwen2.5-VL series
        - ``qwen3_vl_embodied``   — Qwen3-VL-Thinking series

    Args:
        cfg: Model configuration. Must contain ``model_type`` and ``model_path``.
        torch_dtype: Optional torch dtype override. Defaults to ``torch.bfloat16``.

    Returns:
        A VLM policy instance (not yet moved to device).
    """
    model_type = cfg.model_type
    if model_type == "qwen2.5_vl_embodied":
        from rlinf.models.embodiment.VLM.qwen2_5_vl_policy import Qwen2_5_VLPolicy

        return Qwen2_5_VLPolicy(cfg, torch_dtype=torch_dtype)
    elif model_type == "qwen3_vl_embodied":
        from rlinf.models.embodiment.VLM.qwen3_vl_policy import Qwen3_VLPolicy

        return Qwen3_VLPolicy(cfg, torch_dtype=torch_dtype)
    else:
        raise NotImplementedError(
            f"Unsupported VLM model type: {model_type}. "
            "Supported: 'qwen2.5_vl_embodied', 'qwen3_vl_embodied'."
        )
