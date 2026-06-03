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

"""Self-contained PyTorch OpenPI 0.5 model package for embodied BEHAVIOR eval.

This package vendors the optimized PyTorch OpenPI 0.5 implementation so that the
eval / action-generation path is fully self-contained: it does not import the
externally installed ``openpi`` package and does not patch ``transformers``.

Layout:
  utils/                  vendored model core (pi0, gemma, siglip, ...).
  policies/               BEHAVIOR input/output transforms.
  normalize.py            norm-stats load + quantile (un)normalization.
  tokenizer.py            PaliGemma tokenizer (bundled SentencePiece asset).
  image_tools.py          PIL resize-with-pad (matches the old eval).
  processing.py           BehaviorEvalProcessor (env_obs -> model.Observation).
  openpi_action_model.py  high-level entry point preserving the old interface.
  convert_checkpoint.py   old-format -> new-format checkpoint converter.
"""

from __future__ import annotations

import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 eval model from a model config (factory entry).

    Expects ``cfg.model_path`` to point at a *new-format* checkpoint directory
    (produced by ``convert_checkpoint``) containing ``model.safetensors``,
    ``config.json``, and the ``physical-intelligence/behavior/norm_stats.json``
    asset tree.
    """
    import safetensors.torch
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.processing import BehaviorEvalProcessor
    from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer
    from rlinf.models.embodiment.openpi_pytorch.utils.pi0_config import Pi0Config

    def _select(path, default=None):
        value = OmegaConf.select(cfg, path) if not isinstance(cfg, dict) else cfg.get(path)
        return default if value is None else value

    # Phase 1 eval supports only the minimal BEHAVIOR path; fail loudly otherwise.
    for unsupported in ("openpi.full_pi05", "openpi.use_dsrl", "openpi.add_value_head"):
        if bool(_select(unsupported, False)):
            raise ValueError(
                f"openpi_pytorch (Phase 1 eval) does not support '{unsupported}'. "
                "Use the old 'openpi' model for full_pi05/DSRL/value-head paths."
            )
    config_name = _select("openpi.config_name", "pi05_behavior")
    if "behavior" not in str(config_name):
        raise ValueError(
            f"openpi_pytorch (Phase 1) supports only the BEHAVIOR env; got "
            f"config_name={config_name!r}."
        )

    model_path = pathlib.Path(_select("model_path"))
    config_json = {}
    if (model_path / "config.json").exists():
        config_json = json.loads((model_path / "config.json").read_text())

    pi0_config = Pi0Config(
        pi05=True,
        action_horizon=int(config_json.get("action_horizon", 32)),
        action_dim=int(config_json.get("action_dim", 32)),
        paligemma_variant=config_json.get("paligemma_variant", "gemma_2b"),
        action_expert_variant=config_json.get("action_expert_variant", "gemma_300m"),
        dtype="bfloat16",
        pcd=False,
    )
    model = pi0_config.create()
    weights_path = model_path / "model.safetensors"
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "openpi_pytorch: loaded %s (%.2fB params) strict from %s",
        config_json or pi0_config,
        n_params / 1e9,
        weights_path,
    )
    model = model.to(torch_dtype if torch_dtype is not None else torch.bfloat16)

    norm_stats = load_norm_stats(model_path / "physical-intelligence" / "behavior")
    tokenizer = PaligemmaTokenizer(max_len=pi0_config.max_token_len)

    action_chunk = int(_select("num_action_chunks", pi0_config.action_horizon))
    action_env_dim = int(_select("action_dim", 23))
    num_steps = int(_select("num_steps", 10))

    processor = BehaviorEvalProcessor(
        norm_stats,
        tokenizer,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        model_action_dim=pi0_config.action_dim,
    )
    return OpenPiPytorchActionModel(
        model,
        processor,
        num_steps=num_steps,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
    )
