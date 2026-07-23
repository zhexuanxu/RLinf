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

"""Self-contained PyTorch OpenPI 0.5 model package for embodied BEHAVIOR.

This package vendors the optimized PyTorch OpenPI 0.5 implementation so that the
eval / action-generation path is fully self-contained: it does not import the
externally installed ``openpi`` package and does not patch ``transformers``.

Layout:
  openpi_action_model.py  eval action sampling and SFT-loss entry point.
  pi0_model/              vendored model core (pi0, gemma, siglip, ...).
  utils/                  normalization, tokenizer, and image tooling.
  policies/               BEHAVIOR input/output transforms.

The BEHAVIOR streaming SFT dataset / data loader lives under
``rlinf.data.datasets.openpi_pytorch.behavior``; checkpoint conversion lives
under ``rlinf.utils.ckpt_convertor.openpi``.
"""

from __future__ import annotations

import logging
import pathlib

from rlinf.config import torch_dtype_from_precision

logger = logging.getLogger(__name__)


def _get_discrete_state_input(model_cfg) -> bool:
    """Read the pi05 language-state switch from YAML."""
    value = model_cfg.get("discrete_state_input", True)
    if not isinstance(value, bool):
        raise TypeError(
            f"actor.model.openpi.discrete_state_input must be a boolean, got {value!r}."
        )
    return value


def _get_state_order(model_cfg) -> str:
    """Read + validate the proprio->state channel layout from YAML.

    Canonical key is ``openpi.state_token`` (abs_joint_old / abs_joint / abs_eef);
    the legacy ``openpi.state_order`` (comet / align) still resolves.
    """
    from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
        resolve_state_token,
    )

    value = model_cfg.get("state_token", model_cfg.get("state_order", "abs_joint_old"))
    return resolve_state_token(str(value))


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 model from a model config (factory entry).

    ``cfg`` is ``actor.model``; ``cfg.model_path`` points at a *new-format*
    checkpoint directory containing ``model.safetensors``. The Pi0 model shape is
    built entirely from YAML fields (``num_action_chunks`` plus
    ``openpi.model_action_dim`` / ``openpi.paligemma_variant`` /
    ``openpi.action_expert_variant``); a checkpoint ``config.json`` is never read.

    The model dtype is precision-driven: ``precision: fp32`` keeps fp32 weights as
    the FSDP master (FSDP MixedPrecision casts to bf16 for compute and the
    optimizer updates the fp32 master, so warmup-LR updates are not lost to bf16
    rounding), while ``precision: bf16`` casts the weights to bf16 for eval. Norm
    stats and the PaliGemma tokenizer are resolved from YAML (``openpi.assets_dir``
    + ``openpi.asset_id`` and ``openpi.paligemma_tokenizer``); gradient
    checkpointing is governed by the FSDP manager
    (``fsdp_config.gradient_checkpointing``), not here.
    """
    import safetensors.torch

    from rlinf.data.datasets.openpi_pytorch import get_eval_processer
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
        PaligemmaTokenizer,
    )

    model_cfg = cfg.openpi

    # Precision drives the weight dtype; the compute dtype (FSDP MixedPrecision
    # param_dtype) is a separate knob configured in the experiment YAML.
    target_dtype = (
        torch_dtype
        if torch_dtype is not None
        else torch_dtype_from_precision(cfg.precision)
    )

    model_path = pathlib.Path(cfg.model_path)
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    discrete_state_input = _get_discrete_state_input(model_cfg)
    pi0_config = Pi0Config(
        pi05=True,
        discrete_state_input=discrete_state_input,
        action_horizon=int(cfg.num_action_chunks),
        action_dim=int(model_cfg.model_action_dim),
        max_token_len=int(model_cfg.get("max_token_len", 200)),
        paligemma_variant=str(model_cfg.paligemma_variant),
        action_expert_variant=str(model_cfg.action_expert_variant),
        # Compute dtype is bf16 (the FSDP MixedPrecision compute dtype). The
        # `precision` knob controls the MASTER weight dtype (fp32 master + bf16
        # compute under FSDP), NOT the compute dtype — so this must stay bf16 or
        # training hits a Float-vs-BFloat16 mismatch. (A one-off fully-fp32 eval
        # for the bf16-vs-fp32 diagnostic was done separately and showed no
        # difference, so honoring precision here is unnecessary and unsafe.)
        dtype="bfloat16",
        pcd=False,
        mode=str(model_cfg.get("mode", "vla")),
        language_loss_weight=float(model_cfg.get("language_loss_weight", 1.0)),
        action_loss_weight=float(model_cfg.get("action_loss_weight", 1.0)),
        stop_gradient_to_vlm=bool(model_cfg.get("stop_gradient_to_vlm", False)),
        max_new_tokens=int(model_cfg.get("max_new_tokens", 24)),
        language_temperature=float(model_cfg.get("language_temperature", 0.0)),
    )
    model = pi0_config.create()
    # Strict load enforces key/shape parity. Weights are materialized in fp32, so a
    # bf16 base checkpoint widens losslessly into the fp32 master (the intended SFT
    # init); the dtype cast below then sets the requested weight precision.
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    if target_dtype is not None:
        model = model.to(target_dtype)

    num_steps = int(cfg.num_steps)
    action_chunk = int(cfg.num_action_chunks)
    action_env_dim = int(cfg.action_dim)

    # Norm stats + tokenizer resolve strictly from YAML. control_mode selects the
    # stats asset via the SAME resolver contract the SFT loader uses, so eval and
    # SFT resolve one delta asset by mode. For the BEHAVIOR env control_mode is
    # required (no silent default); non-behavior envs keep the base fields.
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
        validate_norm_stats_for_control_mode,
    )

    env_type = model_cfg.get("env", "behavior")
    if env_type == "behavior":
        if "control_mode" not in model_cfg:
            raise ValueError(
                "actor.model.openpi.control_mode is required for BEHAVIOR pi0.5 "
                "eval (one of joint_absolute / eef_delta_pose)."
            )
        control_mode = str(model_cfg.control_mode)
        from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef_delta import (
            resolve_norm_stats_asset,
        )

        assets_dir, asset_id = resolve_norm_stats_asset(model_cfg, control_mode)
    else:
        control_mode = str(model_cfg.get("control_mode", "joint_absolute"))
        assets_dir, asset_id = model_cfg.assets_dir, model_cfg.asset_id

    norm_stats = load_norm_stats(assets_dir, asset_id)
    # Reject a stats asset whose manifest disagrees with the run's control mode /
    # action dim, so a 21-dim delta action can never be normalized against a
    # 23-dim joint stats file.
    validate_norm_stats_for_control_mode(
        assets_dir, asset_id, control_mode, action_env_dim,
        model_action_dim=int(pi0_config.action_dim),
    )
    tokenizer = PaligemmaTokenizer(
        model_cfg.paligemma_tokenizer, max_len=pi0_config.max_token_len
    )
    # The eval processor is selected by env (``env_type`` resolved above;
    # ``openpi.env`` defaults to "behavior", the only env registered today).
    state_order = _get_state_order(model_cfg)
    processor = get_eval_processer(
        env_type,
        norm_stats,
        tokenizer,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        model_action_dim=pi0_config.action_dim,
        vlm_vla=(pi0_config.mode == "vlm_vla"),
        discrete_state_input=discrete_state_input,
        state_order=state_order,
    )

    logger.info(
        "openpi_pytorch: loaded %s (%.2fB params) strict from %s precision=%s "
        "num_steps=%s discrete_state_input=%s state_order=%s",
        pi0_config,
        n_params / 1e9,
        weights_path,
        cfg.precision,
        num_steps,
        discrete_state_input,
        state_order,
    )
    return OpenPiPytorchActionModel(
        model,
        processor,
        num_steps=num_steps,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
    )
