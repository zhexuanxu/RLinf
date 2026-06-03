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

import hashlib
import json
import logging
import pathlib

import torch

logger = logging.getLogger(__name__)


def _state_dict_metadata_digest(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a stable digest over checkpoint key/dtype/shape metadata."""
    hasher = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key]
        hasher.update(key.encode("utf-8"))
        hasher.update(str(tensor.dtype).encode("utf-8"))
        hasher.update(str(tuple(tensor.shape)).encode("utf-8"))
    return hasher.hexdigest()[:16]


def _file_digest(path: pathlib.Path) -> str:
    """Return a short SHA256 digest of a local file's bytes."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def _validate_checkpoint_state_dict(
    checkpoint_state: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    *,
    expected_dtype: torch.dtype,
) -> None:
    """Validate checkpoint keys, shapes, and tensor dtypes before loading."""
    checkpoint_keys = set(checkpoint_state)
    model_keys = set(model_state)
    missing = sorted(model_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - model_keys)
    if missing or unexpected:
        raise ValueError(
            "openpi_pytorch checkpoint key mismatch: "
            f"missing={missing[:10]} unexpected={unexpected[:10]} "
            f"(missing_count={len(missing)}, unexpected_count={len(unexpected)})"
        )

    shape_mismatches = []
    dtype_mismatches = []
    for key in sorted(model_keys):
        ckpt_tensor = checkpoint_state[key]
        model_tensor = model_state[key]
        if tuple(ckpt_tensor.shape) != tuple(model_tensor.shape):
            shape_mismatches.append(
                (key, tuple(model_tensor.shape), tuple(ckpt_tensor.shape))
            )
        if ckpt_tensor.dtype != expected_dtype:
            dtype_mismatches.append((key, expected_dtype, ckpt_tensor.dtype))

    if shape_mismatches:
        preview = ", ".join(
            f"{key}: expected {expected}, got {actual}"
            for key, expected, actual in shape_mismatches[:10]
        )
        raise ValueError(
            "openpi_pytorch checkpoint shape mismatch: "
            f"{preview} (count={len(shape_mismatches)})"
        )
    if dtype_mismatches:
        preview = ", ".join(
            f"{key}: expected {expected}, got {actual}"
            for key, expected, actual in dtype_mismatches[:10]
        )
        raise ValueError(
            "openpi_pytorch checkpoint dtype mismatch: "
            f"{preview} (count={len(dtype_mismatches)})"
        )


def get_model(cfg, torch_dtype=None):
    """Build the BEHAVIOR pi05 eval model from a model config (factory entry).

    Expects ``cfg.model_path`` to point at a *new-format* checkpoint directory
    (produced by ``convert_checkpoint``) containing ``model.safetensors``,
    ``config.json``, and the ``physical-intelligence/behavior/norm_stats.json``
    asset tree.
    """
    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.processing import BehaviorEvalProcessor
    from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer
    from rlinf.models.embodiment.openpi_pytorch.utils.pi0_config import Pi0Config

    def _select(path, default=None):
        if isinstance(cfg, dict):
            value = cfg
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    return default
                value = value[part]
        else:
            value = OmegaConf.select(cfg, path)
        return default if value is None else value

    # This eval-only model supports only the minimal BEHAVIOR path; fail loudly otherwise.
    for unsupported in (
        "add_value_head",
        "openpi.full_pi05",
        "openpi.use_dsrl",
        "openpi.add_value_head",
    ):
        if bool(_select(unsupported, False)):
            raise ValueError(
                f"openpi_pytorch (eval-only) does not support '{unsupported}'. "
                "Use the old 'openpi' model for full_pi05/DSRL/value-head paths."
            )
    if torch_dtype not in (None, torch.bfloat16):
        raise ValueError(
            "openpi_pytorch (eval-only) supports only precision=null or bf16. "
            f"Got torch_dtype={torch_dtype}."
        )
    precision = _select("precision", None)
    if precision not in (None, "null", "bf16", "bf16-mixed"):
        raise ValueError(
            "openpi_pytorch (eval-only) supports only precision=null or bf16. "
            f"Got precision={precision!r}."
        )
    config_name = _select("openpi.config_name", "pi05_behavior")
    if "behavior" not in str(config_name):
        raise ValueError(
            f"openpi_pytorch supports only the BEHAVIOR env; got "
            f"config_name={config_name!r}."
        )

    model_path = pathlib.Path(_select("model_path"))
    weights_path = model_path / "model.safetensors"
    norm_stats_path = (
        model_path / "physical-intelligence" / "behavior" / "norm_stats.json"
    )
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"openpi_pytorch norm stats not found: {norm_stats_path}"
        )
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
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    _validate_checkpoint_state_dict(
        state_dict, model.state_dict(), expected_dtype=torch.bfloat16
    )
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        "openpi_pytorch: loaded %s (%.2fB params) strict from %s "
        "state_metadata_digest=%s norm_stats_digest=%s",
        config_json or pi0_config,
        n_params / 1e9,
        weights_path,
        _state_dict_metadata_digest(state_dict),
        _file_digest(norm_stats_path),
    )
    model = model.to(torch_dtype if torch_dtype is not None else torch.bfloat16)

    norm_stats = load_norm_stats(norm_stats_path.parent)
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
