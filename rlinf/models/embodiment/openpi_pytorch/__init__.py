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
  pi0_model/              vendored model core + preprocessing (pi0, gemma,
                          siglip, normalize, processing, tokenizer, ...).
  utils/                  checkpoint conversion + image tooling
                          (jax_to_new_pytorch, old_to_new, new_to_old,
                          export_sft_checkpoint, image_tools).
  policies/               BEHAVIOR input/output transforms.
  dataconfig/             BEHAVIOR data config (mirrors openpi/dataconfig).
"""

from __future__ import annotations

import hashlib
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
    """Build the BEHAVIOR pi05 model from a model config (factory entry).

    Expects ``cfg.model_path`` to point at a *new-format* checkpoint directory
    containing ``model.safetensors``. The Pi0 model shape is built entirely from
    YAML fields (``num_action_chunks`` plus ``openpi.model_action_dim`` /
    ``openpi.paligemma_variant`` / ``openpi.action_expert_variant``); a checkpoint
    ``config.json`` is never read. Eval builds resolve the BEHAVIOR norm stats via
    ``openpi.assets_dir`` + ``openpi.asset_id`` (the same canonical task-0000 stats
    the SFT data loader resolves). Training builds set ``load_for_training=True``
    (or ``openpi.load_for_training=True``), load fp32 new-format weights strictly,
    then cast to bf16 for SFT (norm stats are left to the data loader).
    """
    import safetensors.torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
        load_norm_stats,
        resolve_norm_stats_dir,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.processing import (
        BehaviorEvalProcessor,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer import (
        PaligemmaTokenizer,
    )

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

    load_for_training = bool(
        _select("load_for_training", False)
        or _select("openpi.load_for_training", False)
    )

    # This model supports only the minimal BEHAVIOR path; fail loudly otherwise.
    for unsupported in (
        "add_value_head",
        "openpi.full_pi05",
        "openpi.use_dsrl",
        "openpi.add_value_head",
    ):
        if bool(_select(unsupported, False)):
            raise ValueError(
                f"openpi_pytorch does not support '{unsupported}'. "
                "Use the old 'openpi' model for full_pi05/DSRL/value-head paths."
            )
    if torch_dtype not in (None, torch.bfloat16):
        raise ValueError(
            "openpi_pytorch supports only precision=null or bf16. "
            f"Got torch_dtype={torch_dtype}."
        )
    precision = _select("precision", None)
    if precision not in (None, "null", "bf16", "bf16-mixed"):
        raise ValueError(
            "openpi_pytorch supports only precision=null or bf16. "
            f"Got precision={precision!r}."
        )

    # Model-shape fields come exclusively from the YAML model config (no
    # checkpoint config.json read). Missing required fields fail loudly.
    def _require_shape(key, label):
        value = _select(key)
        if value is None:
            raise ValueError(
                f"openpi_pytorch requires actor.model.{key} ({label}) in the YAML "
                "model config; model-shape fields are not read from a checkpoint "
                "config.json."
            )
        return value

    model_path = pathlib.Path(_select("model_path"))
    weights_path = model_path / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"openpi_pytorch checkpoint not found: {weights_path}")

    pi0_config = Pi0Config(
        pi05=True,
        action_horizon=int(_require_shape("num_action_chunks", "action_horizon")),
        action_dim=int(_require_shape("openpi.model_action_dim", "model action_dim")),
        paligemma_variant=str(
            _require_shape("openpi.paligemma_variant", "paligemma_variant")
        ),
        action_expert_variant=str(
            _require_shape("openpi.action_expert_variant", "action_expert_variant")
        ),
        dtype="bfloat16",
        pcd=False,
    )
    model = pi0_config.create()
    state_dict = safetensors.torch.load_file(str(weights_path), device="cpu")
    expected_dtype = torch.float32 if load_for_training else torch.bfloat16
    _validate_checkpoint_state_dict(
        state_dict, model.state_dict(), expected_dtype=expected_dtype
    )
    model.load_state_dict(state_dict, strict=True)
    n_params = sum(p.numel() for p in model.parameters())

    # Norm stats resolve from YAML assets_dir + asset_id (the SAME resolution the
    # SFT data loader uses, so eval and SFT share the canonical task-0000 stats).
    # Both fields are required for eval — there is NO hard-coded asset-id default,
    # so eval can never silently load a non-task-0000 distribution. Training leaves
    # the processor (and thus norm stats) to the data loader.
    assets_dir = _select("openpi.assets_dir")
    asset_id = _select("openpi.asset_id")
    norm_stats = None
    norm_stats_dir = None
    if not load_for_training:
        if assets_dir is None:
            missing = "assets_dir"
        elif asset_id is None:
            missing = "asset_id"
        else:
            missing = None
        if missing is not None:
            raise FileNotFoundError(
                f"openpi_pytorch eval requires actor.model.openpi.{missing} (no "
                "default is applied); the canonical BEHAVIOR task-0000 norm stats "
                "are resolved strictly from YAML."
            )
        norm_stats_dir = resolve_norm_stats_dir(assets_dir, asset_id)
        norm_stats = load_norm_stats(norm_stats_dir)

    if norm_stats_dir is not None:
        norm_stats_digest = _file_digest(norm_stats_dir / "norm_stats.json")
    else:
        norm_stats_digest = "deferred"
    logger.info(
        "openpi_pytorch: loaded %s (%.2fB params) strict from %s for %s "
        "state_metadata_digest=%s norm_stats_digest=%s",
        pi0_config,
        n_params / 1e9,
        weights_path,
        "training" if load_for_training else "eval",
        _state_dict_metadata_digest(state_dict),
        norm_stats_digest,
    )
    target_dtype = torch_dtype if torch_dtype is not None else torch.bfloat16
    model = model.to(target_dtype)
    if load_for_training:
        model.gradient_checkpointing_enable()

    action_chunk = int(_select("num_action_chunks", pi0_config.action_horizon))
    action_env_dim = int(_select("action_dim", 23))
    num_steps = int(_select("num_steps", 10))

    processor = None
    if norm_stats is not None:
        tokenizer = PaligemmaTokenizer(max_len=pi0_config.max_token_len)
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
