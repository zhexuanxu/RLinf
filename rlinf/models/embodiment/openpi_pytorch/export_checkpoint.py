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

"""Export an SFT-trained checkpoint into the Phase-1 eval directory format.

An SFT run saves the :class:`OpenPiPytorchActionModel` wrapper state dict (keys
prefixed ``model.`` because the vendored ``Pi0`` lives at ``wrapper.model``).
The eval ``get_model`` factory instead loads a bare ``Pi0`` from a *new-format*
checkpoint directory containing ``model.safetensors`` (bf16, bare ``Pi0`` keys),
``config.json``, and the ``physical-intelligence/behavior/norm_stats.json``
asset tree. This module bridges the two so a model trained here can be evaluated
through the unchanged Phase-1 eval path.
"""

from __future__ import annotations

import json
import logging
import pathlib
import shutil
from typing import Any, Mapping

import torch

logger = logging.getLogger(__name__)

_WRAPPER_PREFIX = "model."
_NORM_STATS_SUBDIR = pathlib.Path("physical-intelligence") / "behavior"


def _strip_wrapper_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Drop the ``model.`` wrapper prefix and cast float tensors to bf16.

    The eval loader validates that every checkpoint tensor is bf16, so float
    parameters are cast; integer/bool buffers (if any) are passed through.
    """
    bare: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        bare_key = key[len(_WRAPPER_PREFIX):] if key.startswith(_WRAPPER_PREFIX) else key
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        bare[bare_key] = tensor.detach().cpu().contiguous()
    return bare


def export_sft_checkpoint_for_eval(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str | pathlib.Path,
    *,
    config_json: Mapping[str, Any],
    norm_stats_dir: str | pathlib.Path | None = None,
    norm_stats: Mapping[str, Any] | None = None,
) -> pathlib.Path:
    """Write an SFT wrapper state dict to a Phase-1 eval checkpoint directory.

    Args:
        state_dict: The :class:`OpenPiPytorchActionModel` wrapper state dict
            (or an already-bare ``Pi0`` state dict).
        output_dir: Destination directory for the eval-format checkpoint.
        config_json: Model config written to ``config.json`` (e.g.
            ``action_horizon``, ``action_dim``, ``paligemma_variant``,
            ``action_expert_variant``).
        norm_stats_dir: Directory containing ``norm_stats.json`` to copy into the
            ``physical-intelligence/behavior`` asset tree.
        norm_stats: Inline norm-stats mapping to write when ``norm_stats_dir`` is
            not given. Exactly one of ``norm_stats_dir`` / ``norm_stats`` is
            required.

    Returns:
        The output directory path.
    """
    import safetensors.torch

    if (norm_stats_dir is None) == (norm_stats is None):
        raise ValueError(
            "Provide exactly one of norm_stats_dir or norm_stats to export the "
            "BEHAVIOR norm-stats asset tree."
        )

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bare_state = _strip_wrapper_prefix(state_dict)
    safetensors.torch.save_file(bare_state, str(output_dir / "model.safetensors"))
    (output_dir / "config.json").write_text(json.dumps(dict(config_json), indent=2))

    stats_dir = output_dir / _NORM_STATS_SUBDIR
    stats_dir.mkdir(parents=True, exist_ok=True)
    if norm_stats_dir is not None:
        src = pathlib.Path(norm_stats_dir) / "norm_stats.json"
        if not src.is_file():
            raise FileNotFoundError(f"norm_stats.json not found at {src}.")
        shutil.copyfile(src, stats_dir / "norm_stats.json")
    else:
        (stats_dir / "norm_stats.json").write_text(json.dumps(dict(norm_stats)))

    logger.info(
        "Exported SFT checkpoint to eval format at %s (%d tensors).",
        output_dir,
        len(bare_state),
    )
    return output_dir
