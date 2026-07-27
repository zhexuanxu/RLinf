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

from __future__ import annotations

from typing import Any, Sequence

from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
    PaligemmaTokenizer,
    TokenizeSubtaskPrompt,
)


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
    *,
    norm_stats_dir: str | None = None,
    norm_stats_asset_id: str | None = None,
    mode: str = "vla",
    state_token: str | None = None,
    action_env_dim: int | None = None,
    max_token_len: int | None = None,
) -> tuple[Sequence, Sequence]:
    """Build ``(input_transforms, output_transforms)`` for ``config_name``.

    Returns two lists ready for :func:`openpi.transforms.compose`, matching
    ``rlinf/models/embodiment/openpi/__init__.py`` exactly:

    * input:  ``[InjectDefaultPrompt(None), *data.inputs, Normalize, *model.inputs]``
    * output: ``[*model.outputs, Unnormalize, *data.outputs]``

    Norm stats resolve from ``{norm_stats_dir}/{asset_id}/norm_stats.json`` when
    ``norm_stats_dir`` is given, else from the checkpoint dir via
    ``data_config.asset_id`` (``{model_path}/{asset_id}/norm_stats.json`` — the
    same canonical stats the original openpi path resolves). Eval / RL leave
    ``norm_stats_dir`` unset (their checkpoint bundles the stats); the BEHAVIOR
    SFT loader passes the experiment's ``assets_dir`` + ``asset_id`` so it reads
    the exact same ``norm_stats.json`` the old SFT path did (the SFT *base*
    checkpoint bundles no stats).

    In ``vlm_vla`` mode only the upstream ``TokenizePrompt`` stage is replaced
    with :class:`TokenizeSubtaskPrompt`. Images, normalization, resize, action
    padding, and output unnormalization remain the same shared OpenPI pipeline.
    ``state_token: none`` is the public state-free selector; any concrete state
    layout enables state injection after normalization.
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name,
        model_path=str(model_path),
        data_kwargs=data_kwargs,
        state_token=state_token,
        action_env_dim=action_env_dim,
        max_token_len=max_token_len,
    )
    upstream_model_config = train_config.model

    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )

    asset_id = norm_stats_asset_id or data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is required to load norm_stats.")
    stats_dir = (
        norm_stats_dir
        if norm_stats_dir is not None
        else download.maybe_download(str(model_path))
    )
    norm_stats = _checkpoints.load_norm_stats(stats_dir, asset_id)
    if norm_stats is None:
        raise FileNotFoundError(
            f"openpi_pytorch: norm_stats not found at {stats_dir}/{asset_id}/"
            "norm_stats.json. For eval/RL the checkpoint dir must bundle them; "
            "for SFT set actor.model.openpi.assets_dir/asset_id to the stats dir."
        )

    tokenizer = PaligemmaTokenizer(max_len=int(upstream_model_config.max_token_len))
    inject_state = bool(upstream_model_config.discrete_state_input)
    model_input_transforms = [
        (
            TokenizeSubtaskPrompt(tokenizer, inject_state=inject_state)
            if mode == "vlm_vla"
            else transforms.TokenizePrompt(
                tokenizer,
                discrete_state_input=inject_state,
            )
        )
        if isinstance(transform, transforms.TokenizePrompt)
        else transform
        for transform in data_config.model_transforms.inputs
    ]

    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *model_input_transforms,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return input_transforms, output_transforms


def find_subtask_tokenizer(
    input_transforms: Sequence,
) -> PaligemmaTokenizer | None:
    """Return the VLM tokenizer installed in a shared transform sequence."""
    for transform in input_transforms:
        if isinstance(transform, TokenizeSubtaskPrompt):
            return transform.tokenizer
    return None
