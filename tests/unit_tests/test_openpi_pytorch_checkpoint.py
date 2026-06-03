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

"""Regression tests for the old->new checkpoint converter.

The fast test guards the two import-time fixes that make the old BEHAVIOR
checkpoint load into the new ``Pi0``: (1) the shared token embedder must come
from PaliGemma's wide ``lm_head`` (not the narrower action-expert head), and
(2) the SigLIP position embedding gains a leading broadcast dimension.

The heavyweight test (skipped unless the real checkpoint is present) asserts the
converted state dict matches a ``meta``-constructed BEHAVIOR ``Pi0`` exactly.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

from rlinf.models.embodiment.openpi_pytorch.utils import checkpoint_format as cf
from rlinf.models.embodiment.openpi_pytorch.utils.pi0_config import Pi0Config

_SIGLIP_OLD = "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
_OLD_CKPT = pathlib.Path(
    "/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999/model.safetensors"
)


def test_converter_embedder_and_posembed_fixes():
    old_sd = {
        "paligemma_with_expert.paligemma.lm_head.weight": torch.zeros(10, 2048),
        "paligemma_with_expert.gemma_expert.lm_head.weight": torch.zeros(10, 1024),
        _SIGLIP_OLD + "embeddings.position_embedding.weight": torch.zeros(256, 1152),
    }
    new_sd = cf.old_to_new_state_dict(old_sd)

    # Embedder must be PaliGemma's 2048-wide embedding, not the 1024-wide expert head.
    assert tuple(new_sd["llm.embedder.embedding.weight"].shape) == (10, 2048)
    # SigLIP position embedding gains the leading broadcast dim.
    assert tuple(new_sd["img.pos_embedding"].shape) == (1, 256, 1152)


def test_convert_checkpoint_overwrites_stale_assets(tmp_path):
    import json

    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.convert_checkpoint import (
        convert_checkpoint,
    )

    input_dir = tmp_path / "old"
    output_dir = tmp_path / "new"
    asset_dir = input_dir / "physical-intelligence" / "behavior"
    stale_asset_dir = output_dir / "physical-intelligence" / "behavior"
    asset_dir.mkdir(parents=True)
    stale_asset_dir.mkdir(parents=True)

    safetensors.torch.save_file(
        {
            "paligemma_with_expert.paligemma.lm_head.weight": torch.zeros(
                2, 2048, dtype=torch.bfloat16
            )
        },
        str(input_dir / "model.safetensors"),
    )
    (input_dir / "config.json").write_text(json.dumps({"action_dim": 32}))
    (asset_dir / "norm_stats.json").write_text(json.dumps({"fresh": True}))
    (stale_asset_dir / "norm_stats.json").write_text(json.dumps({"stale": True}))

    convert_checkpoint(input_dir, output_dir)

    copied = json.loads((stale_asset_dir / "norm_stats.json").read_text())
    assert copied == {"fresh": True}


def test_checkpoint_validation_rejects_dtype_mismatch():
    from rlinf.models.embodiment.openpi_pytorch import (
        _validate_checkpoint_state_dict,
    )

    model_state = {"w": torch.empty(2, 3)}
    checkpoint_state = {"w": torch.empty(2, 3, dtype=torch.float32)}

    with pytest.raises(ValueError, match="dtype mismatch"):
        _validate_checkpoint_state_dict(
            checkpoint_state,
            model_state,
            expected_dtype=torch.bfloat16,
        )


@pytest.mark.skipif(
    not _OLD_CKPT.exists(), reason="real BEHAVIOR checkpoint not available"
)
def test_converted_checkpoint_matches_behavior_model_exactly():
    import safetensors.torch

    cfg = Pi0Config(
        pi05=True, action_horizon=32, action_dim=32, dtype="float32", pcd=False
    )
    with torch.device("meta"):
        model_keys = cfg.create().state_dict()

    old_sd = safetensors.torch.load_file(str(_OLD_CKPT), device="cpu")
    new_sd = cf.old_to_new_state_dict(old_sd)

    assert set(new_sd) == set(model_keys), "converted key set must match the model"
    mism = {
        k: (tuple(model_keys[k].shape), tuple(new_sd[k].shape))
        for k in model_keys
        if tuple(model_keys[k].shape) != tuple(new_sd[k].shape)
    }
    assert not mism, f"shape mismatches: {mism}"
