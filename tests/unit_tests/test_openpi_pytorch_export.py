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

"""Round-trip: an SFT-trained checkpoint exports into the eval directory format.

Builds a tiny (``dummy`` variant) wrapper, exports its state dict, and loads the
result through the unchanged Phase-1 eval ``get_model`` (strict, bf16) — proving
an SFT-trained model can be evaluated without touching the eval path.
"""

from __future__ import annotations

import pytest


def _norm_stats_json():
    return {
        "norm_stats": {
            key: {
                "mean": [0.0] * 32,
                "std": [1.0] * 32,
                "q01": [0.0] * 32,
                "q99": [1.0] * 32,
            }
            for key in ("state", "actions")
        }
    }


def test_sft_checkpoint_exports_to_eval_format(tmp_path):
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.utils.export_checkpoint import (
        export_sft_checkpoint_for_eval,
    )
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    cfg = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=32,
        pcd=False,
    )
    wrapper = OpenPiPytorchActionModel(
        cfg.create(),
        processor=None,
        num_steps=10,
        action_chunk=4,
        action_env_dim=23,
    )
    state_dict = wrapper.state_dict()
    assert all(k.startswith("model.") for k in state_dict)  # wrapper prefix present

    config_json = {
        "action_horizon": 4,
        "action_dim": 32,
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
    }
    out = tmp_path / "exported"
    export_sft_checkpoint_for_eval(
        state_dict, out, config_json=config_json, norm_stats=_norm_stats_json()
    )
    assert (out / "model.safetensors").is_file()
    assert (out / "config.json").is_file()
    assert (out / "physical-intelligence" / "behavior" / "norm_stats.json").is_file()

    # Round-trip through the unchanged eval loader (strict, bf16).
    eval_cfg = OmegaConf.create(
        {
            "model_path": str(out),
            "precision": "bf16",
            "num_action_chunks": 4,
            "action_dim": 23,
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    model = get_model(eval_cfg)
    assert model.processor is not None
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}


def test_sft_checkpoint_dir_exports_from_real_layout(tmp_path):
    # Export from the actual RLinf FSDP save layout
    # (checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt), then
    # round-trip through the unchanged eval loader.
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.utils.export_checkpoint import (
        export_sft_checkpoint_dir_for_eval,
    )
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    cfg = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=32,
        pcd=False,
    )
    wrapper = OpenPiPytorchActionModel(
        cfg.create(), processor=None, num_steps=10, action_chunk=4, action_env_dim=23
    )

    # Write the consolidated weights at the real saved-checkpoint path.
    ckpt_dir = tmp_path / "checkpoints" / "global_step_150"
    weights_path = ckpt_dir / "actor" / "model_state_dict" / "full_weights.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.state_dict(), str(weights_path))

    out = tmp_path / "exported"
    export_sft_checkpoint_dir_for_eval(
        ckpt_dir,
        out,
        config_json={
            "action_horizon": 4,
            "action_dim": 32,
            "paligemma_variant": "dummy",
            "action_expert_variant": "dummy",
        },
        norm_stats=_norm_stats_json(),
    )
    assert (out / "model.safetensors").is_file()

    eval_cfg = OmegaConf.create(
        {
            "model_path": str(out),
            "precision": "bf16",
            "num_action_chunks": 4,
            "action_dim": 23,
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    model = get_model(eval_cfg)
    assert model.processor is not None
    assert {param.dtype for param in model.parameters()} == {torch.bfloat16}


def test_export_dir_missing_weights_fails_loud(tmp_path):
    from rlinf.models.embodiment.openpi_pytorch.utils.export_checkpoint import (
        export_sft_checkpoint_dir_for_eval,
    )

    with pytest.raises(FileNotFoundError, match="full_weights.pt"):
        export_sft_checkpoint_dir_for_eval(
            tmp_path / "empty",
            tmp_path / "o",
            config_json={},
            norm_stats=_norm_stats_json(),
        )


def test_export_requires_exactly_one_norm_stats_source(tmp_path):
    from rlinf.models.embodiment.openpi_pytorch.utils.export_checkpoint import (
        export_sft_checkpoint_for_eval,
    )

    with pytest.raises(ValueError, match="norm_stats"):
        export_sft_checkpoint_for_eval({}, tmp_path / "o", config_json={})

    with pytest.raises(ValueError, match="norm_stats"):
        export_sft_checkpoint_for_eval(
            {},
            tmp_path / "o2",
            config_json={},
            norm_stats_dir=str(tmp_path),
            norm_stats=_norm_stats_json(),
        )
