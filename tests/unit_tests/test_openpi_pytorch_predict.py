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

"""End-to-end contract test for the BEHAVIOR pi05 action model (AC-4, AC-10, AC-11).

Builds the model from the converted checkpoint via ``get_model`` and runs
``predict_action_batch`` on a synthetic BEHAVIOR-shaped observation, asserting
the drop-in return contract ``[B, action_chunk, action_env_dim]``. Also checks
the reserved SFT methods raise. GPU + checkpoint gated.
"""

from __future__ import annotations

import pathlib

import pytest

_NEW_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")


def test_reserved_sft_methods_raise():
    # This part needs no GPU/checkpoint: the reserved interface must fail loudly.
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )

    dummy = OpenPiPytorchActionModel.__new__(OpenPiPytorchActionModel)
    with pytest.raises(NotImplementedError):
        OpenPiPytorchActionModel.compute_loss(dummy)
    with pytest.raises(NotImplementedError):
        OpenPiPytorchActionModel.sft_forward(dummy)


def test_predict_action_batch_contract():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not (_NEW_CKPT / "model.safetensors").exists():
        pytest.skip("converted BEHAVIOR checkpoint not available")
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "num_action_chunks": 32,
            "action_dim": 23,
            "num_steps": 10,
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    model = get_model(cfg).to("cuda").eval()

    batch = 2
    env_obs = {
        "main_images": torch.randint(0, 256, (batch, 720, 720, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(
            0, 256, (batch, 2, 480, 480, 3), dtype=torch.uint8
        ),
        "states": torch.rand(batch, 256).double(),
        "task_descriptions": ["turn on radio"] * batch,
        "extra_view_images": None,
    }
    actions, result = model.predict_action_batch(env_obs, mode="eval")

    assert tuple(actions.shape) == (batch, 32, 23)
    assert torch.isfinite(actions).all()
    assert set(result) == {"prev_logprobs", "prev_values", "forward_inputs"}
    assert set(result["forward_inputs"]) == {"action", "model_action"}


def test_get_model_rejects_unsupported_paths():
    pytest.importorskip("omegaconf")
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    # full_pi05 must fail loudly (reserved out of Phase 1).
    cfg = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "openpi": {"config_name": "pi05_behavior", "full_pi05": True},
        }
    )
    with pytest.raises(ValueError):
        get_model(cfg)

    # Plain dict configs must not bypass dotted openpi.* guard lookup.
    with pytest.raises(ValueError, match="full_pi05"):
        get_model(
            {
                "model_path": str(_NEW_CKPT),
                "openpi": {"config_name": "pi05_behavior", "full_pi05": True},
            }
        )

    # Non-BEHAVIOR config must fail loudly.
    cfg2 = OmegaConf.create(
        {"model_path": str(_NEW_CKPT), "openpi": {"config_name": "pi05_libero"}}
    )
    with pytest.raises(ValueError):
        get_model(cfg2)

    # Standard RLinf top-level value-head flag must also fail loudly.
    cfg3 = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "add_value_head": True,
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    with pytest.raises(ValueError, match="add_value_head"):
        get_model(cfg3)

    # Explicit fp32 precision is unsupported in the eval factory because the
    # vendored model carries internal bf16 activation dtype state.
    cfg4 = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "precision": "fp32",
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    with pytest.raises(ValueError, match="precision"):
        get_model(cfg4)


def test_get_model_rejects_missing_checkpoint_files(tmp_path):
    pytest.importorskip("omegaconf")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    torch = pytest.importorskip("torch")
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "openpi": {"config_name": "pi05_behavior"},
        }
    )
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        get_model(cfg)

    safetensors_torch.save_file(
        {"placeholder": torch.zeros(1, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(FileNotFoundError, match="norm stats"):
        get_model(cfg)
