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

"""End-to-end contract test for the BEHAVIOR pi05 action model.

Builds the model from the converted checkpoint via ``get_model`` and runs
``predict_action_batch`` on a synthetic BEHAVIOR-shaped observation, asserting
the drop-in return contract ``[B, action_chunk, action_env_dim]``. Also exercises
the SFT loss path on CPU with a lightweight fake core. GPU + checkpoint gated.
"""

from __future__ import annotations

import pathlib

import pytest

_NEW_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")


def test_sft_forward_cpu_dummy():
    """sft_forward wires the SFT loss path; verified on CPU with a fake core.

    Uses a lightweight stand-in for the vendored Pi0 so the loss-path contract
    (tuple/dict batch, device move, scalar reduction, malformed-batch errors,
    gradient-checkpointing pass-through) is exercised without a GPU/checkpoint.
    """
    torch = pytest.importorskip("torch")
    import numpy as np
    import torch.nn as nn

    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import NormStats
    from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
        OpenPiPytorchActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    class _FakeCore(nn.Module):
        action_dim = 32

        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))
            self.gc = False
            self.last_actions = None
            self.last_train = None
            self.last_noise = None
            self.last_time = None

        def compute_loss(
            self, observation, actions, *, train=False, rng=None, noise=None, time=None
        ):
            # (B, action_horizon) per-timestep loss; depends on a param so the
            # reduced scalar is differentiable. Mirrors Pi0.compute_loss, which
            # accepts optional pinned noise/time (None in normal training).
            self.last_actions = actions
            self.last_train = train
            self.last_noise = noise
            self.last_time = time
            return (actions.float() ** 2).mean(dim=-1) + self.dummy

        def gradient_checkpointing_enable(self):
            self.gc = True

        def gradient_checkpointing_disable(self):
            self.gc = False

    class _FakeProcessor:
        action_stats = NormStats(
            mean=np.zeros(32),
            std=np.ones(32),
            q01=np.zeros(32),
            q99=np.ones(32),
        )

    model = OpenPiPytorchActionModel(
        _FakeCore(),
        processor=None,
        num_steps=10,
        action_chunk=32,
        action_env_dim=23,
    )
    obs = Observation(images={}, image_masks={}, state=torch.zeros(2, 32))
    actions = torch.randn(2, 32, 32)

    # Tuple form -> finite differentiable scalar.
    loss = model.sft_forward((obs, actions))
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert model.model.last_train is True
    loss.backward()

    # SFT loss semantics remain training-mode even if the wrapper is in eval mode.
    model.eval()
    loss_eval_wrapper = model.sft_forward((obs, actions))
    assert loss_eval_wrapper.ndim == 0 and torch.isfinite(loss_eval_wrapper)
    assert model.model.last_train is True

    # Dict form via forward(ForwardType.SFT).
    loss2 = model(
        forward_type=ForwardType.SFT,
        data={"observation": obs, "actions": actions},
    )
    assert loss2.ndim == 0 and torch.isfinite(loss2)

    # Gradient-checkpointing pass-through reaches the inner core.
    model.gradient_checkpointing_enable()
    assert model.model.gc is True
    model.gradient_checkpointing_disable()
    assert model.model.gc is False

    # Malformed batches fail loudly.
    with pytest.raises(ValueError):
        model.sft_forward((obs,))
    with pytest.raises(ValueError):
        model.sft_forward({"observation": obs})
    with pytest.raises(ValueError):  # no processor means no action stats
        model.sft_forward((obs, torch.randn(2, 32, 23)))
    with pytest.raises(TypeError):
        model.sft_forward(42)

    # Env-dim raw actions normalize before zero-padding when action stats exist.
    model_with_stats = OpenPiPytorchActionModel(
        _FakeCore(),
        processor=_FakeProcessor(),
        num_steps=10,
        action_chunk=32,
        action_env_dim=23,
    )
    raw_actions = torch.ones(2, 32, 23)
    loss3 = model_with_stats.sft_forward((obs, raw_actions))
    assert loss3.ndim == 0 and torch.isfinite(loss3)
    assert tuple(model_with_stats.model.last_actions.shape) == (2, 32, 32)
    torch.testing.assert_close(
        model_with_stats.model.last_actions[..., :23], torch.ones(2, 32, 23)
    )
    torch.testing.assert_close(
        model_with_stats.model.last_actions[..., 23:], torch.zeros(2, 32, 9)
    )


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
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "assets_dir": str(_NEW_CKPT),
                "asset_id": "physical-intelligence/behavior",
            },
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

    # These unsupported-path guards fire before any model-shape resolution, so a
    # minimal config (no config_name — removed per DEC-2) still triggers them.
    # full_pi05 must fail loudly (unsupported by the eval-only model).
    cfg = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "openpi": {"full_pi05": True},
        }
    )
    with pytest.raises(ValueError):
        get_model(cfg)

    # Plain dict configs must not bypass dotted openpi.* guard lookup.
    with pytest.raises(ValueError, match="full_pi05"):
        get_model(
            {
                "model_path": str(_NEW_CKPT),
                "openpi": {"full_pi05": True},
            }
        )

    # Standard RLinf top-level value-head flag must also fail loudly.
    cfg3 = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "add_value_head": True,
            "openpi": {},
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
            "openpi": {},
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
            "openpi": {},
        }
    )
    # No model.safetensors -> checkpoint-not-found (before any shape resolution).
    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        get_model(cfg)

    # Weights present but no YAML model-shape fields -> loud missing-shape error
    # (model shape is built from YAML, not a checkpoint config.json — AC-5).
    safetensors_torch.save_file(
        {"placeholder": torch.zeros(1, dtype=torch.bfloat16)},
        str(tmp_path / "model.safetensors"),
    )
    with pytest.raises(ValueError, match="num_action_chunks"):
        get_model(cfg)
