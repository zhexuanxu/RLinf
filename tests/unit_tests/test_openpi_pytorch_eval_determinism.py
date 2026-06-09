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

"""Deterministic-eval hook: the basis for a paired evaluation.

By default the flow-matching sampler draws fresh noise, so the eval is stochastic
(unpaired). ``predict_action_batch`` accepts an optional injected ``noise`` (or a
seeded ``rng``) that makes the same observation yield the same actions — so two
models fed the same per-episode noise see paired conditions. This test verifies
that hook on the real ``get_model`` eval model; the production default (no
injection) is unchanged and covered by ``test_predict_action_batch_contract``.
"""

from __future__ import annotations

import pathlib

import pytest

_NEW_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")


def _eval_model(torch):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": str(_NEW_CKPT),
            "num_action_chunks": 32,
            "action_dim": 23,
            "num_steps": 5,
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "assets_dir": str(_NEW_CKPT),
                "asset_id": "physical-intelligence/behavior",
            },
        }
    )
    return get_model(cfg).to("cuda").eval()


def _fixed_env_obs(torch):
    g = torch.Generator().manual_seed(0)
    return {
        "main_images": torch.randint(
            0, 256, (1, 720, 720, 3), dtype=torch.uint8, generator=g
        ),
        "wrist_images": torch.randint(
            0, 256, (1, 2, 480, 480, 3), dtype=torch.uint8, generator=g
        ),
        "states": torch.rand(1, 256, generator=g).double(),
        "task_descriptions": ["turn on radio"],
        "extra_view_images": None,
    }


@pytest.mark.skipif(
    not (_NEW_CKPT / "model.safetensors").exists(),
    reason="converted BEHAVIOR checkpoint not available",
)
def test_injected_noise_makes_eval_deterministic():
    """Same observation + same injected noise -> identical actions (the property a
    paired eval relies on); a freshly-seeded rng reproduces actions too."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    model = _eval_model(torch)
    env_obs = _fixed_env_obs(torch)
    noise = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(321)).to(
        "cuda"
    )

    a1, _ = model.predict_action_batch(env_obs, mode="eval", noise=noise.clone())
    a2, _ = model.predict_action_batch(env_obs, mode="eval", noise=noise.clone())
    assert torch.equal(a1, a2), "injected-noise eval is not reproducible"

    b1, _ = model.predict_action_batch(
        env_obs, mode="eval", rng=torch.Generator(device="cuda").manual_seed(7)
    )
    b2, _ = model.predict_action_batch(
        env_obs, mode="eval", rng=torch.Generator(device="cuda").manual_seed(7)
    )
    assert torch.equal(b1, b2), "seeded-rng eval is not reproducible"
