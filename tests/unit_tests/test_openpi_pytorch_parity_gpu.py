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

"""Deterministic action-parity gate: new self-contained Pi0 vs the old path.

This is the primary correctness gate for the BEHAVIOR pi05 migration (AC-6). It
loads the new vendored ``Pi0`` (from the converted checkpoint) and the old
``openpi`` ``PI0Pytorch`` (from the original checkpoint), feeds both the *same*
fixed observation, the *same* injected flow-matching noise, and the *same*
number of denoising steps, then asserts the sampled actions match within a tight
fp32 tolerance — both over the full 32-dim model output and the env-relevant
first 23 dims.

Parity contract (task9 — pinned):
- batch 1; images: seed-0 uint8 (1,224,224,3) for base/left_wrist/right_wrist;
- state: seed-0 uniform[-1,1] (1,32); prompt "turn on radio";
- tokenizer: PaligemmaTokenizer(max_len=200) — MUST fit the full pi05 prompt;
  a too-short max_len truncates the "Action:" suffix and corrupts the first
  action-horizon steps (this was empirically confirmed).
- noise: torch.randn (1,32,32), generator seed 123; num_steps=10; dtype fp32.
- tolerance: max |Δ| over the env-relevant 23 dims <= 2e-2 (observed ~5e-3).

The test is skipped unless a CUDA device, the installed ``openpi`` package, and
both checkpoints are available, since it loads two ~3.35B-param models.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pytest

_OLD_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999")
_NEW_CKPT = pathlib.Path("/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew")
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_NUM_STEPS = 10
_FP32_TOL = 2e-2


def _raw_observation():
    np.random.seed(0)
    from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer

    images = {
        k: np.random.randint(0, 256, (1, 224, 224, 3), dtype=np.uint8) for k in _IMAGE_KEYS
    }
    masks = {k: np.ones((1,), dtype=bool) for k in _IMAGE_KEYS}
    state = np.random.uniform(-1.0, 1.0, (1, 32)).astype(np.float32)
    tokens, tmask = PaligemmaTokenizer(max_len=200).tokenize("turn on radio", state[0, :23])
    return images, masks, state, tokens[None].astype(np.int64), tmask[None].astype(bool)


def test_action_parity_new_vs_old_path():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    if not (_OLD_CKPT / "model.safetensors").exists() or not (
        _NEW_CKPT / "model.safetensors"
    ).exists():
        pytest.skip("BEHAVIOR checkpoints not available")
    pytest.importorskip("openpi")
    import safetensors.torch

    dev = "cuda"
    images, masks, state, tokp, tokm = _raw_observation()

    def td(x):
        return torch.from_numpy(x).to(dev)

    raw = {
        "image": {k: td(images[k]) for k in _IMAGE_KEYS},
        "image_mask": {k: td(masks[k]) for k in _IMAGE_KEYS},
        "state": td(state),
        "tokenized_prompt": td(tokp),
        "tokenized_prompt_mask": td(tokm),
    }
    noise = torch.randn(1, 32, 32, generator=torch.Generator().manual_seed(123)).to(dev)

    # New self-contained model from the converted checkpoint.
    from rlinf.models.embodiment.openpi_pytorch.utils import model as vmodel
    from rlinf.models.embodiment.openpi_pytorch.utils.pi0_config import Pi0Config

    new_model = Pi0Config(
        pi05=True, action_horizon=32, action_dim=32, dtype="float32", pcd=False
    ).create()
    new_model.load_state_dict(
        safetensors.torch.load_file(str(_NEW_CKPT / "model.safetensors"), device="cpu"),
        strict=True,
    )
    new_model = new_model.to(dev).eval().float()
    with torch.no_grad():
        new_actions = new_model.sample_actions(
            vmodel.Observation.from_dict(copy.deepcopy(raw)),
            num_steps=_NUM_STEPS,
            noise=noise.clone(),
        ).float()

    # Old path: installed openpi PI0Pytorch from the original checkpoint.
    from openpi.models import model as omodel
    from openpi.models.pi0_config import Pi0Config as OldPi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    old_model = PI0Pytorch(OldPi0Config(pi05=True, action_horizon=32))
    old_model.load_state_dict(
        safetensors.torch.load_file(str(_OLD_CKPT / "model.safetensors"), device="cpu"),
        strict=False,
    )
    old_model = old_model.to(dev).eval().float()
    with torch.no_grad():
        old_actions = old_model.sample_actions(
            dev,
            omodel.Observation.from_dict(copy.deepcopy(raw)),
            noise=noise.clone(),
            num_steps=_NUM_STEPS,
        ).float()

    diff = (new_actions - old_actions).abs()
    full_max = float(diff.max())
    env_max = float(diff[..., :23].max())
    assert env_max <= _FP32_TOL, (
        f"env-dim action parity exceeded tolerance: max|Δ|(23-dim)={env_max:.4g} "
        f"> {_FP32_TOL} (full 32-dim max={full_max:.4g})"
    )
