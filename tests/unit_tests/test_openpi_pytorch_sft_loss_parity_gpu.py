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

"""AC-11 / DEC-1(a) fixed-batch forward-loss parity vs the reference MODEL.

The correct fixed-batch parity gate compares RLinf's flow-matching loss to the
reference model's loss on the SAME batch + SAME weights + SAME noise/time -- NOT to
the reference's logged training loss (which is computed on a different, easier early
frame set; comparing against it spuriously reports a divergence).

The reference subprocess dumper (``_ref_model_loss_dump.py``, run in the reference
py3.11 venv on a GPU) loads the reference Pi0 from ``pi05_base_pytorch_new``, builds
fixed ``turning_on_radio`` batches, and dumps the reference per-batch loss at
``train=False`` and at ``train=True, rng=None`` (the production deterministic-crop
path) together with the batches and the per-batch noise/time. This test feeds the
SAME batches + noise/time through the RLinf ``Pi0.compute_loss`` and asserts the
per-batch loss is within the DEC-1 tolerance (relative 5% or absolute 0.01).

Skip-gated when the GPU, the reference venv/src, the model, or the data are absent.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess

import numpy as np
import pytest

_REF_PY = "/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python"
_DUMP = pathlib.Path(__file__).parent / "_ref_model_loss_dump.py"
_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
_MODEL = "/mnt/public/xzxuan/models/pi05_base_pytorch_new"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_ASSET_ID = "behavior-1k/2025-challenge-demos"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_REF_STEP0_LOSS = 0.24609375
_DEC1_BAND = max(0.05 * _REF_STEP0_LOSS, 0.01)


def _have_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="module")
def ref_loss_dump(tmp_path_factory):
    if not _have_gpu():
        pytest.skip("CUDA not available")
    if not os.path.exists(_REF_PY) or not os.path.isdir(_REF_SRC):
        pytest.skip("reference py3.11 venv / src not available")
    if not os.path.isdir(_DATA) or not os.path.exists(f"{_MODEL}/model.safetensors"):
        pytest.skip("BEHAVIOR data / new-format base checkpoint not available")
    out = tmp_path_factory.mktemp("ref_model_loss")
    proc = subprocess.run(
        [_REF_PY, str(_DUMP), str(out)],
        capture_output=True,
        text=True,
        timeout=1800,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl"},
    )
    lines = [
        ln
        for ln in (proc.stdout + "\n" + proc.stderr).splitlines()
        if ln.startswith("REF_MODEL_LOSS ")
    ]
    if not lines:
        pytest.skip(
            f"reference model-loss dump produced no result; stderr tail: {proc.stderr[-300:]}"
        )
    result = json.loads(lines[-1].split("REF_MODEL_LOSS ", 1)[1])
    if not result.get("ok"):
        pytest.skip(f"reference model-loss dump failed: {result.get('err')}")
    return out


def _rlinf_per_batch_losses(out):
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    model_cfg = OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": _MODEL,
            "num_action_chunks": 32,
            "action_dim": 23,
            "precision": "bf16",
            "load_for_training": True,
            "openpi": {
                "assets_dir": _ASSETS_DIR,
                "asset_id": _ASSET_ID,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "num_images_in_input": 3,
                "action_chunk": 32,
                "action_env_dim": 23,
                "load_for_training": True,
            },
        }
    )
    model = get_model(model_cfg).to("cuda")
    model.train()
    inner = model.model

    d = np.load(out / "ref_model_batches.npz")
    loss = np.load(out / "ref_model_loss.npz")
    n = d["state"].shape[0]
    noaug, aug = [], []
    for bi in range(n):
        obs = Observation.from_dict(
            {
                "image": {
                    k: torch.from_numpy(d[f"image__{k}"][bi]).to("cuda", torch.float32)
                    for k in _IMG
                },
                "image_mask": {
                    k: torch.from_numpy(d[f"image_mask__{k}"][bi]).to("cuda")
                    for k in _IMG
                },
                "state": torch.from_numpy(d["state"][bi]).to("cuda", torch.float32),
                "tokenized_prompt": torch.from_numpy(d["tokenized_prompt"][bi])
                .to("cuda")
                .long(),
                "tokenized_prompt_mask": torch.from_numpy(
                    d["tokenized_prompt_mask"][bi]
                )
                .to("cuda")
                .bool(),
            }
        )
        actions = torch.from_numpy(d["actions"][bi]).to("cuda", torch.float32)
        noise = torch.from_numpy(loss["noise"][bi]).to("cuda")
        time = torch.from_numpy(loss["time"][bi]).to("cuda")
        with torch.no_grad():
            noaug.append(
                float(
                    inner.compute_loss(
                        obs, actions, train=False, noise=noise, time=time
                    )
                    .float()
                    .mean()
                )
            )
            aug.append(
                float(
                    inner.compute_loss(
                        obs, actions, train=True, rng=None, noise=noise, time=time
                    )
                    .float()
                    .mean()
                )
            )
    return (
        np.asarray(noaug),
        np.asarray(aug),
        loss["ref_loss_noaug"],
        loss["ref_loss_aug"],
    )


def test_fixed_batch_loss_parity_vs_reference_model(ref_loss_dump):
    """RLinf's flow-matching loss matches the reference model's on the SAME batch +
    weights + noise/time, within DEC-1 -- both at train=False and at the production
    train=True/rng=None path. This is the AC-11 fixed-batch parity gate."""
    rl_noaug, rl_aug, ref_noaug, ref_aug = _rlinf_per_batch_losses(ref_loss_dump)

    # Per-batch parity within the DEC-1 band (rel 5% or abs 0.01).
    assert np.abs(rl_noaug - ref_noaug).max() <= _DEC1_BAND, (
        f"train=False per-batch diff {np.abs(rl_noaug - ref_noaug).max():.5f} "
        f"exceeds DEC-1 band {_DEC1_BAND:.5f}: rlinf={rl_noaug}, ref={ref_noaug}"
    )
    assert np.abs(rl_aug - ref_aug).max() <= _DEC1_BAND, (
        f"train=True/rng=None per-batch diff {np.abs(rl_aug - ref_aug).max():.5f} "
        f"exceeds DEC-1 band {_DEC1_BAND:.5f}: rlinf={rl_aug}, ref={ref_aug}"
    )
    # The mean losses also agree well within tolerance.
    assert abs(rl_aug.mean() - ref_aug.mean()) <= _DEC1_BAND
