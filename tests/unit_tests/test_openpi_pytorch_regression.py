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

"""Regression: the openpi_pytorch SFT additions don't break the old `openpi` path.

The Phase-2 changes are additive — a new `OPENPI_PYTORCH` dispatch branch in the
SFT worker, a relaxed config guard, and a new dataconfig package. These tests
prove the old `openpi` SFT path is unaffected: its config still validates and the
worker still routes `model_type: openpi` to the OpenPI dataloader branch (not the
new `OPENPI_PYTORCH` branch, and not the generic "unsupported model" KeyError).
The Phase-1 `openpi_pytorch` action-parity GPU test
(`test_openpi_pytorch_parity_gpu.py`) is the separate eval-path regression.
"""

from __future__ import annotations

import os
import types

import pytest
from omegaconf import OmegaConf

_DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
_OLD_MODEL = "/mnt/public/xzxuan/models/pi05_base_pytorch"


def test_old_openpi_sft_config_validation_unaffected():
    # `model_type: openpi` (the old model) must still pass SFT validation, and
    # the openpi_pytorch guard must NOT touch it (full_pi05 is valid for old openpi).
    from rlinf.config import validate_sft_cfg

    cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 256,
                "micro_batch_size": 32,
                "model": {
                    "model_type": "openpi",
                    "openpi": {"config_name": "pi05_libero", "full_pi05": True},
                },
            },
            "data": {"train_data_paths": "/tmp/data"},
            "runner": {},
        }
    )
    validate_sft_cfg(cfg)  # must not raise


def test_old_openpi_sft_dispatch_still_routes_to_openpi_branch():
    # For `model_type: openpi`, build_dataloader must reach the OpenPI branch
    # (which fails loudly on missing data) — proving the new OPENPI_PYTORCH branch
    # does not shadow it and the old path is not a KeyError fall-through.
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi",
                    "openpi": {"full_pi05": False, "forward_mode": "vla"},
                }
            }
        }
    )
    stub = types.SimpleNamespace(cfg=cfg)
    stub._is_pi05_vlm_only = FSDPVlaSftWorker._is_pi05_vlm_only.__get__(stub)

    with pytest.raises(ValueError, match="OpenPI SFT requires data.train_data_paths"):
        FSDPVlaSftWorker.build_dataloader(stub, data_paths=None)


def test_old_openpi_behavior_sft_dataloader_smoke():
    # Actually exercise the old `openpi` BEHAVIOR SFT data path (installed openpi,
    # not the new self-contained one): build the old loader and pull one batch.
    # Proves the Phase-2 additions did not break old-openpi SFT. Skip-gated.
    if not os.path.isdir(_DATA):
        pytest.skip("BEHAVIOR data not available")
    if not os.path.isdir(_OLD_MODEL):
        pytest.skip("old openpi model/norm-stats not available")
    pytest.importorskip("openpi")

    import dataclasses

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.dataconfig.behavior_data_loader import (
        create_behavior_data_loader,
    )

    # The config already points at the local BEHAVIOR data + its assets
    # (pi05_base_pytorch/physical-intelligence/behavior); don't override repo_id
    # (that would rewrite asset_id and break norm-stats resolution). Force
    # num_workers=0 so the batch is built in-process (avoids spawn/pickling
    # quirks under pytest), and use shuffle=True — the setting the production SFT
    # worker uses (rlinf/workers/sft/fsdp_vla_sft_worker.py: shuffle=True).
    config = get_openpi_config("pi05_behavior_b1k_local", batch_size=2)
    config = dataclasses.replace(config, num_workers=0)
    loader = create_behavior_data_loader(config, shuffle=True, seed=42)
    assert loader is not None

    import numpy as np

    # Pull one real batch from the old BEHAVIOR SFT path and assert its contract.
    observation, actions = next(iter(loader))
    actions = np.asarray(actions)
    state = np.asarray(observation.state)
    assert observation is not None
    assert actions.shape[1:] == (32, 32)  # [B, action_horizon, action_dim]
    assert state.shape[-1] == 32
    assert np.isfinite(actions).all()


def test_unsupported_model_type_still_raises_keyerror():
    # A genuinely unsupported model type must still hit the explicit KeyError, so
    # the new branch did not turn unknown types into silent failures.
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    cfg = OmegaConf.create({"actor": {"model": {"model_type": "openvla"}}})
    stub = types.SimpleNamespace(cfg=cfg)
    stub._is_pi05_vlm_only = FSDPVlaSftWorker._is_pi05_vlm_only.__get__(stub)

    with pytest.raises(KeyError, match="not support such model type"):
        FSDPVlaSftWorker.build_dataloader(stub, data_paths=None)
