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

"""Config-resolution gate for the FSDP compute-dtype decouple.

The BEHAVIOR pi0.5 SFT FSDP compute ``param_dtype`` must be set EXPLICITLY (bf16),
NOT interpolated from the model precision/load selector. This is a dry-run config
resolution (no GPU): it proves that overriding the model precision selector to
``fp32`` leaves the FSDP compute ``param_dtype`` at ``bf16`` -- so changing the
load/precision field can never silently change the compute dtype. Negative test:
the resolved value must not be the literal interpolation of the precision field.
"""

from __future__ import annotations

import os
import pathlib

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
_CONFIG_DIR = _REPO / "examples/sft/config"


def _compose(overrides=None):
    pytest.importorskip("hydra")
    from hydra import compose, initialize_config_dir

    # The SFT config's Hydra searchpath uses ${oc.env:EMBODIED_PATH}.
    os.environ.setdefault("EMBODIED_PATH", str(_REPO / "examples/embodiment"))
    os.environ.setdefault("REPO_PATH", str(_REPO))
    if not _CONFIG_DIR.is_dir():
        pytest.skip("SFT config dir not present")
    with initialize_config_dir(version_base="1.1", config_dir=str(_CONFIG_DIR)):
        return compose(config_name="behavior_pi05_vla", overrides=overrides or [])


def _param_dtype(cfg):
    return cfg.actor.fsdp_config.mixed_precision.param_dtype


def test_compute_param_dtype_is_explicit_bf16():
    cfg = _compose()
    assert str(_param_dtype(cfg)) == "bf16"


def test_param_dtype_decoupled_from_precision_selector():
    """Overriding the model precision/load selector to fp32 must NOT change the FSDP
    compute param_dtype (it stays bf16) -- the decouple."""
    cfg = _compose(overrides=["actor.model.precision=fp32"])
    assert str(_param_dtype(cfg)) == "bf16", (
        "FSDP compute param_dtype changed when the precision selector was overridden "
        "-- the param_dtype/precision coupling was not removed."
    )
    # The fp32-master training load path is independent and unchanged.
    assert bool(cfg.actor.model.get("load_for_training", False)) is True


def test_param_dtype_is_not_an_interpolation_of_precision():
    """The raw config value must be a literal dtype, not `${actor.model.precision}`."""
    import yaml

    raw = (_CONFIG_DIR / "behavior_pi05_vla.yaml").read_text()
    doc = yaml.safe_load(raw)
    val = doc["actor"]["fsdp_config"]["mixed_precision"]["param_dtype"]
    assert val == "bf16"
    assert "actor.model.precision" not in str(val)
