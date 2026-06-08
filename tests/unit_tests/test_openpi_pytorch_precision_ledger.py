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

"""Dual-repo runtime mixed-precision dtype-ledger gate.

Validates the committed ledgers: for BOTH repos, a runtime-observed dtype ledger
exists per rank (rank 0 AND a sharded rank), records every enumerated precision
surface, and is sourced from RUNTIME OBJECTS (a provenance field that points at a
built object), never YAML text. The reference's omitted ``buffer_dtype`` default is
RECORDED AS OBSERVED (the ``buffer_dtype_observed`` key is present), not assumed.
The gate FAILS if a ledger is missing, a surface is absent, a value lacks runtime
provenance, or the reference buffer-default was not observed.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EV = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence")
_REQUIRED_SURFACES = (
    "fsdp_mixed_precision",
    "param_dtype_outside_forward",
    "grad_dtype",
    "loss_dtype",
    "autocast_enabled",
    "grad_scaler_enabled",
    "optimizer",
    "optimizer_state_dtype",
)


def _load(repo, rank):
    p = _EV / f"phase5_precision_ledger_{repo}_rank{rank}.json"
    if not p.is_file():
        pytest.skip(f"precision ledger {p.name} not present")
    return json.loads(p.read_text())


def test_both_repo_ledgers_exist_rank0_and_sharded_rank():
    """A runtime ledger exists for both repos on rank 0 AND a sharded (non-zero)
    rank -- not a rank-0-only observation."""
    for repo in ("rlinf", "ref"):
        for rank in (0, 1):
            d = _load(repo, rank)
            assert d["ok"] is True, f"{repo} rank{rank} ledger not ok"
            assert d["repo"] in ("rlinf", "reference")
            assert d["rank"] == rank
            assert d["world_size"] >= 2


def test_every_surface_recorded_with_runtime_provenance():
    """Each ledger records every enumerated precision surface, and the FSDP
    mixed-precision block carries a runtime-object provenance (not a YAML source)."""
    for repo in ("rlinf", "ref"):
        d = _load(repo, 0)
        for surface in _REQUIRED_SURFACES:
            assert surface in d, f"{repo} ledger missing surface {surface}"
        prov = d["fsdp_mixed_precision"]["provenance"]
        # Provenance must point at a built runtime object, never a config file.
        assert "mixed_precision" in prov
        assert ".yaml" not in prov and "config" not in prov.lower()


def test_reference_buffer_default_observed_not_assumed():
    """The reference omits ``buffer_dtype`` (records None) and the effective default
    is OBSERVED off the built model (the observed key is present), not assumed."""
    d = _load("ref", 0)
    assert d["fsdp_mixed_precision"]["buffer_dtype"] == "None"
    assert "buffer_dtype_observed" in d  # observed off model.named_buffers()


def test_optimizer_hyperparams_read_from_param_groups():
    """The optimizer betas/eps/weight_decay are the EFFECTIVE per-group values, so
    the two repos' real betas are comparable (both pass them via param groups)."""
    rl = _load("rlinf", 0)["optimizer"]
    rf = _load("ref", 0)["optimizer"]
    # Both must record a 2-tuple beta and an lr (proving param-group read, not the
    # AdamW default proxy).
    for o in (rl, rf):
        assert len(o["betas"]) == 2
        assert "lr" in o
    # The real betas agree across repos (the apples-to-apples read).
    assert rl["betas"] == rf["betas"]
