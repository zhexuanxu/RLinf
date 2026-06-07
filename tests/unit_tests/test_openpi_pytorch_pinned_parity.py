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

"""Pinned first-N training-step parity gate (AC-5).

The real 8-GPU FSDP `train_vla_sft.py`, replaying the reference rank-0-fanout
first-N pinned batches + shared noise/time under the current canonical config,
reproduces the reference model's per-step loss within `|Δ|≤0.03`. This validates
the committed evidence (per-step deltas), which is the gate the GPU run produced.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EVIDENCE = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase4_pinned_first50_parity.json"
)


def _load() -> dict:
    if not _EVIDENCE.is_file():
        pytest.skip("pinned-parity evidence not present")
    return json.loads(_EVIDENCE.read_text())


def test_pinned_first_n_parity_within_gate():
    """Every step of the pinned first-N FSDP run is within |Δ|≤0.03 of the
    reference model on identical inputs (re-confirming R37's 50/50)."""
    ev = _load()
    assert ev["gate_met"], f"not all steps within the gate: {ev['within_0.03_count']}"
    assert ev["max_abs_delta"] <= ev["gate"]
    assert all(step["within_0.03"] == 1 for step in ev["steps"])
    within, total = ev["within_0.03_count"].split("/")
    assert within == total
    assert int(total) >= 10  # a first-N re-confirmation, N >= 10


def test_pinned_parity_max_delta_recomputes_from_steps():
    """The reported max |Δ| recomputes from the per-step rlinf/ref losses."""
    ev = _load()
    max_abs = max(
        abs(step["rlinf_pinned_loss"] - step["ref_pinned_loss"]) for step in ev["steps"]
    )
    assert max_abs == pytest.approx(ev["max_abs_delta"], abs=1e-6)
