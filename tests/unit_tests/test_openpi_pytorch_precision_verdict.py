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

"""Per-surface effective-equivalence verdict gate (AC-2).

Validates the committed verdict artifact: every precision surface is PASS or a
documented BENIGN mechanism difference (with a justification note) on rank 0 AND a
sharded rank; there is NO unreconciled MISMATCH; and the verdict covers every
runtime ledger surface. The gate FAILS on any MISMATCH, an undocumented BENIGN, or
a surface absent from the verdict.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_VERDICT = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase5_precision_surface_verdict.json"
)
_LEDGER = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase5_precision_ledger_rlinf_rank0.json"
)


def _load():
    if not _VERDICT.is_file():
        pytest.skip("precision surface verdict not present")
    return json.loads(_VERDICT.read_text())


def test_verdict_all_pass_or_benign_no_mismatch():
    v = _load()
    assert v["summary"]["all_pass_or_benign"] is True
    assert all(r["verdict"] != "MISMATCH" for r in v["rows"])
    assert v["summary"]["counts"].get("PASS", 0) > 0


def test_every_benign_row_has_a_justification():
    for r in _load()["rows"]:
        if r["verdict"] == "BENIGN":
            assert r.get("note"), f"BENIGN surface {r['surface']} lacks a justification"


def test_verdict_covers_every_ledger_surface_both_ranks():
    """Every surface in the runtime ledger has a verdict row on BOTH ranks."""
    v = _load()
    surfaces = set(json.loads(_LEDGER.read_text())["surfaces"])
    for rank in (0, 1):
        covered = {r["surface"] for r in v["rows"] if r["rank"] == rank}
        missing = surfaces - covered
        assert not missing, f"rank {rank} verdict missing surfaces: {sorted(missing)}"
