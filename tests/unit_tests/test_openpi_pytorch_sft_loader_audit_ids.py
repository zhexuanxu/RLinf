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

"""STRICT gate for the value-independent (episode:frame) loader multiset audit (AC-2).

Pure-logic tests prove the comparator. Committed-evidence tests enforce the real AC-2
contract: both id audits cover ALL 30000 global steps with 256 distinct ids/step and the
production num_workers (8); the verdict is CONCLUSIVE (IDENTICAL or DIVERGENT — INCONCLUSIVE
FAILS); the rolling hashes agree iff the verdict is IDENTICAL; a divergence names the first
step + concrete differing ids; and a non-perturbation re-run reproduces the rolling hash
byte-for-byte.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))

from tools.sft_loader_audit_compare_ids import compare_id_audits  # noqa: E402

_EV = _REPO / "docs/evidence"
_RLINF = _EV / "phase6_loader_audit_ids_rlinf.json"
_REF = _EV / "phase6_loader_audit_ids_ref.json"
_COMPARE = _EV / "phase6_loader_audit_ids_compare.json"
_NONPERTURB = _EV / "phase6_loader_audit_ids_nonperturb.json"

_EXPECTED_STEPS = 30000


def _dump(hashes, **extra):
    d = {"ok": True, "per_step_global_set_hash": hashes,
         "per_step_unique": [256] * len(hashes), "audited_steps": len(hashes),
         "num_workers": 8, "rolling_hash": "r"}
    d.update(extra)
    return d


def _detail(sets):
    return {"per_step_global_set": sets}


def test_compare_identical_when_all_step_hashes_match():
    out = compare_id_audits(_dump(["a", "b"]), _dump(["a", "b"]),
                            _detail([["x"], ["y"]]), _detail([["x"], ["y"]]))
    assert out["verdict"] == "IDENTICAL_MULTISET"
    assert out["frame_identity_comparable"] is True and out["first_mismatch"] is None


def test_compare_divergent_reports_first_step_and_ids():
    out = compare_id_audits(
        _dump(["a", "b"]), _dump(["a", "c"]),
        _detail([["x", "y"], ["p", "q"]]), _detail([["x", "y"], ["p", "z"]]),
    )
    assert out["verdict"] == "DIVERGENT_MULTISET"
    fm = out["first_mismatch"]
    assert fm["step"] == 1 and "q" in fm["rlinf_only_sample"] and "z" in fm["ref_only_sample"]


def test_compare_flags_non_comparable_id_space_as_inconclusive():
    out = compare_id_audits(_dump(["a"]), _dump(["b"]),
                            _detail([["x1"]]), _detail([["y1"]]))  # disjoint id spaces
    assert out["verdict"] == "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE"
    assert out["frame_identity_comparable"] is False


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (id audit not yet run/committed)")
    return json.loads(p.read_text())


def test_committed_id_audits_cover_full_30k_with_256_per_step():
    for p in (_RLINF, _REF):
        d = _load(p)
        assert d["ok"] is True
        assert d["audited_steps"] == _EXPECTED_STEPS, f"{p.name} steps {d['audited_steps']}"
        assert d["num_workers"] == 8, f"{p.name} num_workers {d['num_workers']} (null/non-prod)"
        uniq = d["per_step_unique"]
        assert len(uniq) == _EXPECTED_STEPS
        assert min(uniq) == 256 and max(uniq) == 256


def test_committed_compare_is_conclusive_and_consistent():
    d = _load(_COMPARE)
    assert d["verdict"] in ("IDENTICAL_MULTISET", "DIVERGENT_MULTISET"), (
        f"INCONCLUSIVE/other not allowed for the final AC-2 gate: {d['verdict']}"
    )
    assert d["frame_identity_comparable"] is True
    assert (d["cross_dump_distinct_overlap"] or 0) > 0
    assert d["audited_steps"] == _EXPECTED_STEPS
    # rolling hashes agree IFF the multisets are identical at every step
    rl_roll = d["rlinf"]["rolling_hash"]
    rf_roll = d["ref"]["rolling_hash"]
    if d["verdict"] == "IDENTICAL_MULTISET":
        assert rl_roll == rf_roll
        assert d["first_mismatch"] is None
    else:
        assert rl_roll != rf_roll
        assert d["first_mismatch"] and "step" in d["first_mismatch"]


def test_committed_nonperturbation_rolling_hash_is_reproducible():
    """Two captures with the same config/seed reproduce the rolling hash byte-for-byte
    (the audit is deterministic; nondeterminism would invalidate the comparison)."""
    d = _load(_NONPERTURB)
    assert d["rlinf_run1_rolling"] == d["rlinf_run2_rolling"]
    assert d["audited_steps"] >= 2
