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

"""Gate for the reference-vs-RLinf loader per-global-step multiset audit.

Pure-logic tests prove the comparator: it declares IDENTICAL only when EVERY audited
step's global frame set matches, and on a divergence it reports the FIRST mismatching
step plus a sample of the differing ids. Committed-evidence tests (skip-gated until the
data-only audit has run) validate the committed comparison + the two audit dumps.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))

from tools.sft_loader_audit_compare import compare_audits  # noqa: E402

_EV = _REPO / "docs/evidence"
_COMPARE = _EV / "phase6_loader_audit_compare.json"
_RLINF = _EV / "phase6_loader_audit_rlinf.json"
_REF = _EV / "phase6_loader_audit_ref.json"


def _audit(per_rank_per_step, **extra):
    d = {"ok": True, "per_rank_per_step_frame_hashes": per_rank_per_step}
    d.update(extra)
    return d


def test_compare_declares_identical_when_every_step_matches():
    steps = [[[f"r{r}f{i}" for i in range(4)] for r in range(2)] for _ in range(3)]
    out = compare_audits(_audit(steps), _audit([list(s) for s in steps]))
    assert out["verdict"] == "IDENTICAL_MULTISET"
    assert out["all_steps_identical"] is True
    assert out["first_mismatch"] is None
    assert out["steps_matched"] == out["audited_steps"] == 3
    assert out["mean_jaccard_overlap"] == 1.0


def test_compare_reports_first_divergent_step_and_ids():
    rl = [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "h"]]]
    rf = [[["a", "b"], ["c", "d"]], [["e", "f"], ["g", "X"]]]  # step 1 differs (h vs X)
    out = compare_audits(_audit(rl), _audit(rf))
    assert out["verdict"] == "DIVERGENT_MULTISET"
    assert out["all_steps_identical"] is False
    fm = out["first_mismatch"]
    assert fm["step"] == 1
    assert "h" in fm["rlinf_only_sample"] and "X" in fm["ref_only_sample"]
    assert out["steps_matched"] == 1
    assert 0.0 < out["mean_jaccard_overlap"] < 1.0


def test_compare_step0_divergence_is_caught():
    rl = [[["a", "b", "c", "d"]]]
    rf = [[["a", "b", "c", "z"]]]
    out = compare_audits(_audit(rl), _audit(rf))
    assert out["verdict"] == "DIVERGENT_MULTISET"
    assert out["first_mismatch"]["step"] == 0


def test_compare_flags_non_byte_comparable_identity_as_inconclusive():
    """If the two dumps share NO frame ids at all (e.g. the same underlying frame hashes
    differently across repos due to float-level normalization differences), the per-step
    multiset cannot be judged via this identity — the verdict must be INCONCLUSIVE, not a
    false DIVERGENT."""
    rl = [[["a1", "a2"], ["a3", "a4"]]]
    rf = [[["b1", "b2"], ["b3", "b4"]]]  # disjoint id spaces -> cross overlap 0
    out = compare_audits(_audit(rl), _audit(rf))
    assert out["verdict"] == "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE"
    assert out["frame_identity_comparable"] is False
    assert out["cross_dump_distinct_overlap"] == 0


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (loader audit not yet run/committed)")
    return json.loads(p.read_text())


def test_committed_audit_dumps_are_ok_and_rank_disjoint_256():
    for p in (_RLINF, _REF):
        d = _load(p)
        assert d["ok"] is True
        # every audited global step must have the full effective batch distinct
        steps = d["per_rank_per_step_frame_hashes"]
        for s, step in enumerate(steps):
            union = {f for rh in step for f in rh}
            assert len(union) == 256, f"{p.name} step {s} distinct={len(union)} != 256"


def test_committed_compare_reports_a_verdict_with_evidence():
    d = _load(_COMPARE)
    assert d["audited_steps"] >= 1
    assert d["verdict"] in (
        "IDENTICAL_MULTISET",
        "DIVERGENT_MULTISET",
        "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE",
    )
    if d["verdict"] == "DIVERGENT_MULTISET":
        # a real loader divergence must name the first mismatching step + differing ids
        fm = d["first_mismatch"]
        assert fm and "step" in fm
        assert fm["rlinf_only_sample"] or fm["ref_only_sample"]
    elif d["verdict"] == "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE":
        # the diagnosis must be evidenced: zero cross-dump frame overlap
        assert d["frame_identity_comparable"] is False
        assert d["cross_dump_distinct_overlap"] == 0
        assert d["rlinf_total_distinct"] > 256 and d["ref_total_distinct"] > 256
    else:
        assert d["first_mismatch"] is None and d["all_steps_identical"] is True


def test_committed_audit_recomputes_to_same_verdict():
    """Recomputing the comparison from the committed dumps reproduces the committed
    verdict (the comparison is a pure function of the dumps)."""
    cmp_committed = _load(_COMPARE)
    rl, rf = _load(_RLINF), _load(_REF)
    recomputed = compare_audits(rl, rf)
    assert recomputed["verdict"] == cmp_committed["verdict"]
    assert recomputed["steps_matched"] == cmp_committed["steps_matched"]
