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

"""STRICT gate for the value-independent (episode:frame) loader multiset audit (AC-2),
schema v2.

The full-30k audit established TWO distinct divergence causes (see
``docs/evidence/phase6_loader_lane_root_cause.md``):

* LANE COUNT — at production num_workers the loaders shard into a different number of lanes
  (reference rank-0 fanout = num_workers=8; RLinf decentralized = world_size*num_workers=64),
  so the global multiset diverges from STEP 0 (``DIVERGENT_MULTISET``, cause ``lane_count``).
* EPOCH BOUNDARY — at matched lane count (RLinf num_workers=1 → 8 lanes) the global multiset is
  IDENTICAL for the ENTIRE FIRST EPOCH (epoch_len steps, incl. step 0), then diverges exactly at
  the epoch boundary because the reference fanout re-creates its single DataLoader iterator on
  exhaustion (13435 batches, not a multiple of world_size), while RLinf's per-rank loaders stream
  a full lane uniformly. Both keep sampling the SAME 429,928-frame pool, 256/step — a benign
  reorder (``IDENTICAL_FIRST_EPOCH``, cause ``epoch_boundary``).

So the matched-lane loaders feed identical data for the whole first epoch INCLUDING the
first-step (step-0) loss/grad comparison; full-30k identity is architecturally impossible while
each pipeline keeps its native (centralized vs decentralized) epoch handling. The gate enforces
exactly this: a clean IDENTICAL leading prefix covering the full first epoch for the aligned
config, and a step-0 lane_count divergence for production — neither is allowed to masquerade as
``IDENTICAL_MULTISET`` (which requires identity at EVERY step).
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
_REF = _EV / "phase6_loader_audit_ids_ref.json"
_ALIGNED = _EV / "phase6_loader_audit_ids_rlinf_aligned.json"
_PROD = _EV / "phase6_loader_audit_ids_rlinf_prod.json"
_ALIGNED_CMP = _EV / "phase6_loader_audit_ids_aligned_compare.json"
_PROD_CMP = _EV / "phase6_loader_audit_ids_prod_compare.json"
_EQUIV = _EV / "phase6_loader_audit_ids_equiv.json"
_NONPERTURB = _EV / "phase6_loader_audit_ids_nonperturb.json"

_EXPECTED_STEPS = 30000
_EPOCH_LEN = 1679  # (429928 // 32) // 8 — reference fanout epoch in global steps


# ----------------------------- pure-logic comparator tests -----------------------------

def _dump(global_roll, ranks, steps=3, **extra):
    d = {
        "ok": True, "schema_version": 2, "audited_steps": steps, "num_workers": 8,
        "global_rolling_hash": global_roll, "per_rank_rolling_hash": ranks,
        "per_step_unique": [256] * steps,
    }
    d.update(extra)
    return d


def _detail(sets, hashes):
    return {"per_step_global_set": sets, "per_step_global_hash": hashes}


def test_compare_identical_full_multiset():
    a = _dump("GH", ["r0", "r1"])
    b = _dump("GH", ["r0", "r1"], num_workers=1)
    d = _detail([["x:1", "y:2"]] * 3, ["h", "h", "h"])
    out = compare_id_audits(a, b, d, d, epoch_len=2)
    assert out["verdict"] == "IDENTICAL_MULTISET"
    assert out["identical_full"] is True
    assert out["identical_prefix_steps"] == 3
    assert out["frame_identity_comparable"] is True


def test_compare_identical_first_epoch_then_epoch_boundary():
    # identical for the first epoch (2 steps) then diverges at the epoch boundary.
    a = _dump("GH_A", ["r0"], steps=3)
    b = _dump("GH_B", ["r0"], steps=3)
    da = _detail([["x:1"], ["y:2"], ["p:1"]], ["h0", "h1", "h2a"])
    db = _detail([["x:1"], ["y:2"], ["q:9"]], ["h0", "h1", "h2b"])
    out = compare_id_audits(a, b, da, db, epoch_len=2)
    assert out["verdict"] == "IDENTICAL_FIRST_EPOCH"
    assert out["identical_first_epoch"] is True
    assert out["identical_full"] is False
    assert out["identical_prefix_steps"] == 2
    assert out["first_divergence_cause"] == "epoch_boundary"


def test_compare_divergent_lane_count_at_step0():
    a = _dump("GH_A", ["r0"], steps=3)
    b = _dump("GH_B", ["r0"], steps=3)
    da = _detail([["x:1", "a:2"], ["y:2"], ["p:1"]], ["h0a", "h1", "h2"])
    db = _detail([["x:1", "b:9"], ["y:2"], ["p:1"]], ["h0b", "h1", "h2"])
    out = compare_id_audits(a, b, da, db, epoch_len=2)
    assert out["verdict"] == "DIVERGENT_MULTISET"
    assert out["identical_full"] is False and out["identical_first_epoch"] is False
    assert out["identical_prefix_steps"] == 0
    assert out["first_divergence_cause"] == "lane_count"
    fm = out["first_mismatch"]
    assert fm["step"] == 0 and "a:2" in fm["rlinf_only_sample"] and "b:9" in fm["ref_only_sample"]


def test_compare_inconclusive_when_id_space_not_comparable():
    a = _dump("GH_A", ["r0"])
    b = _dump("GH_B", ["r0"])
    out = compare_id_audits(a, b, _detail([["x:1"]], ["h"]), _detail([["y:9"]], ["h2"]), epoch_len=1)
    assert out["verdict"] == "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE"
    assert out["frame_identity_comparable"] is False
    assert out["identical_full"] is False


def test_divergent_never_reports_identical_full():
    """A first-epoch or step-0 divergence must never set identical_full (no masquerade)."""
    a = _dump("GH_A", ["r0"], steps=3)
    b = _dump("GH_B", ["r0"], steps=3)
    da = _detail([["x:1"], ["y:2"], ["p:1"]], ["h0", "h1", "h2a"])
    db = _detail([["x:1"], ["y:2"], ["q:9"]], ["h0", "h1", "h2b"])
    out = compare_id_audits(a, b, da, db, epoch_len=2)
    assert out["identical_full"] is False
    assert out["global_rolling_hash_match"] is False


def test_canonical_set_hash_is_unambiguous():
    from tools.sft_loader_audit_rlinf import _canonical_set_hash

    assert _canonical_set_hash(["a", "bc"]) != _canonical_set_hash(["ab", "c"])
    assert _canonical_set_hash(["b:2", "a:1"]) == _canonical_set_hash(["a:1", "b:2"])


# ----------------------------- committed-evidence tests -----------------------------

def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (id audit not yet run/committed)")
    return json.loads(p.read_text())


def _assert_schema_v2_full_30k(d, name, expected_nw):
    assert d["ok"] is True, name
    assert d["schema_version"] == 2, f"{name} schema {d.get('schema_version')}"
    assert d["audited_steps"] == _EXPECTED_STEPS, f"{name} steps {d['audited_steps']}"
    assert d["num_workers"] == expected_nw, f"{name} num_workers {d['num_workers']}"
    uniq = d["per_step_unique"]
    assert len(uniq) == _EXPECTED_STEPS and min(uniq) == 256 and max(uniq) == 256, name
    assert d.get("global_rolling_hash"), f"{name} missing global_rolling_hash"
    prr = d.get("per_rank_rolling_hash")
    assert isinstance(prr, list) and len(prr) == 8, f"{name} per_rank_rolling_hash"


def test_committed_reference_aligned_and_prod_cover_full_30k():
    _assert_schema_v2_full_30k(_load(_REF), "ref", 8)            # 8 lanes (fanout nw=8)
    _assert_schema_v2_full_30k(_load(_ALIGNED), "rlinf_aligned", 1)  # 8 lanes (nw=1)
    _assert_schema_v2_full_30k(_load(_PROD), "rlinf_prod", 8)    # 64 lanes (nw=8)


def test_committed_aligned_is_IDENTICAL_FIRST_EPOCH_including_step0():
    """At matched lane count the global multiset is IDENTICAL for the whole first epoch
    (incl. step 0 — the first-step loss/grad data); it then reorders the SAME pool at the
    epoch boundary. This is the AC-2 result for the reference-aligned config."""
    d = _load(_ALIGNED_CMP)
    assert d["verdict"] == "IDENTICAL_FIRST_EPOCH", d["verdict"]
    assert d["identical_first_epoch"] is True
    assert d["identical_full"] is False
    assert d["identical_prefix_steps"] >= _EPOCH_LEN  # full first epoch matches
    assert d["identical_prefix_steps"] >= 1           # step 0 (and step 1) included
    assert d["first_divergence_cause"] == "epoch_boundary"
    assert d["frame_identity_comparable"] is True
    assert (d["cross_dump_distinct_overlap"] or 0) > 0
    assert d["audited_steps"] == _EXPECTED_STEPS


def test_committed_production_is_DIVERGENT_from_step0_lane_count():
    """Production RLinf (nw=8 -> 64 lanes) vs the reference (8 lanes) diverges from step 0;
    this is the lane-count cause and is explicitly NOT an AC-2 identity pass."""
    d = _load(_PROD_CMP)
    assert d["verdict"] == "DIVERGENT_MULTISET"
    assert d["identical_full"] is False and d["identical_first_epoch"] is False
    assert d["identical_prefix_steps"] == 0
    assert d["first_divergence_cause"] == "lane_count"
    assert d["frame_identity_comparable"] is True
    assert (d["cross_dump_distinct_overlap"] or 0) > 0
    assert d["first_mismatch"] and d["first_mismatch"]["step"] == 0


def test_committed_reference_shortcut_equals_production_wrapper():
    """B3: the id_only raw-DataLoader shortcut emits the same id stream as the production
    create_behavior_data_loader_torch DataLoader settings (shuffle/sampler/worker_init no-ops)."""
    d = _load(_EQUIV)
    assert d["ok"] is True
    assert d["identical"] is True
    assert d["prod_settings_rolling"] == d["shortcut_rolling"]


def test_committed_nonperturbation_both_repos_byte_identical():
    """B4: two independent full-30k captures reproduce the rolling hash byte-for-byte, for the
    reference (nw=8) AND the reference-aligned RLinf (nw=1) — the streams in the AC-2 comparison."""
    d = _load(_NONPERTURB)
    assert d["ref_run1_rolling"] == d["ref_run2_rolling"], "reference audit perturbed"
    assert d["rlinf_run1_rolling"] == d["rlinf_run2_rolling"], "RLinf audit perturbed"
    assert d["audited_steps"] == _EXPECTED_STEPS
