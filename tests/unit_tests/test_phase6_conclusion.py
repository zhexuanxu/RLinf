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

"""Gate for the final Phase-6 conclusion (AC-5).

The conclusion (``docs/evidence/phase6_conclusion.{json,md}``) must be backed by the committed
AC-1..AC-4 artifacts -- every headline claim links to the specific committed artifact/gate that
enforces it, with concrete numbers that MATCH those artifacts. The negative tests reject a
conclusion that (a) claims the decentralized default ``per_rank_stream`` is strictly identical to
the reference, (b) omits an AC's evidence link, or (c) omits the concrete step-0 loss/grad/lr.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
_EV = _REPO / "docs/evidence"
_CONC = _EV / "phase6_conclusion.json"
_CONC_MD = _EV / "phase6_conclusion.md"


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (conclusion not yet committed)")
    return json.loads(p.read_text())


def test_conclusion_links_every_ac_to_an_existing_artifact_and_gate():
    """Every AC-1..AC-4 cites at least one committed artifact + gate, and they all exist on disk."""
    d = _load(_CONC)
    for ac in ("AC-1", "AC-2", "AC-3", "AC-4"):
        e = d["ac_evidence"][ac]
        assert e["artifacts"], ac
        assert e["gates"], ac
        for rel in e["artifacts"] + e["gates"]:
            # artifacts are under docs/evidence/ (bare name) or a repo-relative path; gates are repo-relative.
            cand = (_EV / rel) if "/" not in rel else (_REPO / rel)
            assert cand.is_file(), f"{ac}: missing {rel}"
        assert e["claim"] and e["numbers"], ac


def test_conclusion_numbers_match_the_committed_artifacts():
    """The conclusion's concrete numbers are not free-floating -- they match the source artifacts."""
    d = _load(_CONC)
    n = d["ac_evidence"]
    # AC-1: the decentralized step-0 capture
    cap = json.loads((_EV / "phase6_sft_step0_capture.json").read_text())["step0"]
    assert n["AC-1"]["numbers"]["loss"] == cap["loss"]
    assert n["AC-1"]["numbers"]["grad_norm"] == cap["grad_norm"]
    assert n["AC-1"]["numbers"]["lr"] == cap["lr"]
    # AC-2: the full-30k reference_fanout multiset audit
    fan = json.loads((_EV / "phase6_loader_audit_ids_fanout_compare.json").read_text())
    assert n["AC-2"]["numbers"]["verdict"] == fan["verdict"] == "IDENTICAL_MULTISET"
    assert n["AC-2"]["numbers"]["identical_prefix_steps"] == fan["identical_prefix_steps"] == 30000
    assert fan["global_rolling_hash_match"] is True
    # AC-3: the real production-stack gate + the 2x2 disjoint frames
    stack = json.loads((_EV / "phase6_ac3_production_stack_controlled.json").read_text())
    assert stack["gate"]["pass"] is True
    assert n["AC-3"]["numbers"]["production_stack_real_8gpu"]["loss"] == stack["production_run"]["loss"]
    assert (
        n["AC-3"]["numbers"]["production_stack_real_8gpu"]["grad_norm"]
        == stack["production_run"]["grad_norm_preclip_fp32_global"]
    )
    x2 = json.loads((_EV / "phase6_ac3_cross_feed_2x2.json").read_text())
    assert x2["episode_frame_cross_overlap"] == 0
    assert n["AC-3"]["numbers"]["x2_episode_frame_overlap"] == 0
    # AC-4: step-0 lr is the reference warmup init (matches the AC-1 capture lr)
    assert n["AC-4"]["numbers"]["step0_lr"] == cap["lr"]


def test_conclusion_rejects_per_rank_stream_strictly_identical_claim():
    """NEGATIVE: the conclusion must NOT claim the default per_rank_stream is strictly identical to
    the reference; strict identity is the user-approved reference_fanout mode."""
    d = _load(_CONC)
    assert d["default_per_rank_stream_strictly_identical_to_reference"] is False
    assert d["strict_identity_mode"] == "reference_fanout"
    assert d["outcome"] in ("loader_misalignment_fixed", "old_run_resolved_by_rerun", "compute_bug_fixed")
    # the prior "benign RNG/shuffle" claim is explicitly superseded by the 30k multiset proof.
    assert "benign" in d["supersedes_prior_claim"].lower()


def test_conclusion_md_contains_concrete_numbers_and_links():
    """The human conclusion doc carries the concrete step-0 numbers and the AC artifact filenames."""
    if not _CONC_MD.is_file():
        pytest.skip("phase6_conclusion.md not present")
    md = _CONC_MD.read_text()
    for s in ("0.3275", "6.69", "2.4975e-08", "0.24609", "2.172", "IDENTICAL_MULTISET", "30000",
              "0.2281494", "reference_fanout", "per_rank_stream"):
        assert s in md, f"conclusion.md missing {s!r}"
    for art in ("phase6_sft_step0_capture.json", "phase6_loader_audit_ids_fanout_compare.json",
                "phase6_ac3_cross_feed_2x2.json", "phase6_ac3_production_stack_controlled.json",
                "test_openpi_pytorch_sft_optimizer_parity.py"):
        assert art in md, f"conclusion.md missing link {art}"


def test_conclusion_audit_artifact_present_and_verdict_recorded():
    """task5 independent audit artifact is committed and records its verdict.

    A GAP verdict is allowed only as an explicit pre-AC-5 blocker state with a
    concrete fix. It must not masquerade as a completed final synthesis.
    """
    d = _load(_CONC)
    audit_rel = d["audit"]["artifact"]
    assert (_EV / audit_rel).is_file(), f"missing audit artifact {audit_rel}"
    verdict = d["audit"]["verdict"].upper()
    assert verdict in ("PASS", "GAP"), d["audit"]["verdict"]
    if verdict == "GAP":
        blockers = d["audit"].get("blockers") or []
        assert blockers, "GAP audit must record concrete blockers"
        assert "fix" in blockers[0] and blockers[0]["fix"]
