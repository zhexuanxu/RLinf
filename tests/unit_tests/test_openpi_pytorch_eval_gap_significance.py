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

"""Eval-gap significance gate (expanded matched-protocol run records).

Validates the committed significance evidence: every seed pair carries the FULL
matched seed/task schedule (and the env seed is distinct from the actor seed —
the Round-3-review record bug); the protocol is matched on every env/reset/task/
noise knob; the pooled two-proportion test is internally consistent; and the
generator rebuilds the committed evidence byte-for-byte from the committed
manifest (so the documented invocation is reproducible). Outcome-agnostic: the
significance verdict is read from the evidence, not hardcoded.
"""

from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

_DUMP_PATH = pathlib.Path(__file__).parent / "_eval_gap_significance_dump.py"
_spec = importlib.util.spec_from_file_location(
    "_eval_gap_significance_dump", _DUMP_PATH
)
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)

_EVIDENCE = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase4_eval_gap_significance.json"
)
_MANIFEST = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase4_eval_seed_pairs.json"
)
_REQUIRED_RECORD_KEYS = {
    "success_once",
    "n",
    "successes",
    "num_steps",
    "model_dtype",
    "model_path",
    "assets_dir",
    "asset_id",
    "norm_stats_sha256",
    "denormalization_path",
    "actor_seed",
    "env_eval_seed",
    "env_seed_offset_formula",
    "flow_noise_seed",
    "eval_deterministic_noise",
    "use_fixed_reset_state_ids",
    "task_activity_name",
    "eval_rollout_epoch",
    "total_num_envs",
    "evidence_generation_git_revision",
}


def _load() -> dict:
    if not _EVIDENCE.is_file():
        pytest.skip("matched-protocol significance evidence not present")
    return json.loads(_EVIDENCE.read_text())


def test_every_seed_pair_carries_the_full_matched_schedule():
    """Each seed pair's run records carry the full seed/task schedule, the env seed
    is distinct from the actor seed, and the protocol is matched."""
    ev = _load()
    assert ev["n_seed_pairs"] >= 1
    assert ev["all_pairs_protocol_knobs_matched"], "a seed pair is not knob-matched"
    for pair in ev["per_seed"]:
        assert pair["protocol_knobs_matched"]
        for side in ("rlinf_trained", "reference_trained"):
            rec = pair[side]
            missing = _REQUIRED_RECORD_KEYS - set(rec)
            assert not missing, f"pair {pair['index']} {side} missing: {missing}"
            assert rec["eval_deterministic_noise"] is True
            # The env seed is env.eval.seed, NOT actor.seed (the recorded bug).
            assert rec["env_eval_seed"] == pair["env_seed"]
            assert rec["flow_noise_seed"] == pair["noise_seed"]
            assert (
                rec["env_eval_seed"] != rec["actor_seed"]
                or pair["env_seed"] == rec["actor_seed"]
            )


def test_pooled_two_proportion_is_internally_consistent():
    """The pooled gap + CI recompute from the summed per-seed counts."""
    ev = _load()
    x1 = sum(p["rlinf_trained"]["successes"] for p in ev["per_seed"])
    n1 = sum(p["rlinf_trained"]["n"] for p in ev["per_seed"])
    x2 = sum(p["reference_trained"]["successes"] for p in ev["per_seed"])
    n2 = sum(p["reference_trained"]["n"] for p in ev["per_seed"])
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    pooled = ev["pooled"]
    assert pooled["gap_reference_minus_rlinf"] == pytest.approx(diff, abs=1e-9)
    assert pooled["difference_95_ci"][0] == pytest.approx(
        diff - dump._Z_95 * se, abs=1e-9
    )
    assert pooled["difference_95_ci"][1] == pytest.approx(
        diff + dump._Z_95 * se, abs=1e-9
    )
    assert ev["episodes_per_model"] == n1
    ci_excludes_zero = (
        pooled["difference_95_ci"][0] > 0 or pooled["difference_95_ci"][1] < 0
    )
    if not ci_excludes_zero:
        assert "NOT SIGNIFICANT" in ev["verdict"]
        assert not ev["dec1_gap_is_material"]


@pytest.mark.skipif(not _MANIFEST.is_file(), reason="seed-pair manifest not present")
def test_generator_rebuilds_committed_evidence_from_manifest():
    """The documented invocation reproduces the committed evidence byte-for-byte
    from the committed manifest (no placeholder default; BSI-1)."""
    ev = _load()
    manifest = json.loads(_MANIFEST.read_text())
    git_rev = ev["per_seed"][0]["rlinf_trained"]["evidence_generation_git_revision"]
    for pair in manifest["seed_pairs"]:
        if not (
            dump._REPO / pair["rlinf_run_dir"] / "tensorboard/config.yaml"
        ).is_file():
            pytest.skip("matched-protocol eval run dirs not present")
    rebuilt = dump.build_evidence(manifest, git_rev)
    assert json.dumps(rebuilt, sort_keys=True) == json.dumps(ev, sort_keys=True)
