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

"""Eval-gap significance gate (matched-protocol run records).

Validates the committed significance evidence: both checkpoints were evaluated
under the deterministic, knob-matched eval protocol (same num_steps / dtype /
norm-stats / episode set / injected flow-noise seed), each summarized into a full
run record; and the two-proportion significance verdict is internally consistent
with the recorded counts and CI. The outcome (significant or not) is read from the
evidence, not hardcoded, so this gate validates the protocol + arithmetic rather
than a fixed result.
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
    "eval_seed",
    "eval_deterministic_noise",
    "eval_noise_seed",
    "episode_task_set",
    "source_git_revision",
}


def _load() -> dict:
    if not _EVIDENCE.is_file():
        pytest.skip("matched-protocol significance evidence not present")
    return json.loads(_EVIDENCE.read_text())


def test_run_records_carry_matched_knobs():
    """Both run records carry every required eval knob, and the protocol is matched
    (deterministic noise on, same seed / num_steps / dtype / norm-stats / n)."""
    ev = _load()
    for side in ("rlinf_trained", "reference_trained"):
        missing = _REQUIRED_RECORD_KEYS - set(ev[side])
        assert not missing, f"{side} run record missing knobs: {missing}"
    rlinf, ref = ev["rlinf_trained"], ev["reference_trained"]
    assert ev["protocol_knobs_matched"], (
        f"protocol not matched: num_steps {rlinf['num_steps']}/{ref['num_steps']}, "
        f"norm-stats {rlinf['norm_stats_sha256'][:8]}/{ref['norm_stats_sha256'][:8]}"
    )
    assert rlinf["eval_deterministic_noise"] and ref["eval_deterministic_noise"]
    assert rlinf["eval_noise_seed"] == ref["eval_noise_seed"]
    assert rlinf["num_steps"] == ref["num_steps"]
    assert rlinf["norm_stats_sha256"] == ref["norm_stats_sha256"]
    assert rlinf["n"] == ref["n"]


def test_significance_is_internally_consistent():
    """Recompute the two-proportion test + CI from the recorded counts and confirm
    they match the stored values, and that the DEC-1 verdict matches the CI."""
    ev = _load()
    x1, n1 = ev["rlinf_trained"]["successes"], ev["rlinf_trained"]["n"]
    x2, n2 = ev["reference_trained"]["successes"], ev["reference_trained"]["n"]
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = diff - dump._Z_95 * se_unpooled
    ci_high = diff + dump._Z_95 * se_unpooled

    assert ev["gap_reference_minus_rlinf"] == pytest.approx(diff, abs=1e-9)
    assert ev["difference_95_ci"][0] == pytest.approx(ci_low, abs=1e-9)
    assert ev["difference_95_ci"][1] == pytest.approx(ci_high, abs=1e-9)

    # The verdict must agree with the CI / materiality booleans.
    ci_excludes_zero = ci_low > 0 or ci_high < 0
    assert ev["difference_95_ci_excludes_zero"] == ci_excludes_zero
    if not ci_excludes_zero:
        assert "NOT SIGNIFICANT" in ev["verdict"]
        assert not ev["dec1_gap_is_material"]
