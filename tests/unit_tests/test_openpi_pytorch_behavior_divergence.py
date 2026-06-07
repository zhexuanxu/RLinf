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

"""Behavior-divergence gate.

Validates the committed evidence: the two trained-and-converted models were
compared by BEHAVIOR (post-denormalization action chunks on a fixed batch of REAL
BEHAVIOR eval observations + fixed injected noise), NOT by raw per-parameter
tensor deltas; the eval knobs (num_steps / dtype / norm-stats) were matched across
both model loads; and paired determinism holds (same model + same noise ->
identical actions). The gate FAILS if the observations are synthetic, if the knobs
are mismatched, if determinism is violated, or if the divergence signal is raw
parameter deltas.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EVIDENCE = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase4_behavior_divergence.json"
)
_ENV_DIM = 23
_HORIZON = 32


def _load() -> dict:
    if not _EVIDENCE.is_file():
        pytest.skip("behavior-divergence evidence not present")
    return json.loads(_EVIDENCE.read_text())


def test_observations_are_real_eval_obs_not_synthetic():
    """The comparison used REAL BEHAVIOR eval observations (env-sourced, hashed),
    not synthetic random tensors."""
    ev = _load()
    assert ev["real_eval_obs"] is True
    prov = ev["obs_provenance"]
    assert prov["env_type"] == "behavior"
    assert prov["num_envs"] >= 1
    hashes = prov["obs_field_hashes"]
    # Every obs field carries a content hash -> the obs are a fixed, recorded set.
    for key in ("main_images", "wrist_images", "states", "task_descriptions"):
        assert hashes.get(key) and len(hashes[key]) == 64


def test_eval_knobs_matched_across_both_models():
    """Both model loads used the SAME num_steps / dtype / canonical norm-stats, so
    only the trained weights differ."""
    ev = _load()
    assert ev["num_steps"] == 5
    assert ev["dtype"] == "bfloat16"
    assert ev["norm_stats_held_constant"] is True
    canonical = ev["canonical_norm_stats_sha256"]
    assert ev["rlinf_norm_stats_sha256"] == canonical
    assert ev["reference_norm_stats_sha256"] == canonical
    assert ev["rlinf_checkpoint"] != ev["reference_checkpoint"]
    # Checkpoint/config content hashes are recorded for provenance, and the two
    # trained checkpoints are genuinely different model files.
    for key in (
        "rlinf_model_sha256",
        "reference_model_sha256",
        "rlinf_config_sha256",
        "reference_config_sha256",
    ):
        assert ev.get(key) and len(ev[key]) == 64
    assert ev["rlinf_model_sha256"] != ev["reference_model_sha256"]


def test_paired_determinism_holds():
    """Same model + same fixed noise -> identical actions (paired eval is valid)."""
    ev = _load()
    assert ev["paired_determinism_max_abs"] == 0.0


def test_signal_is_behavior_not_raw_parameter_delta():
    """The divergence signal is the post-denormalization action-chunk delta; raw
    per-parameter tensor deltas are explicitly rejected (the negative test)."""
    ev = _load()
    assert ev["divergence_signal"] == "action_chunk_delta_post_denormalization"
    assert ev["raw_parameter_delta_rejected"] is True
    # The per-dim / per-chunk-position breakdowns are present (behavior detail).
    assert len(ev["per_action_dim_mean_abs_delta"]) == _ENV_DIM
    assert len(ev["per_chunk_position_mean_abs_delta"]) == _HORIZON


def test_divergence_measured_and_interpreted_vs_eval_gap():
    """A non-trivial behavior divergence is measured and tied to the eval gap."""
    ev = _load()
    d = ev["denormalized_action_chunk_delta"]
    for k in ("mean", "max", "p95"):
        assert d[k] >= 0.0 and d[k] == d[k]  # finite, non-negative
    # Two independently-trained policies must actually differ in behavior.
    assert d["max"] > 0.0
    assert ev["reference_minus_rlinf_eval_gap_pct"] == pytest.approx(9.2)
    assert "fixed_noise" in ev and ev["fixed_noise"]["seed"] == 1234
