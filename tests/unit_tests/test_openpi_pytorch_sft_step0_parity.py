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

"""Gate for the identical-data first-step loss/grad parity (data-vs-compute classification).

Two committed artifacts, both data-only (the heavy GPU runs that produced them are reproducible
via tools/sft_grad_parity_probe.py and an 8-GPU FSDP SFT step with data.loader_mode=reference_fanout):

* cross-feed (phase6_ac3_cross_feed.json): the SAME batch + weights + noise/time fed through BOTH
  models agree on loss (abs) and global grad norm (rel) -- COMPUTE isolated and ruled out.
* production step-0 (phase6_ac3_production_step0.json): with IDENTICAL step-0 data (reference_fanout)
  the production loss matches the reference within the bf16 band and grad_norm collapses toward the
  reference, while the decentralized default (different data) is far off -- the first-step gap is
  DATA-side.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EV = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence")
_CROSS = _EV / "phase6_ac3_cross_feed.json"
_PROD = _EV / "phase6_ac3_production_step0.json"
_CAP = _EV / "phase6_ac3_fanout_step0_capture.json"

_LOSS_BAND = 0.01  # bf16 absolute loss band


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (parity run not yet committed)")
    return json.loads(p.read_text())


def test_committed_cross_feed_compute_is_identical():
    """Same batch + weights + noise/time -> both models agree (compute ruled out)."""
    d = _load(_CROSS)
    assert d["classification"] == "COMPUTE_IDENTICAL"
    assert d["n_batches"] >= 1
    assert d["loss_within_dec1"] is True
    assert d["loss_max_abs_delta"] <= _LOSS_BAND
    assert d["grad_within_1pct"] is True
    assert d["grad_norm_max_rel_delta"] <= 0.01
    # every per-batch cell holds the inputs identical and agrees
    for row in d["per_batch"]:
        assert row["loss_abs_delta"] <= _LOSS_BAND
        assert row["grad_rel_delta"] <= 0.01


def test_committed_production_identical_data_is_data_side():
    """reference_fanout (identical step-0 data) -> production loss matches the reference within the
    bf16 band and grad_norm collapses toward the reference; the decentralized default (different
    data) is far off. Classifies the first-step gap as DATA-side."""
    d = _load(_PROD)
    assert d["classification"] == "DATA_SIDE"
    ref = d["reference_production"]
    dec = d["rlinf_decentralized"]
    fan = d["rlinf_reference_fanout"]
    assert dec["loader_mode"] == "per_rank_stream" and fan["loader_mode"] == "reference_fanout"
    # identical-data (fanout) loss is within the bf16 band of the reference ...
    assert d["loss_fanout_within_0p01"] is True
    assert d["loss_abs_delta_fanout_vs_reference"] <= _LOSS_BAND
    # ... and dramatically closer than the decentralized default (data is the cause).
    assert d["loss_abs_delta_fanout_vs_reference"] < d["loss_abs_delta_decentralized_vs_reference"]
    assert d["grad_rel_delta_fanout_vs_reference"] < d["grad_rel_delta_decentralized_vs_reference"]
    # the lr matches the reference warmup init on every arm (logging-semantics check)
    for arm in (dec, fan):
        assert abs(arm["lr"] - ref["lr"]) <= 1e-12


def test_committed_fanout_step0_capture_is_production_faithful():
    """The reference_fanout step-0 capture came from a real 8-GPU run with the rank-disjoint
    256-frame effective batch (the loader fix exercised), and its loss/grad/lr match the
    production-comparison artifact."""
    cap = _load(_CAP)
    assert cap["rank_disjoint"] is True
    assert cap["global_unique_frames"] == 256
    assert cap["fix_exercised"] is True
    prod = _load(_PROD)
    assert abs(cap["step0"]["loss"] - prod["rlinf_reference_fanout"]["loss"]) <= 1e-9
    assert abs(cap["step0"]["grad_norm"] - prod["rlinf_reference_fanout"]["grad_norm"]) <= 1e-6
