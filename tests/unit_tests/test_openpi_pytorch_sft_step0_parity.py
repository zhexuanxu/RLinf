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

"""Gate for the first-step loss/grad parity (data-vs-compute classification).

Committed, reproducible artifacts (the heavy GPU runs are reproducible via
tools/sft_cross_feed_2x2.py and an 8-GPU FSDP SFT step with data.loader_mode=reference_fanout):

* 2x2 cross-feed (phase6_ac3_cross_feed_2x2.json): EACH repo's ACTUAL materialized step-0 batch
  (256 fully-transformed frames; frame ids + content hashes recorded) is fed through BOTH models
  with a SHARED flow noise/time, production-faithful (fp32 master + bf16 compute) loss + pre-clip
  fp32 global grad norm. Each COLUMN (same batch, both models) agrees within loss abs <= 0.01 and
  grad rel <= 2% (compute parity / the identical-data production grad with controlled noise); each
  ROW (two distinct batches, same noise) differs (the data difference) -> the gap is DATA-side.
* production step-0 (phase6_ac3_production_step0.json + phase6_ac3_fanout_step0_capture.json): a real
  8-GPU FSDP step-0; with identical data (reference_fanout) the live-noise production loss matches the
  reference within the bf16 band and collapses from the decentralized default, demonstrating the
  data-side cause in production.
* production-stack controlled noise (phase6_ac3_production_stack_controlled.json): the REAL 8-GPU FSDP
  stack fed the IDENTICAL reference_fanout step-0 batch (bit-for-bit the 2x2 reference batch) + the
  CONTROLLED seed-4242 flow noise/time reproduces the reference step-0 loss (abs <= 0.01) AND pre-clip
  fp32 global grad norm (rel <= 2%) directly -- no defect in the production stack on identical data.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EV = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence")
_X2 = _EV / "phase6_ac3_cross_feed_2x2.json"
_PROD = _EV / "phase6_ac3_production_step0.json"
_CAP = _EV / "phase6_ac3_fanout_step0_capture.json"
_STACK = _EV / "phase6_ac3_production_stack_controlled.json"

_LOSS_BAND = 0.01  # bf16 absolute loss band
_GRAD_REL = 0.02  # global grad-norm relative tolerance


def _load(p):
    if not p.is_file():
        pytest.skip(f"{p.name} not present (parity run not yet committed)")
    return json.loads(p.read_text())


def test_committed_2x2_compute_parity_per_column():
    """Each ACTUAL step-0 batch fed through BOTH models (shared weights + noise/time) agrees on
    loss (abs <= 0.01) and pre-clip fp32 global grad norm (rel <= 2%) -- compute is identical, and
    this is the production identical-data grad with CONTROLLED noise (ruling out the model)."""
    d = _load(_X2)
    assert d["total_frames"] == 256
    assert d["compute_loss_within_dec1"] is True
    assert d["compute_grad_within_2pct"] is True
    for batch in ("ref_batch", "rlinf_batch"):
        assert d["compute_loss_abs_delta"][batch] <= _LOSS_BAND, batch
        assert d["compute_grad_rel_delta"][batch] <= _GRAD_REL, batch
        # all four cells were computed (batch_source x model_side)
        for side in ("ref_model", "rlinf_model"):
            assert "loss" in d["cells"][batch][side] and "grad_norm" in d["cells"][batch][side]


def test_committed_2x2_is_data_side():
    """The two materialized step-0 batches are DISTINCT (different frames) and, on the SAME model
    with the SAME noise, give materially different loss -> the first-step gap is DATA-side."""
    d = _load(_X2)
    assert d["classification"] == "DATA_SIDE"
    assert d["batches_distinct"] is True
    assert d["ref_batch_content_hash"] != d["rlinf_batch_content_hash"]
    # completely different 256-frame batches
    assert len(set(d["ref_batch_frame_ids"]) & set(d["rlinf_batch_frame_ids"])) == 0
    assert len(d["ref_batch_frame_ids"]) == 256 and len(d["rlinf_batch_frame_ids"]) == 256
    # same model, two batches -> distinguishable loss (well beyond the bf16 band)
    for side in ("ref_model", "rlinf_model"):
        assert d["data_loss_abs_delta"][side] > _LOSS_BAND


def test_committed_2x2_per_surface_content_differs():
    """Per-surface integrity hashes: every CONTENT surface (state, actions, each image, tokenized
    prompt) differs between the two distinct step-0 batches; the all-True image masks are identical
    (same task), proving the materialization difference is in the data, not a missing surface."""
    d = _load(_X2)
    ps = d["per_surface_hashes"]
    assert d["content_surfaces_all_differ"] is True
    for k in d["content_surfaces"]:
        assert ps["ref_batch"][k] != ps["rlinf_batch"][k], k
    assert d["mask_surfaces_all_identical"] is True
    for k in d["mask_surfaces"]:
        assert ps["ref_batch"][k] == ps["rlinf_batch"][k], k
    assert {"state", "actions"} <= set(d["content_surfaces"])
    assert any(k.startswith("image__") for k in d["content_surfaces"])


def test_committed_2x2_episode_frame_ids_disjoint():
    """Value-independent (episode_index, frame_index) ids (cross-repo comparable, from the id_only
    loader audits): each step-0 batch is 256 DISTINCT frames and the two batches share 0 frames."""
    d = _load(_X2)
    ef = d["episode_frame_ids"]
    ref, rl = ef["ref_batch"], ef["rlinf_batch"]
    assert len(ref) == 256 and len(set(ref)) == 256
    assert len(rl) == 256 and len(set(rl)) == 256
    assert len(set(ref) & set(rl)) == 0
    assert d["episode_frame_unique"]["ref_batch"] == 256
    assert d["episode_frame_unique"]["rlinf_batch"] == 256
    assert d["episode_frame_cross_overlap"] == 0
    # the ids are "episode:frame" (value-independent), not byte-hashes
    assert all(":" in x for x in ref[:8] + rl[:8])


def test_committed_production_identical_data_grad_within_2pct():
    """The production identical-data step-0 grad norm matches within rel <= 2% when the flow
    noise/time is controlled: the 2x2 ref-batch column (both production-faithful stacks on the
    identical reference step-0 batch + a shared noise draw) agrees on grad norm within 2%."""
    d = _load(_X2)
    assert d["compute_grad_rel_delta"]["ref_batch"] <= _GRAD_REL
    # the ref-batch grad reproduces the reference production-scale step-0 grad (~2.1-2.4)
    g = d["cells"]["ref_batch"]["ref_model"]["grad_norm"]
    assert 1.5 <= g <= 3.0, g


def test_committed_production_step0_demonstration_is_data_side():
    """Live-noise 8-GPU production: with identical data (reference_fanout) the step-0 loss matches
    the reference within the bf16 band and is far closer than the decentralized default."""
    d = _load(_PROD)
    assert d["classification"] == "DATA_SIDE"
    assert d["loss_fanout_within_0p01"] is True
    assert d["loss_abs_delta_fanout_vs_reference"] <= _LOSS_BAND
    assert d["loss_abs_delta_fanout_vs_reference"] < d["loss_abs_delta_decentralized_vs_reference"]
    assert d["grad_rel_delta_fanout_vs_reference"] < d["grad_rel_delta_decentralized_vs_reference"]
    ref, dec, fan = (d["reference_production"], d["rlinf_decentralized"], d["rlinf_reference_fanout"])
    assert dec["loader_mode"] == "per_rank_stream" and fan["loader_mode"] == "reference_fanout"
    for arm in (dec, fan):
        assert abs(arm["lr"] - ref["lr"]) <= 1e-12


def test_committed_production_stack_controlled_within_2pct():
    """The REAL 8-GPU FSDP production stack, fed the IDENTICAL reference_fanout step-0 batch +
    CONTROLLED seed-4242 noise, reproduces the reference step-0 loss (abs <= 0.01) and pre-clip
    fp32 global grad norm (rel <= 2%) -- gated directly on the production-stack artifact (not the
    single-GPU 2x2 replay). The pinned inputs are bit-for-bit the 2x2 reference batch + noise."""
    d = _load(_STACK)
    # the production run consumed inputs bit-identical to the committed 2x2 reference batch + noise.
    ii = d["input_identity"]
    assert ii["matches_2x2_ref_batch_content_hash"] is True
    assert ii["all_surfaces_match_2x2_ref_batch"] is True
    assert ii["noise_bit_identical_to_2x2"] is True and ii["time_bit_identical_to_2x2"] is True
    assert ii["noise_seed"] == 4242
    # the real 8-GPU FSDP stack (global-mean loss, all-reduced mean grad, pre-clip fp32 norm).
    assert d["production_run"]["world_size"] == 8 and d["production_run"]["global_batch"] == 256
    # direct gate vs the reference target on the bit-identical batch + noise.
    g = d["gate"]
    assert g["loss_abs_delta"] <= _LOSS_BAND and g["loss_pass"] is True
    assert g["grad_rel_delta"] <= _GRAD_REL and g["grad_pass"] is True
    assert g["pass"] is True
    # the reference target is the committed 2x2 ref-batch ref-model cell (same inputs).
    x2 = _load(_X2)["cells"]["ref_batch"]["ref_model"]
    assert abs(d["reference_target"]["loss"] - x2["loss"]) <= 1e-9
    assert abs(d["reference_target"]["grad_norm_preclip_fp32_global"] - x2["grad_norm"]) <= 1e-6


def test_committed_fanout_step0_capture_is_production_faithful():
    cap = _load(_CAP)
    assert cap["rank_disjoint"] is True
    assert cap["global_unique_frames"] == 256
    assert cap["fix_exercised"] is True
    prod = _load(_PROD)
    assert abs(cap["step0"]["loss"] - prod["rlinf_reference_fanout"]["loss"]) <= 1e-9
    assert abs(cap["step0"]["grad_norm"] - prod["rlinf_reference_fanout"]["grad_norm"]) <= 1e-6
