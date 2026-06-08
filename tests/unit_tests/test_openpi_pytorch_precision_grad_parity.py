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

"""Controlled-step numeric grad-norm parity on IDENTICAL weights.

Both repos load the SAME base checkpoint (``pi05_base_pytorch_new``) and run ONE
forward/backward on the SAME real pinned SFT batch (same noise/time). This gate proves,
from RUNTIME tensors, that:

1. the loaded weights are byte-identical across repos -- a key-independent set-hash over
   each fp32 param value (gathered from an unsharded FULL_STATE_DICT *before* the
   optimizer step) matches between RLinf and the reference on every rank; and
2. the resulting grad-norm is numerically equivalent -- identical fp32 accumulation
   dtype, and the value agrees within a documented bf16-compute tolerance.

The residual value gap is bf16-compute non-determinism across different execution paths
(RLinf eager vs reference ``torch.compile``; differing FSDP wrap granularity / all-reduce
order), NOT a precision-recipe difference. The dtype -- the precision surface -- is
identical fp32.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EV = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence")
_RANKS = (0, 1)

# Documented tolerance for the grad-norm VALUE across the two execution paths. The
# observed gap is ~0.5% (bf16 compute, compile-vs-eager, FSDP all-reduce order); this
# generous bound asserts numeric equivalence without pinning the exact non-deterministic
# value.
_GRAD_NORM_REL_TOL = 0.02


def _load(repo, rank):
    p = _EV / f"phase5_precision_ledger_{repo}_rank{rank}.json"
    if not p.is_file():
        pytest.skip(f"precision ledger {p.name} not present")
    return json.loads(p.read_text())


def test_loaded_weights_byte_identical_across_repos_both_ranks():
    """The key-independent fp32 weight set-hash matches RLinf<->reference on each rank,
    proving both consumed the SAME base checkpoint at the controlled step."""
    for rank in _RANKS:
        rl = _load("rlinf", rank)["surfaces"].get("loaded_weight_fingerprint")
        rf = _load("ref", rank)["surfaces"].get("loaded_weight_fingerprint")
        assert rl and rf, f"rank{rank} missing loaded_weight_fingerprint"
        rlv, rfv = rl["value"], rf["value"]
        assert rlv["n_distinct_float_tensors"] > 8
        assert rlv["set_hash"] == rfv["set_hash"], (
            f"rank{rank} loaded weights differ across repos: "
            f"{rlv['set_hash'][:12]} != {rfv['set_hash'][:12]}"
        )


def test_loaded_weight_fingerprint_read_from_runtime_full_state_dict():
    """The fingerprint provenance points at a gathered FULL_STATE_DICT (a runtime
    tensor sweep), never config text."""
    for repo in ("rlinf", "ref"):
        rec = _load(repo, 0)["surfaces"]["loaded_weight_fingerprint"]
        prov = rec["provenance"].lower()
        assert "full_state_dict" in prov and ".yaml" not in prov
        assert rec.get("stage") == "pre_step"


def test_grad_norm_numeric_parity_within_tolerance():
    """Grad-norm dtype is identical fp32 and the value agrees within the documented
    bf16-compute tolerance -- numeric parity on identical weights."""
    for rank in _RANKS:
        gl = _load("rlinf", rank)["surfaces"]["grad_norm"]["value"]
        gf = _load("ref", rank)["surfaces"]["grad_norm"]["value"]
        assert gl["dtype"] == "float32" and gf["dtype"] == "float32"
        a, b = float(gl["value"]), float(gf["value"])
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        assert rel <= _GRAD_NORM_REL_TOL, (
            f"rank{rank} grad-norm parity {a:.5f} vs {b:.5f} rel={rel:.4f} "
            f"exceeds {_GRAD_NORM_REL_TOL}"
        )


def test_grad_norm_parity_uses_same_pinned_real_batch():
    """The parity is on the SAME real pinned SFT batch (matching per-field sha256, the
    committed non-synthetic ref_pinned artifact)."""
    rl = _load("rlinf", 0)["pinned_input"]
    rf = _load("ref", 0)["pinned_input"]
    assert rl["field_sha256"] == rf["field_sha256"] and len(rl["field_sha256"]) >= 5
    assert rl["artifact"]["synthetic"] is False
    assert rl["artifact"]["format"] == "ref_pinned_npz"
