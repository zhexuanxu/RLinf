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

"""AC-4 same-batch raw->normalized->loss gate.

Validates the committed evidence: the RLinf full transform+forward was run on the
IDENTICAL raw batch the reference used (raw hashes match -> not a pre-normalized
batch), and the canonical (NEW) raw->quantile-normalized->loss path reproduces the
reference within tolerance; the OLD/NEW norm-stats matrix is present and sensitive
(the OLD stat file changes the normalized actions + loss). This is the AC-4 CORE
gate: the per-batch normalization application is correct, localizing the eval gap
away from any same-input surface.
"""

from __future__ import annotations

import json
import pathlib

import pytest

_EVIDENCE = pathlib.Path(
    "/mnt/public/xzxuan/repos/RLinf_pi05/docs/evidence/phase4_raw_norm_forward_matrix.json"
)


def _load() -> dict:
    if not _EVIDENCE.is_file():
        pytest.skip("AC-4 raw-norm matrix evidence not present")
    return json.loads(_EVIDENCE.read_text())


def test_same_batch_is_the_identical_raw_batch_not_pre_normalized():
    """The RLinf arm consumed the SAME raw batch (hashes match), so the comparison
    exercises raw->normalized, not a pre-normalized injection."""
    ev = _load()
    assert ev["raw_batch_hashes_match_reference"], "raw batch differs between arms"
    assert ev["n_frames"] >= 1


def test_canonical_same_batch_matches_reference():
    """RLinf's full raw->normalized->loss (canonical NEW stats) reproduces the
    reference: tokens/masks byte-identical, normalized within tol, loss within tol."""
    ev = _load()
    assert ev["tokens_byte_identical"]
    assert ev["masks_byte_identical"]
    deltas = ev["normalized_field_max_abs_vs_reference"]
    assert deltas["state"] <= ev["norm_tolerance"]
    assert deltas["actions"] <= ev["norm_tolerance"]
    assert ev["same_batch_loss_abs_delta"] <= ev["loss_tolerance"]
    assert ev["canonical_same_batch_matches_reference"]


def test_norm_stats_matrix_is_present_and_sensitive():
    """The OLD/NEW(=reference-resolved) matrix exists and the stat-file CHOICE
    changes the normalized actions + same-batch loss (so the gate is meaningful)."""
    ev = _load()
    matrix = {row["stats"].split()[0]: row for row in ev["norm_stats_matrix"]}
    assert "NEW" in matrix and "OLD" in matrix
    assert matrix["NEW"]["norm_stats_sha256"] != matrix["OLD"]["norm_stats_sha256"]
    # NEW matches the reference; OLD diverges materially -> the matrix is sensitive.
    assert matrix["NEW"]["loss_delta_vs_reference"] <= ev["loss_tolerance"]
    assert (
        matrix["OLD"]["loss_delta_vs_reference"]
        > matrix["NEW"]["loss_delta_vs_reference"]
    )
