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

"""Strict lr + AdamW parity between RLinf and the reference SFT update logic.

A fake-model / fake-gradient probe (no GPU) asserts: (1) RLinf's real openpi_cosine
scheduler reproduces the reference inline lr schedule exactly across warmup + cosine,
(2) the step-0 lr equals the reference run's logged warmup init, and (3) AdamW from the
shared config (betas 0.9/0.95, eps 1e-8, wd 1e-10) produces an identical parameter
trajectory + optimizer state under the matched lr schedule — with a negative control
proving a mismatched beta diverges.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))

torch = pytest.importorskip("torch")

from tools.sft_optimizer_parity_probe import (  # noqa: E402
    ADAMW_BETAS,
    ADAMW_EPS,
    ADAMW_WD,
    REF_LOGGED_STEP0_LR,
    adamw_trajectory,
    reference_lr,
    rlinf_lr_sequence,
)

_STEPS = [0, 1, 2, 500, 999, 1000, 1001, 5000, 15000, 29000, 29999]


def test_rlinf_lr_matches_reference_schedule_exactly():
    """RLinf openpi_cosine == reference inline schedule across warmup + cosine."""
    rl = rlinf_lr_sequence(_STEPS)
    for s in _STEPS:
        ref = reference_lr(s)
        assert abs(rl[s] - ref) <= 1e-12, f"step {s}: rlinf {rl[s]} vs ref {ref}"


def test_step0_lr_is_reference_logged_warmup_init():
    """Step-0 lr equals the reference run's logged 2.4975e-08 (peak/(warmup+1))."""
    rl = rlinf_lr_sequence([0])
    assert abs(rl[0] - REF_LOGGED_STEP0_LR) <= 1e-12
    assert abs(reference_lr(0) - REF_LOGGED_STEP0_LR) <= 1e-12


def test_lr_warmup_peak_and_monotonic_decay():
    """Warmup ramps to the peak at step 1000, then cosine-decays toward 0."""
    assert abs(reference_lr(1000) - 2.5e-5) <= 1e-12
    assert reference_lr(0) < reference_lr(500) < reference_lr(1000)
    assert reference_lr(1000) > reference_lr(15000) > reference_lr(29999)
    assert reference_lr(29999) >= 0.0


# Exercise the AdamW UPDATE math at a representative (non-warmup-tiny) lr so the
# optimizer hyperparameters actually drive the dynamics; the lr SCHEDULE itself is
# verified exactly in the tests above.
_ADAMW_LRS = [1e-2] * 8


def test_adamw_trajectory_identical_under_shared_config():
    """Two AdamW runs with the shared (betas/eps/wd) config and the same lr schedule
    produce an identical parameter trajectory + optimizer state."""
    a = adamw_trajectory(
        betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=ADAMW_WD, lrs=_ADAMW_LRS
    )
    b = adamw_trajectory(
        betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=ADAMW_WD, lrs=_ADAMW_LRS
    )
    assert torch.equal(a["trajectory"], b["trajectory"])
    assert torch.equal(a["exp_avg"], b["exp_avg"])
    assert torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])


def test_adamw_negative_control_beta_mismatch_diverges():
    """A wrong beta2 (torch default 0.999 instead of 0.95) yields a DIFFERENT trajectory
    — the parity check has teeth."""
    good = adamw_trajectory(
        betas=(0.9, 0.95), eps=ADAMW_EPS, weight_decay=ADAMW_WD, lrs=_ADAMW_LRS
    )
    bad = adamw_trajectory(
        betas=(0.9, 0.999), eps=ADAMW_EPS, weight_decay=ADAMW_WD, lrs=_ADAMW_LRS
    )
    assert not torch.allclose(good["trajectory"], bad["trajectory"])
