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
    REF_PEAK_LR,
    build_reference_optimizer,
    build_rlinf_optimizer,
    reference_lr,
    rlinf_lr_sequence,
    run_builder,
    _fake_single_param_model,
)


def _fixed_grads(n_steps=8, n=16):
    return [
        torch.sin(torch.arange(n, dtype=torch.float64) + s * 0.5).reshape(1, n)
        for s in range(n_steps)
    ]

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


# Exercise the AdamW UPDATE at a representative (non-warmup-tiny) lr so hyperparameters
# and the construction actually drive the dynamics; the lr SCHEDULE is verified exactly
# above. Two lr regimes: the real warmup schedule and a moderate constant lr.
_MODERATE_LRS = [1e-2] * 8


def _rlinf_traj(lrs):
    m, p = _fake_single_param_model()
    return run_builder(build_rlinf_optimizer(m), p, _fixed_grads(len(lrs)), lrs)


def _ref_traj(lrs, **adamw_overrides):
    m, p = _fake_single_param_model()
    if adamw_overrides:
        opt = torch.optim.AdamW(m.parameters(), lr=REF_PEAK_LR, **adamw_overrides)
    else:
        opt = build_reference_optimizer(m)
    return run_builder(opt, p, _fixed_grads(len(lrs)), lrs)


def test_rlinf_builder_step_counter_is_zero_after_build():
    """RLinf's real build_optimizer (which runs warmup_optimizer_state) must NOT leave
    the AdamW step counter advanced — otherwise its first real update uses step+1 bias
    correction and diverges from the reference."""
    m, p = _fake_single_param_model()
    opt = build_rlinf_optimizer(m)
    st = opt.state[p]["step"]
    step = int(st.item() if torch.is_tensor(st) else st)
    assert step == 0, f"warmup left step={step}; first update would be off-by-one"


def test_rlinf_and_reference_optimizer_updates_identical():
    """RLinf's REAL optimizer builder and the reference builder produce a bit-identical
    parameter trajectory + AdamW state under the same fixed grads + matched lr."""
    for lrs in ([reference_lr(s) for s in range(8)], _MODERATE_LRS):
        a, b = _rlinf_traj(lrs), _ref_traj(lrs)
        assert a["final_step"] == b["final_step"]
        assert torch.equal(a["trajectory"], b["trajectory"]), "optimizer update mismatch"
        assert torch.equal(a["exp_avg"], b["exp_avg"])
        assert torch.equal(a["exp_avg_sq"], b["exp_avg_sq"])


def test_optimizer_negative_controls_diverge():
    """Each of beta2 / eps / weight_decay / lr (param-group) mismatch makes the reference
    builder DIVERGE from RLinf — the parity gate has teeth on every hyperparameter."""
    rl = _rlinf_traj(_MODERATE_LRS)
    # Use unambiguously-different values so each hyperparameter materially changes the
    # update (a near-equal eps, say 1e-6 vs 1e-8, is genuinely negligible when the Adam
    # denominator is O(0.1) — that is correct behavior, not a gate weakness).
    bad_beta = _ref_traj(_MODERATE_LRS, betas=(0.9, 0.5), eps=ADAMW_EPS, weight_decay=ADAMW_WD)
    bad_eps = _ref_traj(_MODERATE_LRS, betas=ADAMW_BETAS, eps=1.0, weight_decay=ADAMW_WD)
    bad_wd = _ref_traj(_MODERATE_LRS, betas=ADAMW_BETAS, eps=ADAMW_EPS, weight_decay=0.1)
    bad_lr = _ref_traj([x * 2 for x in _MODERATE_LRS])
    for bad in (bad_beta, bad_eps, bad_wd, bad_lr):
        assert not torch.allclose(rl["trajectory"], bad["trajectory"])
