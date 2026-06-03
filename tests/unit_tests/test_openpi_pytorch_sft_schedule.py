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

"""LR-schedule alignment for openpi_pytorch SFT.

The reference trainer uses a warmup-then-cosine-to-zero schedule (warmup=1000,
cosine decay over 30000 to 0, peak 2.5e-5). This test asserts RLinf's
``cosine`` scheduler reproduces the same shape: linear warmup to the peak at the
warmup boundary and a cosine decay to ~0, matching the reference cosine phase
within tolerance. (The only divergence is the very first warmup step, where the
reference starts at peak/(warmup+1) and RLinf/HF starts at 0 — a <0.1% effect.)
"""

from __future__ import annotations

import math

import pytest


def _reference_lr(step: int, peak: float, warmup: int, decay: int, end: float = 0.0):
    """Reference warmup (init=peak/(warmup+1)) + cosine-to-``end`` schedule."""
    if step < warmup:
        init = peak / (warmup + 1)
        return init + (peak - init) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, decay - warmup))
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def test_sft_lr_schedule_matches_reference():
    torch = pytest.importorskip("torch")

    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    peak, warmup, total = 2.5e-5, 1000, 30000
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=peak)
    sched = get_lr_scheduler(
        "cosine",
        opt,
        num_warmup_steps=warmup,
        num_training_steps=total,
        min_lr=0.0,
    )

    lrs = [opt.param_groups[0]["lr"]]
    opt.step()  # take one optimizer step so scheduler.step() does not warn
    for _ in range(total):
        sched.step()
        lrs.append(opt.param_groups[0]["lr"])

    # Warmup: starts near zero, monotonically increases, hits the peak at the boundary.
    assert lrs[0] == pytest.approx(0.0, abs=1e-9)
    assert lrs[warmup] == pytest.approx(peak, rel=1e-3)
    assert all(lrs[i] <= lrs[i + 1] + 1e-12 for i in range(warmup))

    # Cosine decay phase matches the reference formula closely.
    for step in (warmup, 5000, 15000, 29000):
        assert lrs[step] == pytest.approx(
            _reference_lr(step, peak, warmup, total), rel=0.02, abs=1e-8
        )

    # Decays to ~0 by the end of training.
    assert lrs[total] == pytest.approx(0.0, abs=1e-6)
