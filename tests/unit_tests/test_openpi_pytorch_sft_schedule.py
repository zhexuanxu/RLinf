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

The reference trainer (`openpi-comet-pytorch-mixed`) uses a warmup-then-cosine
schedule where warmup starts at ``peak / (warmup_steps + 1)`` (NOT 0), ramps
linearly to the peak at ``warmup_steps``, then cosine-decays to ``min_lr`` over
``total_training_steps``. RLinf's ``openpi_cosine`` scheduler mode reproduces
this exactly; the SFT config (`behavior_pi05_vla.yaml`) selects it. This test
asserts the exact initial LR (``peak/(warmup+1) = 2.4975e-08`` for peak 2.5e-5)
and exact agreement with the reference formula across warmup and decay.
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


def test_openpi_cosine_schedule_matches_reference_exactly():
    torch = pytest.importorskip("torch")

    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    peak, warmup, total = 2.5e-5, 1000, 30000
    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=peak)
    sched = get_lr_scheduler(
        "openpi_cosine",
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

    # Exact initial LR: peak/(warmup+1), NOT 0.
    assert lrs[0] == pytest.approx(peak / (warmup + 1), rel=1e-9)
    assert lrs[0] == pytest.approx(2.4975e-08, rel=1e-4)

    # Warmup ramps linearly to exactly the peak at the boundary, monotonically.
    assert lrs[warmup] == pytest.approx(peak, rel=1e-9)
    assert all(lrs[i] < lrs[i + 1] for i in range(warmup))

    # Exact agreement with the reference formula across warmup AND decay.
    for step in (0, 1, 50, 100, 500, 999, 1000, 5000, 15000, 29000):
        assert lrs[step] == pytest.approx(
            _reference_lr(step, peak, warmup, total), rel=1e-9, abs=1e-15
        )

    # Decays to exactly min_lr (0) at the end of training.
    assert lrs[total] == pytest.approx(0.0, abs=1e-12)


def test_default_hf_cosine_starts_at_zero():
    # Guard the distinction: the plain HF ``cosine`` mode starts warmup at 0,
    # which is why the SFT path uses ``openpi_cosine`` instead.
    torch = pytest.importorskip("torch")

    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    peak = 2.5e-5
    opt = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=peak)
    sched = get_lr_scheduler(
        "cosine", opt, num_warmup_steps=1000, num_training_steps=30000, min_lr=0.0
    )
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)
    del sched
