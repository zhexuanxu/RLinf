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


def test_run_training_logs_step_used_lr_not_next_step(monkeypatch):
    """The SFT worker logs the LR USED for the just-finished optimizer step.

    The reference trainer sets ``lr = lr_schedule(global_step)`` before the update
    and logs that same value, so global step 0 logs ``peak/(warmup+1)`` = 2.4975e-08.
    This drives the REAL ``FSDPSftWorker.run_training`` ordering with a duck-typed
    ``self`` (so the actual capture-then-``scheduler.step()`` code is exercised) and an
    identity ``all_reduce_dict``. With the pre-fix code (reading
    ``optimizer.param_groups[0]["lr"]`` AFTER ``lr_scheduler.step()``) the emitted
    step-0 LR is the step-1 value and this test fails.
    """
    torch = pytest.importorskip("torch")

    import contextlib
    from types import SimpleNamespace

    import rlinf.workers.sft.fsdp_sft_worker as worker_mod
    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    # The metric reduce-across-ranks is a no-op here (single process).
    monkeypatch.setattr(worker_mod, "all_reduce_dict", lambda d, op=None: d)

    peak, warmup, total = 2.5e-5, 1000, 30000
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=peak)
    # Constructing the LambdaLR sets the optimizer LR to peak/(warmup+1) (step 0).
    scheduler = get_lr_scheduler(
        "openpi_cosine",
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=total,
        min_lr=0.0,
    )

    def _optimizer_step():
        # Mirror FSDPModelManager.optimizer_step's contract: return (grad_norm,
        # lr_list) where lr_list is the per-group LR captured BEFORE the scheduler
        # advances -- i.e. the LR used for this step.
        return torch.tensor(2.0), [g["lr"] for g in optimizer.param_groups]

    fake_self = SimpleNamespace(
        worker_timer=lambda: contextlib.nullcontext(),
        model=SimpleNamespace(train=lambda: None),
        gradient_accumulation=1,
        before_micro_batch=lambda *a, **k: contextlib.nullcontext(),
        data_iter=iter([{}]),
        get_train_model_output=lambda batch: (
            torch.tensor(0.24609375),
            {"loss": 0.24609375},
        ),
        grad_scaler=SimpleNamespace(
            scale=lambda loss: SimpleNamespace(backward=lambda: None)
        ),
        optimizer_step=_optimizer_step,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        global_step=0,
        _data_iter_offset=0,
    )

    metrics = worker_mod.FSDPSftWorker.run_training(fake_self)

    # Logged LR for global step 0 is the step-0 LR (peak/(warmup+1)), NOT step 1's.
    assert metrics["learning_rate"] == pytest.approx(peak / (warmup + 1), rel=1e-9)
    assert metrics["learning_rate"] == pytest.approx(2.4975e-08, rel=1e-4)
    # The scheduler still advanced: the optimizer now holds the step-1 LR.
    assert optimizer.param_groups[0]["lr"] == pytest.approx(
        2 * peak / (warmup + 1), rel=1e-9
    )
