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

"""Deterministic lr + AdamW parity between RLinf and the reference SFT update logic.

Uses a fake model + fixed fake gradients (no GPU, no data) to verify the two update
logics are strictly identical:

* ``reference_lr`` encodes the reference trainer's inline learning-rate schedule
  verbatim from ``scripts/train_pytorch_new.py`` ``lr_schedule`` (lines 485-498), with
  the parameters from ``src/openpi/training/optimizer.py`` ``CosineDecaySchedule``
  (warmup 1000, peak 2.5e-5, decay 30000, end 0.0): a linear warmup from
  ``peak/(warmup+1)`` to ``peak`` over ``warmup`` steps, then cosine decay to ``end``
  over ``decay-warmup`` steps.
* ``rlinf_lr_sequence`` drives RLinf's REAL ``get_lr_scheduler("openpi_cosine", ...)``
  on a dummy optimizer and reads back the per-step lr.
* ``adamw_trajectory`` runs ``torch.optim.AdamW`` from a fixed init on fixed gradients
  with a given lr schedule, returning the parameter trajectory + final optimizer state,
  so a RLinf-config run and a reference-config run can be compared element-wise.
"""

from __future__ import annotations

import math
from typing import Any

# Reference run config (CosineDecaySchedule defaults; src/openpi/training/optimizer.py).
REF_PEAK_LR = 2.5e-5
REF_WARMUP = 1000
REF_DECAY = 30000
REF_END_LR = 0.0
# AdamW (reference + RLinf both): betas (0.9, 0.95), eps 1e-8, weight_decay 1e-10.
ADAMW_BETAS = (0.9, 0.95)
ADAMW_EPS = 1e-8
ADAMW_WD = 1e-10
# The reference run's logged step-0 lr (output.log): peak/(warmup+1) in float64.
REF_LOGGED_STEP0_LR = 2.4975024975024977e-08


def reference_lr(
    step: int,
    *,
    peak: float = REF_PEAK_LR,
    warmup: int = REF_WARMUP,
    decay: int = REF_DECAY,
    end: float = REF_END_LR,
) -> float:
    """Reference trainer's inline lr schedule, verbatim (train_pytorch_new.py:485-498)."""
    if step < warmup:
        init_lr = peak / (warmup + 1)
        return init_lr + (peak - init_lr) * step / warmup
    progress = min(1.0, (step - warmup) / max(1, decay - warmup))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return end + (peak - end) * cos


def rlinf_lr_sequence(
    steps: list[int],
    *,
    peak: float = REF_PEAK_LR,
    warmup: int = REF_WARMUP,
    total: int = REF_DECAY,
    min_lr: float = REF_END_LR,
) -> dict[int, float]:
    """Drive RLinf's real openpi_cosine scheduler and read the lr at each requested
    step (LambdaLR multiplier x base lr = peak)."""
    import torch

    from rlinf.hybrid_engines.fsdp.utils import get_lr_scheduler

    param = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.AdamW([param], lr=peak)
    sched = get_lr_scheduler(
        "openpi_cosine",
        opt,
        num_warmup_steps=warmup,
        num_training_steps=total,
        min_lr=min_lr,
    )
    want = set(steps)
    out: dict[int, float] = {}
    for s in range(max(steps) + 1):
        if s in want:
            out[s] = opt.param_groups[0]["lr"]
        opt.step()
        sched.step()
    return out


def adamw_trajectory(
    *,
    betas: tuple[float, float] = ADAMW_BETAS,
    eps: float = ADAMW_EPS,
    weight_decay: float = ADAMW_WD,
    n_params: int = 16,
    n_steps: int = 8,
    lrs: list[float] | None = None,
) -> dict[str, Any]:
    """Run AdamW from a fixed init on fixed gradients with a given per-step lr; return
    the parameter trajectory + final (exp_avg, exp_avg_sq). Deterministic — no RNG at
    call time (init/grads are fixed lattices)."""
    import torch

    gen = torch.Generator().manual_seed(0)
    init = torch.linspace(-1.0, 1.0, n_params, dtype=torch.float64)
    grads = [
        torch.sin(torch.arange(n_params, dtype=torch.float64) + step * 0.5)
        for step in range(n_steps)
    ]
    _ = gen  # determinism is via the fixed lattices above, not RNG
    param = torch.nn.Parameter(init.clone())
    opt = torch.optim.AdamW(
        [param], lr=(lrs[0] if lrs else 1e-4), betas=betas, eps=eps,
        weight_decay=weight_decay,
    )
    traj = []
    for step in range(n_steps):
        if lrs is not None:
            for pg in opt.param_groups:
                pg["lr"] = lrs[step]
        opt.zero_grad(set_to_none=False)
        param.grad = grads[step].clone()
        opt.step()
        traj.append(param.detach().clone())
    state = opt.state[param]
    return {
        "trajectory": torch.stack(traj),
        "exp_avg": state["exp_avg"].clone(),
        "exp_avg_sq": state["exp_avg_sq"].clone(),
    }
