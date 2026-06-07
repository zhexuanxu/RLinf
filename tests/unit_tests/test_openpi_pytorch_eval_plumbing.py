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

"""Seed derivation for the rollout worker's deterministic / paired eval mode.

When ``rollout.eval_deterministic_noise`` is enabled, ``predict()`` injects a
generator seeded by :func:`deterministic_eval_seed` into the flow-matching
sampler. The schedule must be reproducible across runs (so two eval runs of
different checkpoints inject the same per-step noise) and distinct per (rank,
step) within a run (so ranks do not share a schedule).
"""

from __future__ import annotations

import pytest

deterministic_eval_seed = pytest.importorskip(
    "rlinf.workers.rollout.hf.huggingface_worker"
).deterministic_eval_seed


def test_seed_is_reproducible():
    assert deterministic_eval_seed(1234, 3, 17) == deterministic_eval_seed(1234, 3, 17)


def test_seed_is_distinct_per_rank_and_step():
    # No two (rank, step) pairs collide over a realistic eval (8 ranks, many steps);
    # the rank offset (1_000_003) exceeds any per-run step count.
    seeds = {
        deterministic_eval_seed(1234, rank, step)
        for rank in range(8)
        for step in range(2000)
    }
    assert len(seeds) == 8 * 2000


def test_seed_depends_on_base_seed():
    # A different base seed yields a different noise schedule (so the protocol's
    # seed is recorded and varied deliberately, not incidentally).
    assert deterministic_eval_seed(1234, 0, 0) != deterministic_eval_seed(5678, 0, 0)


def _fake_worker(torch, *, deterministic, supports=True, rank=0, seed=1234):
    """A minimal worker exposing only what _next_eval_noise_generator reads."""
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = object.__new__(MultiStepRolloutWorker)
    worker._eval_deterministic_noise = deterministic
    worker._eval_supports_det_noise = supports
    worker._eval_noise_seed = seed
    worker._eval_noise_step = 0
    worker._rank = rank
    worker.device = torch.device("cpu")

    class _Model:
        device = torch.device("cpu")

    worker.hf_model = _Model()
    return worker


def test_predict_injects_no_rng_in_train_mode():
    torch = pytest.importorskip("torch")
    worker = _fake_worker(torch, deterministic=True)
    assert worker._next_eval_noise_generator("train") is None
    assert worker._eval_noise_step == 0


def test_predict_injects_no_rng_when_flag_off():
    torch = pytest.importorskip("torch")
    worker = _fake_worker(torch, deterministic=False)
    assert worker._next_eval_noise_generator("eval") is None


def test_predict_injects_no_rng_for_unsupported_model():
    torch = pytest.importorskip("torch")
    worker = _fake_worker(torch, deterministic=True, supports=False)
    assert worker._next_eval_noise_generator("eval") is None


def test_eval_injects_seeded_generator_advances_and_resets():
    torch = pytest.importorskip("torch")
    worker = _fake_worker(torch, deterministic=True, seed=1234, rank=0)
    g0 = worker._next_eval_noise_generator("eval")
    assert isinstance(g0, torch.Generator)
    assert worker._eval_noise_step == 1
    assert g0.initial_seed() == deterministic_eval_seed(1234, 0, 0)
    g1 = worker._next_eval_noise_generator("eval")
    assert worker._eval_noise_step == 2
    assert g1.initial_seed() == deterministic_eval_seed(1234, 0, 1) != g0.initial_seed()
    # evaluate() resets the counter -> the first seed is reproduced.
    worker._eval_noise_step = 0
    assert worker._next_eval_noise_generator("eval").initial_seed() == g0.initial_seed()
