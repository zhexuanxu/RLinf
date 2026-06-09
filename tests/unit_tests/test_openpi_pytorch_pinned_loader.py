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

"""CPU tests for the reproducibility-only pinned BEHAVIOR SFT input path.

These cover (a) that the pinned loader shards the reference rank-0-fanout
sequence so rank ``r`` consumes the chunk at flat index ``s * world_size + r``
and the matching ``noise``/``time`` slice ``[r * micro : (r + 1) * micro]``, and
(b) that the SFT forward threads explicit ``noise``/``time`` through to
``compute_loss`` while staying a no-op for normal ``(observation, actions)``
batches.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.data.datasets.behavior import build_pinned_behavior_sft_dataloader
from rlinf.models.embodiment.openpi_pytorch.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _write_pinned_npzs(tmp_path, world_size, micro, n_steps, ah=4, ad=4, h=4, w=4):
    """Synthetic pinned NPZs whose values encode the flat index / global slot.

    ``actions[i]`` is filled with the flat index ``i`` (= ``s * world_size + c``)
    and ``noise[s, j]`` with ``s * 1000 + j``, so the per-rank slicing can be
    asserted exactly.
    """
    n = n_steps * world_size
    batches = {}
    for k in _IMG:
        img = np.zeros((n, micro, h, w, 3), dtype=np.float32)
        for i in range(n):
            img[i] = float(i)
        batches[f"image__{k}"] = img
        batches[f"image_mask__{k}"] = np.ones((n, micro), dtype=bool)
    batches["state"] = np.stack(
        [np.full((micro, 8), float(i), np.float32) for i in range(n)]
    )
    batches["tokenized_prompt"] = np.zeros((n, micro, 5), dtype=np.int64)
    batches["tokenized_prompt_mask"] = np.ones((n, micro, 5), dtype=bool)
    batches["actions"] = np.stack(
        [np.full((micro, ah, ad), float(i), np.float32) for i in range(n)]
    )
    batches_path = str(tmp_path / "ref_pinned_batches.npz")
    np.savez(batches_path, **batches)

    noise = np.zeros((n_steps, world_size * micro, ah, ad), np.float32)
    time = np.zeros((n_steps, world_size * micro), np.float32)
    for s in range(n_steps):
        for j in range(world_size * micro):
            noise[s, j] = float(s * 1000 + j)
            time[s, j] = float(s) + j / 1000.0
    nt_path = str(tmp_path / "ref_pinned_noise_time.npz")
    np.savez(nt_path, noise=noise, time=time)
    return batches_path, nt_path


def _cfg(batches_path, nt_path, micro):
    return OmegaConf.create(
        {
            "data": {
                "pinned_inputs_npz": batches_path,
                "pinned_noise_time_npz": nt_path,
                "tasks": ["t"],
            },
            "actor": {"micro_batch_size": micro},
        }
    )


def test_pinned_loader_shards_chunk_and_noise_per_rank(tmp_path):
    world_size, micro, n_steps = 4, 2, 3
    batches_path, nt_path = _write_pinned_npzs(tmp_path, world_size, micro, n_steps)
    cfg = _cfg(batches_path, nt_path, micro)

    seen_chunks = []
    for rank in range(world_size):
        loader, _ = build_pinned_behavior_sft_dataloader(cfg, world_size, rank)
        assert len(loader) == n_steps
        steps = list(loader)
        assert len(steps) == n_steps
        for s, batch in enumerate(steps):
            flat_index = s * world_size + rank
            seen_chunks.append(flat_index)
            # the chunk the reference fanout assigned to position `rank`
            assert torch.allclose(
                batch["actions"], torch.full_like(batch["actions"], float(flat_index))
            )
            assert isinstance(batch["observation"], Observation)
            # the matching noise/time slice [rank*micro : (rank+1)*micro]
            lo = rank * micro
            expect_noise = torch.arange(
                s * 1000 + lo, s * 1000 + lo + micro, dtype=torch.float32
            )
            assert torch.allclose(batch["noise"][:, 0, 0], expect_noise)
            assert batch["noise"].shape[0] == micro
            assert torch.allclose(
                batch["time"], (s + torch.arange(lo, lo + micro) / 1000.0).float()
            )

    # every global chunk consumed exactly once across all ranks/steps (disjoint union)
    assert sorted(seen_chunks) == list(range(n_steps * world_size))


def test_pinned_loader_rejects_mismatched_global_batch(tmp_path):
    # noise/time dumped for world_size*micro = 8, but the run claims world_size=2 -> 4
    batches_path, nt_path = _write_pinned_npzs(
        tmp_path, world_size=4, micro=2, n_steps=3
    )
    cfg = _cfg(batches_path, nt_path, micro=2)
    try:
        build_pinned_behavior_sft_dataloader(cfg, world_size=2, rank=0)
    except ValueError as e:
        assert "global batch" in str(e)
    else:
        raise AssertionError("expected a ValueError on a noise/world-size mismatch")


def test_unpack_sft_batch_extracts_optional_noise_time():
    obs, actions = object(), torch.zeros(2, 3, 4)
    o, a, n, t = OpenPiPytorchActionModel._unpack_sft_batch((obs, actions))
    assert o is obs and a is actions and n is None and t is None

    noise, time = torch.ones(2, 3, 4), torch.full((2,), 0.5)
    o, a, n, t = OpenPiPytorchActionModel._unpack_sft_batch(
        {"observation": obs, "actions": actions, "noise": noise, "time": time}
    )
    assert n is noise and t is time

    o, a, n, t = OpenPiPytorchActionModel._unpack_sft_batch(
        {"observation": obs, "actions": actions}
    )
    assert n is None and t is None


class _CaptureInner(nn.Module):
    action_dim = 4

    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1))
        self.captured = {}

    def compute_loss(
        self, observation, actions, *, train=False, rng=None, noise=None, time=None
    ):
        self.captured = {"noise": noise, "time": time, "train": train}
        return torch.zeros(actions.shape[0], actions.shape[1])


def _bare_action_model(inner):
    m = object.__new__(OpenPiPytorchActionModel)
    nn.Module.__init__(m)
    m.model = inner
    m.processor = None
    m.action_env_dim = inner.action_dim
    # focus the test on the noise/time passthrough, not device/normalization plumbing
    m._observation_to_device = lambda obs: obs
    m._actions_to_device = lambda act: act
    return m


def test_sft_forward_passes_pinned_noise_time_and_is_a_noop_otherwise():
    inner = _CaptureInner()
    m = _bare_action_model(inner)
    actions = torch.zeros(2, 3, 4)
    noise, time = torch.ones(2, 3, 4), torch.full((2,), 0.5)

    m.sft_forward(
        {"observation": object(), "actions": actions, "noise": noise, "time": time}
    )
    assert inner.captured["noise"] is not None
    assert torch.allclose(inner.captured["noise"], noise)
    assert torch.allclose(inner.captured["time"], time)

    inner.captured = {}
    m.sft_forward((object(), actions))
    assert inner.captured["noise"] is None and inner.captured["time"] is None
