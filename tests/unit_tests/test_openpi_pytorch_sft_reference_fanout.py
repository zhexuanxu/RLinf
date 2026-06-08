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

"""Tests for the config-selectable BEHAVIOR SFT loading strategy and the rank-0 fanout
scatter logic that reproduces the reference openpi-comet centralized data pipeline.

Covers the pure control flow (pull order: other ranks first, rank 0 last; receive path)
and an end-to-end gloo CPU scatter so each rank gets exactly the reference-assigned
micro-batches with no duplication/loss. The full bit-identical-30k data proof is the
committed loader-multiset audit (reference_fanout mode)."""

from __future__ import annotations

import pathlib
import socket
import sys

import pytest

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))

from omegaconf import OmegaConf  # noqa: E402

from rlinf.data.datasets.behavior.behavior_sft_data_loader import (  # noqa: E402
    PER_RANK_STREAM,
    REFERENCE_FANOUT,
    reference_fanout_micro_batches,
    resolve_loader_mode,
)


def test_resolve_loader_mode_default_configured_and_validated():
    assert resolve_loader_mode(OmegaConf.create({"data": {}})) == PER_RANK_STREAM
    assert (
        resolve_loader_mode(OmegaConf.create({"data": {"loader_mode": "reference_fanout"}}))
        == REFERENCE_FANOUT
    )
    # explicit override beats config
    assert resolve_loader_mode(OmegaConf.create({"data": {}}), REFERENCE_FANOUT) == REFERENCE_FANOUT
    with pytest.raises(ValueError):
        resolve_loader_mode(OmegaConf.create({"data": {"loader_mode": "nope"}}))


def test_fanout_rank0_pulls_others_first_then_itself_last():
    """Rank 0 pulls grad_accum micro-batches for ranks 1..W-1 (in order) then itself LAST,
    matching the reference's per-rank assignment (not merely the global multiset)."""
    world_size, grad_accum = 4, 2
    sent = {}
    src = iter(range(1000))
    own = reference_fanout_micro_batches(
        src, 0, world_size, grad_accum, lambda b, dst: sent.__setitem__(dst, list(b)), None
    )
    assert sent[1] == [0, 1]
    assert sent[2] == [2, 3]
    assert sent[3] == [4, 5]
    assert own == [6, 7]  # rank 0 LAST
    # exactly world_size*grad_accum micro-batches consumed for the step, no dup/loss
    assert sorted(sent[1] + sent[2] + sent[3] + own) == list(range(world_size * grad_accum))


def test_fanout_nonzero_rank_receives_from_rank0():
    got = reference_fanout_micro_batches(
        None, 2, 4, 3, None, lambda src: [("recv_from", src)]
    )
    assert got == [("recv_from", 0)]


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _gloo_worker(rank, world_size, grad_accum, port, q):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )
    # Only rank 0 owns the (fake) single loader; others receive via the scatter.
    data_iter = iter([("b", i) for i in range(world_size * grad_accum)]) if rank == 0 else None

    def send_fn(batches, dst):
        dist.send_object_list([batches], dst=dst)

    def recv_fn(src):
        obj = [None]
        dist.recv_object_list(obj, src=src)
        return obj[0]

    my = reference_fanout_micro_batches(
        data_iter, rank, world_size, grad_accum, send_fn, recv_fn
    )
    q.put((rank, my))
    dist.barrier()
    dist.destroy_process_group()


def test_fanout_gloo_scatter_matches_single_loader_pull():
    """End-to-end CPU gloo: rank 0's single-loader pulls are scattered so each rank gets
    exactly the reference-assigned micro-batches and the union is the full step (no dup/loss)."""
    import torch.multiprocessing as mp

    world_size, grad_accum = 4, 2
    port = _free_port()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(target=_gloo_worker, args=(r, world_size, grad_accum, port, q))
        for r in range(world_size)
    ]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(world_size):
            rank, my = q.get(timeout=70)
            results[rank] = my
    finally:
        for p in procs:
            p.join(timeout=15)

    # rank r in 1..W-1 receives pulls [(r-1)*K .. r*K-1]; rank 0 keeps [(W-1)*K .. W*K-1].
    assert results[1] == [("b", 0), ("b", 1)]
    assert results[2] == [("b", 2), ("b", 3)]
    assert results[3] == [("b", 4), ("b", 5)]
    assert results[0] == [("b", 6), ("b", 7)]
    union = sorted(sum(results.values(), []))
    assert union == [("b", i) for i in range(world_size * grad_accum)]
