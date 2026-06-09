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
    builds_train_loader,
    reference_fanout_micro_batches,
    resolve_loader_mode,
)


def test_builds_train_loader_ownership():
    # per_rank_stream: every rank owns its train loader.
    assert all(builds_train_loader(PER_RANK_STREAM, r) for r in range(8))
    # reference_fanout: only rank 0 owns/builds the single loader; others receive via scatter.
    assert builds_train_loader(REFERENCE_FANOUT, 0) is True
    assert not any(builds_train_loader(REFERENCE_FANOUT, r) for r in range(1, 8))


def _run_init_train_dataloader(rank, loader_mode, val_data_paths=None, pinned=None):
    """Drive FSDPSftWorker._init_train_dataloader on a fake self with a counting builder
    (no FSDP/CUDA), to prove who actually constructs + iterates the TRAIN loader."""
    import types

    from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

    calls = {"build": 0, "iter": 0}

    class _FakeLoader:
        def __iter__(self_inner):
            calls["iter"] += 1
            return iter([])

    def fake_build(data_paths, eval_dataset=False):
        calls["build"] += 1
        return _FakeLoader(), "data_config"

    data = {"train_data_paths": "x", "loader_mode": loader_mode}
    if val_data_paths is not None:
        data["val_data_paths"] = val_data_paths
    if pinned is not None:
        data["pinned_inputs_npz"] = pinned
    fake = types.SimpleNamespace(
        _rank=rank,
        cfg=OmegaConf.create({"data": data}),
        build_dataloader=fake_build,
    )
    FSDPSftWorker._init_train_dataloader(fake)
    return fake, calls


def test_reference_fanout_only_rank0_builds_and_iterates_train_loader():
    w0, c0 = _run_init_train_dataloader(0, "reference_fanout")
    assert c0["build"] == 1 and c0["iter"] == 1
    assert w0.data_loader is not None and w0.data_iter is not None
    assert w0._reference_fanout is True
    for r in (1, 3, 7):
        w, c = _run_init_train_dataloader(r, "reference_fanout")
        assert c["build"] == 0 and c["iter"] == 0, f"rank {r} built/iterated the train loader"
        assert w.data_loader is None and w.data_iter is None and w.data_config is None
        assert w._reference_fanout is True  # flag set everywhere; only rank 0 owns the loader


def test_per_rank_stream_every_rank_builds_train_loader():
    for r in (0, 1, 7):
        w, c = _run_init_train_dataloader(r, "per_rank_stream")
        assert c["build"] == 1 and c["iter"] == 1
        assert w.data_loader is not None and w._reference_fanout is False


def test_pinned_inputs_force_per_rank_topology_over_reference_fanout():
    """The pinned reproducibility loader is rank-sliced (inherently per-rank); when it is
    active the worker MUST force the per-rank topology (every rank builds + iterates its own
    rank-sliced loader, NO rank-0 fanout) even if loader_mode=reference_fanout was requested --
    otherwise rank 0 would scatter world_size micro-batches from its single rank-sliced slice."""
    for loader_mode in ("per_rank_stream", "reference_fanout"):
        for r in (0, 1, 7):
            w, c = _run_init_train_dataloader(r, loader_mode, pinned="/tmp/pinned.npz")
            assert c["build"] == 1 and c["iter"] == 1, (loader_mode, r)
            assert w.data_loader is not None and w.data_iter is not None
            # pinned forces per-rank: the fanout/scatter path is never taken.
            assert w._reference_fanout is False, (loader_mode, r)


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
