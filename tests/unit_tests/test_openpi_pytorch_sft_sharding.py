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

"""Per-rank data-sharding for the streaming BEHAVIOR SFT dataset.

The streaming dataset ignores the ``idx`` it is handed and partitions its
keyframe chunks internally, so a ``DistributedSampler`` cannot shard it. The
chunk partition therefore folds the distributed ``rank`` into the per-worker
stride. These tests assert each ``(rank, worker)`` pair streams a distinct,
non-overlapping shard whose union covers every chunk — and that the pre-fix,
rank-independent partition would have given every rank identical data.
"""

from __future__ import annotations

from rlinf.models.embodiment.openpi_pytorch.dataconfig import behavior_sft_dataset
from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_dataset import (
    BehaviorSftDataset,
    partition_chunk_indices,
)


class _FakeDist:
    """Stand-in for ``torch.distributed`` reporting a fixed rank / world size."""

    def __init__(self, rank, world_size):
        self._rank = rank
        self._world_size = world_size

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size


def _select_under_rank(monkeypatch, rank, world_size, num_chunks=8, seed=42):
    """Run the real streaming chunk selection as a given distributed rank."""
    monkeypatch.setattr(
        behavior_sft_dataset, "dist", _FakeDist(rank, world_size)
    )
    ds = BehaviorSftDataset.__new__(BehaviorSftDataset)
    # chunks are (start_frame, end_frame, keyframe_start) tuples.
    ds.chunks = [(i * 100, i * 100 + 50, i * 100) for i in range(num_chunks)]
    ds.seed = seed
    ds.current_streaming_chunk_idx = None
    ds._active_chunks = None
    ds._select_streaming_chunk()
    return ds


def test_partition_is_disjoint_and_complete_across_ranks():
    num_chunks = 257
    world_size, num_workers = 8, 1
    shards = [
        set(
            partition_chunk_indices(
                num_chunks,
                rank=rank,
                world_size=world_size,
                worker_id=0,
                num_workers=num_workers,
            )
        )
        for rank in range(world_size)
    ]

    # Pairwise disjoint: no chunk is streamed by two ranks.
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert shards[i].isdisjoint(shards[j])

    # Complete: the union over ranks covers every chunk exactly once.
    union = sorted(idx for shard in shards for idx in shard)
    assert union == list(range(num_chunks))

    # Each rank's first chunk differs -> per-rank first-batch identity differs.
    assert len({min(shard) for shard in shards}) == world_size


def test_partition_grid_of_ranks_and_workers():
    num_chunks = 100
    world_size, num_workers = 2, 3
    shards = [
        set(
            partition_chunk_indices(
                num_chunks,
                rank=rank,
                world_size=world_size,
                worker_id=worker,
                num_workers=num_workers,
            )
        )
        for rank in range(world_size)
        for worker in range(num_workers)
    ]
    flat = [idx for shard in shards for idx in shard]
    # Every (rank, worker) shard is disjoint and the union covers all chunks.
    assert len(flat) == len(set(flat))
    assert set(flat) == set(range(num_chunks))

    # Aggregated per-rank shards are disjoint (rank 0 vs rank 1 see distinct data).
    rank0 = shards[0] | shards[1] | shards[2]
    rank1 = shards[3] | shards[4] | shards[5]
    assert rank0.isdisjoint(rank1)


def test_single_process_partition_is_identity():
    num_chunks = 16
    assert partition_chunk_indices(
        num_chunks, rank=0, world_size=1, worker_id=0, num_workers=1
    ) == list(range(num_chunks))


def test_rank_independent_partition_would_duplicate_data():
    # The pre-fix behavior keyed the partition on worker_id only (no rank), so
    # every rank ran the same worker layout and streamed identical chunks. This
    # documents the bug the rank-aware partition fixes.
    num_chunks = 50

    def rank_independent(worker_id, num_workers):
        return list(range(worker_id, num_chunks, num_workers))

    # Two ranks, single worker each -> identical index sets (the bug).
    assert rank_independent(0, 1) == rank_independent(0, 1)

    # The rank-aware partition instead gives the two ranks disjoint shards.
    fixed_rank0 = set(
        partition_chunk_indices(
            num_chunks, rank=0, world_size=2, worker_id=0, num_workers=1
        )
    )
    fixed_rank1 = set(
        partition_chunk_indices(
            num_chunks, rank=1, world_size=2, worker_id=0, num_workers=1
        )
    )
    assert fixed_rank0.isdisjoint(fixed_rank1)
    assert fixed_rank0 | fixed_rank1 == set(range(num_chunks))


def test_effective_streaming_selection_distinct_per_rank(monkeypatch):
    # Exercise the real ``_select_streaming_chunk`` (the path ``__getitem__``
    # uses) under two simulated ranks; each must stream a disjoint chunk set and
    # start on a different frame -> the first batches differ across ranks.
    ds0 = _select_under_rank(monkeypatch, rank=0, world_size=2)
    ds1 = _select_under_rank(monkeypatch, rank=1, world_size=2)

    chunks0 = set(ds0._active_chunks)
    chunks1 = set(ds1._active_chunks)
    assert chunks0.isdisjoint(chunks1)
    assert chunks0 | chunks1 == set(ds0.chunks)
    # First streamed frame (the first-batch identity) differs across ranks.
    assert ds0.current_streaming_frame_idx != ds1.current_streaming_frame_idx


def test_effective_global_batch_has_no_rank_duplication(monkeypatch):
    # Eight ranks over eight chunks: each rank streams exactly one distinct
    # chunk, so a global batch built one-sample-per-rank is 8 unique samples,
    # not one sample replicated 8x (the DEC-1 failure mode).
    world_size = 8
    first_frames = []
    chunk_sets = []
    for rank in range(world_size):
        ds = _select_under_rank(monkeypatch, rank=rank, world_size=world_size)
        first_frames.append(ds.current_streaming_frame_idx)
        chunk_sets.append(frozenset(ds._active_chunks))

    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert chunk_sets[i].isdisjoint(chunk_sets[j])
    assert len(set(first_frames)) == world_size
