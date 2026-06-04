#!/usr/bin/env python
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

"""Matched 8-rank / 8-worker BEHAVIOR SFT stream-identity probe (single process).

Drives the REAL ``BehaviorSftDataset._select_streaming_chunk`` for every
``(rank, worker)`` pair of the matched training topology (``world_size=8``,
``num_workers=8``, ``seed=42``, ``turning_on_radio``, ``use_skill:false``) by
mocking ``torch.distributed`` and ``get_worker_info`` (the only two inputs that
define the per-consumer partition), then enumerates each pair's contiguous
keyframe-chunk stream and maps each emitted ``global_frame_idx`` to
``(episode_index, frame_index, chunk_tuple)`` via the dataset metadata.

It reports, for the first ``--steps`` global optimizer steps, the per-step
**unique (episode, frame) coverage** of the global batch (8 ranks x 32 samples
from worker ``step % 8``). A PyTorch ``DataLoader`` over this map-style streaming
dataset assigns batch ``i`` to worker ``i % num_workers``; batch ``i`` is that
worker's ``(i // num_workers)``-th 32-frame run.

Writes per-step coverage + a per-(rank,worker) sample table under ``--out``.

Run (CPU)::

    TMPDIR=/mnt/public/xzxuan/tmp MUJOCO_GL=egl PYTHONPATH=. \\
        python tools/sft_stream_identity_probe.py --out docs/evidence
"""

import argparse
import types

from omegaconf import OmegaConf

WORLD_SIZE = 8
NUM_WORKERS = 8
BATCH = 32
SEED = 42
DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
ASSETS = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"


class _FakeDist:
    def __init__(self, rank, world_size):
        self._r, self._w = rank, world_size

    def is_available(self):
        return True

    def is_initialized(self):
        return True

    def get_rank(self):
        return self._r

    def get_world_size(self):
        return self._w


def _build_dataset():
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi_pytorch",
                    "num_action_chunks": 32,
                    "openpi": {
                        "assets_dir": ASSETS,
                        "asset_id": "behavior-1k/2025-challenge-demos",
                        "model_action_dim": 32,
                        "max_token_len": 200,
                    },
                },
                "micro_batch_size": 32,
                "eval_batch_size": 1,
                "seed": SEED,
            },
            "data": {
                "train_data_paths": DATA,
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": False,
            },
        }
    )
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    loader, _ = build_behavior_sft_dataloader(cfg, 1, 0, DATA)
    return loader.torch_loader.dataset._dataset


def _enumerate_stream(ds, mod, rank, worker, n_frames):
    """Drive the real _select_streaming_chunk for (rank, worker) and enumerate the
    first n_frames emitted (global_frame_idx, chunk_tuple) -- the deterministic
    contiguous walk that __getitem__ performs."""
    mod.dist = _FakeDist(rank, WORLD_SIZE)
    mod.get_worker_info = lambda: types.SimpleNamespace(
        id=worker, num_workers=NUM_WORKERS
    )
    ds.seed = SEED
    ds._active_chunks = None
    ds.current_streaming_chunk_idx = None
    ds.current_streaming_frame_idx = None
    ds._select_streaming_chunk()  # the REAL partition + per-(rank,worker) shuffle
    chunks = ds._active_chunks
    ci = ds.current_streaming_chunk_idx
    fi = ds.current_streaming_frame_idx
    out = []
    for _ in range(n_frames):
        if fi >= chunks[ci][1]:
            ci = (ci + 1) % len(chunks)
            fi = chunks[ci][0]
        out.append((fi, tuple(int(x) for x in chunks[ci])))
        fi += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp")
    args = ap.parse_args()

    import rlinf.data.datasets.behavior.behavior_sft_dataset as mod

    ds = _build_dataset()
    import torch as th

    ep_col = th.stack(ds.hf_dataset["episode_index"]).numpy()
    ts_col = th.stack(ds.hf_dataset["timestamp"]).numpy()
    fps = ds.fps
    print(
        f"[probe] keyframe chunks={len(ds.chunks)} ; fps={fps} ; seed={SEED}",
        flush=True,
    )

    n_batches = (args.steps + NUM_WORKERS - 1) // NUM_WORKERS + 1
    n_frames = n_batches * BATCH

    # streams[(rank, worker)] = list of (global_frame_idx, chunk_tuple)
    streams = {}
    for rank in range(WORLD_SIZE):
        for worker in range(NUM_WORKERS):
            streams[(rank, worker)] = _enumerate_stream(ds, mod, rank, worker, n_frames)

    def ident(fi):
        return (int(ep_col[fi]), int(round(float(ts_col[fi]) * fps)))

    # Per-step unique (episode, frame) coverage of the 256-sample global batch.
    cov_rows = []
    for step in range(args.steps):
        w = step % NUM_WORKERS
        k = step // NUM_WORKERS
        frames = set()
        for rank in range(WORLD_SIZE):
            for fi, _ in streams[(rank, w)][k * BATCH : k * BATCH + BATCH]:
                frames.add(ident(fi))
        cov_rows.append((step, w, len(frames)))

    cov_path = f"{args.out}/r18_rlinf_stream_coverage.csv"
    with open(cov_path, "w") as f:
        f.write("global_step,worker_id,rlinf_unique_frames_per_step\n")
        for s, w, u in cov_rows:
            f.write(f"{s},{w},{u}\n")

    # Per-(rank,worker) first-frame identity sample (chunk tuple + episode/frame).
    samp_path = f"{args.out}/r18_rlinf_stream_sample.csv"
    with open(samp_path, "w") as f:
        f.write(
            "rank,worker,batch_position,global_frame_idx,episode_index,frame_index,chunk_global_start,chunk_global_end,chunk_local_start\n"
        )
        for rank in range(WORLD_SIZE):
            for worker in range(NUM_WORKERS):
                for bp, (fi, ch) in enumerate(streams[(rank, worker)][:4]):
                    ep, fr = ident(fi)
                    f.write(
                        f"{rank},{worker},{bp},{fi},{ep},{fr},{ch[0]},{ch[1]},{ch[2]}\n"
                    )

    uniq = sorted({u for _, _, u in cov_rows})
    print(
        f"[probe] per-step unique-frame coverage values = {uniq}  (expect 256 for rank-folding)",
        flush=True,
    )
    print(f"[probe] wrote {cov_path} and {samp_path}", flush=True)


if __name__ == "__main__":
    main()
