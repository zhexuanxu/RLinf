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

"""Exact per-sample BEHAVIOR SFT stream-identity probe (matched 8-rank/8-worker).

Drives the REAL ``BehaviorSftDataset`` helper path (``_select_streaming_chunk``,
``_get_query_indices``, ``_get_fine_grained_task``) for every ``(rank, worker)`` of
the matched training topology (``num_workers=8``, ``seed=42``, ``turning_on_radio``,
``use_skill:false``; ``world_size=8`` for the production rank-folding mode and
``world_size=1`` for the rank-independent mode that matches the reference loader),
by mocking ``dist`` + ``get_worker_info``. For each emitted sample it records the
full identity: ``global_frame_idx, episode_index, frame_index, chunk tuple,
action_query_start/end, action_is_pad, prompt, skip_count`` (``skip_count=0`` for
``use_skill:false`` with ``skill_list=["all"]`` — no frame is skipped).

Outputs under ``--out``:
- ``r19_rlinf_perworker_identity.csv``: first batch (32 frames) of every worker,
  rank-independent mode (human-auditable);
- ``r19_rlinf_identity_hashes.json``: per-worker sha256 of the full first-50-step
  per-sample identity stream (rank-independent), per-``(rank,worker)`` sha256
  (rank-folding), and per-step unique-frame coverage for both modes.

The reference counterpart is ``tests/unit_tests/_ref_stream_dump.py``.

Run (CPU)::

    TMPDIR=/mnt/public/xzxuan/tmp MUJOCO_GL=egl PYTHONPATH=. \\
        python tools/sft_stream_identity_probe.py --out docs/evidence
"""

import argparse
import hashlib
import json
import types

from omegaconf import OmegaConf

NUM_WORKERS = 8
BATCH = 32
SEED = 42
STEPS = 50
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


def _stream(ds, mod, rank, world_size, worker, n_frames):
    """Drive the REAL _select_streaming_chunk for (rank, world_size, worker) and
    enumerate the contiguous walk __getitem__ performs (global_frame_idx, chunk)."""
    mod.dist = _FakeDist(rank, world_size)
    mod.get_worker_info = lambda: types.SimpleNamespace(
        id=worker, num_workers=NUM_WORKERS
    )
    ds._active_chunks = None
    ds.current_streaming_chunk_idx = None
    ds.current_streaming_frame_idx = None
    ds._select_streaming_chunk()
    chunks = ds._active_chunks
    ci, fi = ds.current_streaming_chunk_idx, ds.current_streaming_frame_idx
    out = []
    for _ in range(n_frames):
        if fi >= chunks[ci][1]:
            ci = (ci + 1) % len(chunks)
            fi = chunks[ci][0]
        out.append((fi, tuple(int(x) for x in chunks[ci])))
        fi += 1
    return out


def _identity(ds, frames):
    """Full per-sample identity via the REAL helpers (query window + prompt)."""
    rows = []
    for fi, ch in frames:
        item = ds.hf_dataset[fi]
        ep = int(item["episode_index"].item())
        fr = int(round(float(item["timestamp"].item()) * ds.fps))
        qi, pad = ds._get_query_indices(fi, ep)
        aq = qi["action"]
        is_pad = bool(pad["action_is_pad"].any().item())
        prompt = ds._get_fine_grained_task(item)
        rows.append(
            (fi, ep, fr, ch[0], ch[1], ch[2], aq[0], aq[-1], int(is_pad), prompt, 0)
        )
    return rows


def _hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(("|".join(str(x) for x in r) + "\n").encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp")
    args = ap.parse_args()

    import rlinf.data.datasets.behavior.behavior_sft_dataset as mod

    ds = _build_dataset()
    n_frames = ((STEPS + NUM_WORKERS - 1) // NUM_WORKERS + 1) * BATCH
    print(f"[probe] chunks={len(ds.chunks)} fps={ds.fps} seed={SEED}", flush=True)

    out = {
        "per_worker_rank_independent_sha256": {},
        "per_rank_worker_rank_folding_sha256": {},
    }

    # Rank-independent (world_size=1): per-worker identity stream (matches reference).
    ri_rows = {}
    for w in range(NUM_WORKERS):
        rows = _identity(ds, _stream(ds, mod, 0, 1, w, n_frames))
        ri_rows[w] = rows
        out["per_worker_rank_independent_sha256"][w] = _hash(rows)

    # Rank-folding (world_size=8): per-(rank,worker) identity stream (production).
    rf_rows = {}
    for rank in range(NUM_WORKERS):
        for w in range(NUM_WORKERS):
            rows = _identity(ds, _stream(ds, mod, rank, NUM_WORKERS, w, n_frames))
            rf_rows[(rank, w)] = rows
            out["per_rank_worker_rank_folding_sha256"][f"{rank},{w}"] = _hash(rows)

    def coverage(rows_by_key, key_for_rank):
        vals = set()
        for step in range(STEPS):
            w, k = step % NUM_WORKERS, step // NUM_WORKERS
            frames = {
                (r[1], r[2])
                for rank in range(NUM_WORKERS)
                for r in rows_by_key[key_for_rank(rank, w)][
                    k * BATCH : k * BATCH + BATCH
                ]
            }
            vals.add(len(frames))
        return sorted(vals)

    out["rank_folding_unique_per_step"] = coverage(rf_rows, lambda rank, w: (rank, w))
    out["rank_independent_unique_per_step"] = coverage(ri_rows, lambda rank, w: w)

    cols = (
        "rank,worker,batch_position,global_frame_idx,episode_index,frame_index,"
        "chunk_global_start,chunk_global_end,chunk_local_start,"
        "action_query_start,action_query_end,action_is_pad,prompt,skip_count"
    )
    with open(f"{args.out}/r19_rlinf_perworker_identity.csv", "w") as f:
        f.write(cols + "\n")
        for w in range(NUM_WORKERS):
            for bp, r in enumerate(ri_rows[w][:BATCH]):
                f.write(f"0,{w},{bp}," + ",".join(str(x) for x in r) + "\n")

    with open(f"{args.out}/r19_rlinf_identity_hashes.json", "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"[probe] rank-folding unique/step={out['rank_folding_unique_per_step']} ; "
        f"rank-independent unique/step={out['rank_independent_unique_per_step']}",
        flush=True,
    )
    print(f"[probe] wrote per-worker identity CSV + hashes to {args.out}", flush=True)


if __name__ == "__main__":
    main()
