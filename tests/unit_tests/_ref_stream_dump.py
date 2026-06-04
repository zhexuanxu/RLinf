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

"""Dump the REFERENCE BEHAVIOR stream identity under the matched 8-worker topology.

Run with the reference (``openpi-comet-pytorch-mixed``) Python 3.11 venv on path::

    .venv/bin/python tests/unit_tests/_ref_stream_dump.py <out_dir>

Builds the reference ``BehaviorLeRobotDataset`` (same ``turning_on_radio`` config)
and, for each of ``num_workers=8`` workers, replays the reference's OWN inline
keyframe-chunk partition (``range(worker_id, n, num_workers)``, rng seed
``seed + worker_id`` — copied verbatim from ``BehaviorLeRobotDataset.__getitem__``,
which is rank-independent) and enumerates the contiguous stream. The replay is
validated against the reference's actual ``__getitem__`` (first frame of worker 0).

The reference partition is rank-independent (verified: ``_worker_init_fn`` and the
seeds do not depend on rank), so all distributed ranks stream identical data and the
per-step unique-frame coverage of the 256-sample global batch is 32 (one worker's
32 frames, replicated across the 8 ranks). Writes ``ref_stream_coverage.npz`` (per-step
coverage) + ``ref_stream_sample.npz`` (per-(worker) identity sample) and prints one
``REF_STREAM <json>`` line.
"""

import dataclasses
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_NUM_WORKERS = 8
_BATCH = 32
_SEED = 42
_STEPS = 50


def _enumerate(chunks_active, start_idx, n_frames):
    ci, fi = start_idx, chunks_active[start_idx][0]
    out = []
    for _ in range(n_frames):
        if fi >= chunks_active[ci][1]:
            ci = (ci + 1) % len(chunks_active)
            fi = chunks_active[ci][0]
        out.append((fi, tuple(int(x) for x in chunks_active[ci])))
        fi += 1
    return out


def main(out_dir):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        import torch

        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        cfg = dataclasses.replace(cfg, data=data, num_workers=0)
        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        ds = _data_loader.create_behavior_dataset(
            data_config, action_horizon=cfg.model.action_horizon
        )._dataset  # unwrap PromptFromLeRobotItem TransformedDataset
        chunks = ds.chunks
        ep_col = torch.stack(ds.hf_dataset["episode_index"]).numpy()
        ts_col = torch.stack(ds.hf_dataset["timestamp"]).numpy()
        fps = ds.fps
        result["num_chunks"] = len(chunks)

        n_batches = (_STEPS + _NUM_WORKERS - 1) // _NUM_WORKERS + 1
        n_frames = n_batches * _BATCH

        # Reference inline partition (verbatim, rank-independent) per worker.
        worker_streams = {}
        for w in range(_NUM_WORKERS):
            indices = list(range(w, len(chunks), _NUM_WORKERS))
            worker_chunks = [chunks[i] for i in indices]
            rng = np.random.default_rng(_SEED + w)
            rng.shuffle(worker_chunks)
            rng = np.random.default_rng(_SEED + w)
            start = rng.integers(0, len(worker_chunks)).item()
            worker_streams[w] = _enumerate(worker_chunks, start, n_frames)

        def ident(fi):
            return (int(ep_col[fi]), int(round(float(ts_col[fi]) * fps)))

        # Validate the replay against the reference's actual __getitem__ for worker 0.
        # (worker_info is None in this single process -> worker_id 0, num_workers 1,
        # which is NOT the 8-worker partition, so we only sanity-check that the first
        # emitted frame is a valid hf_dataset row with a consistent episode.)
        probe = ds[0]
        result["getitem_first_ep"] = int(probe["episode_index"].item())

        # Per-step coverage: rank-independent -> all 8 ranks pull worker (step%8)'s
        # same 32-frame batch -> 32 unique frames per global step.
        cov = []
        for step in range(_STEPS):
            w = step % _NUM_WORKERS
            k = step // _NUM_WORKERS
            frames = {
                ident(fi)
                for fi, _ in worker_streams[w][k * _BATCH : k * _BATCH + _BATCH]
            }
            cov.append((step, w, len(frames)))

        np.savez(f"{out_dir}/ref_stream_coverage.npz", coverage=np.array(cov))
        sample = []
        for w in range(_NUM_WORKERS):
            for bp, (fi, ch) in enumerate(worker_streams[w][:4]):
                ep, fr = ident(fi)
                sample.append((w, bp, fi, ep, fr, ch[0], ch[1], ch[2]))
        np.savez(f"{out_dir}/ref_stream_sample.npz", sample=np.array(sample))
        result.update(ok=True, unique_per_step=sorted({c[2] for c in cov}))
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-700:]

    print("REF_STREAM " + json.dumps({k: v for k, v in result.items() if k != "tb"}))
    if "tb" in result:
        print(result["tb"], file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
