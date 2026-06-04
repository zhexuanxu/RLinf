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

"""Exact per-sample REFERENCE BEHAVIOR stream-identity dump (matched 8-worker).

Run with the reference (``openpi-comet-pytorch-mixed``) Python 3.11 venv on path::

    .venv/bin/python tests/unit_tests/_ref_stream_dump.py <out_dir>

Builds the reference ``BehaviorLeRobotDataset`` and, for each of ``num_workers=8``
workers, drives the REAL chunk selection by monkeypatching the reference module's
``get_worker_info`` to ``id=<worker>, num_workers=8`` and calling the actual
``__getitem__`` once (which runs the reference's own inline partition and populates
``self._active_chunks``). It then enumerates that real partition and records the full
per-sample identity via the reference's REAL helpers ``_get_query_indices`` and
``_get_fine_grained_task``: ``global_frame_idx, episode_index, frame_index, chunk
tuple, action_query_start/end, action_is_pad, prompt, skip_count`` (the same schema as
``tools/sft_stream_identity_probe.py``).

Writes ``ref_perworker_identity.csv`` (first batch of every worker) +
``ref_identity_hashes.json`` (per-worker sha256 of the full first-50-step stream) and
prints one ``REF_STREAM <json>`` line.
"""

import dataclasses
import hashlib
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


def _hash(rows):
    h = hashlib.sha256()
    for r in rows:
        h.update(("|".join(str(x) for x in r) + "\n").encode())
    return h.hexdigest()


def main(out_dir):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import behavior.learning.datas.dataset as ref_ds_mod
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

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
        )._dataset
        result["num_chunks"] = len(ds.chunks)
        n_frames = ((_STEPS + _NUM_WORKERS - 1) // _NUM_WORKERS + 1) * _BATCH

        def identity(frames):
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
                    (
                        fi,
                        ep,
                        fr,
                        ch[0],
                        ch[1],
                        ch[2],
                        aq[0],
                        aq[-1],
                        int(is_pad),
                        prompt,
                        0,
                    )
                )
            return rows

        per_worker, hashes = {}, {}
        for w in range(_NUM_WORKERS):
            # Drive the REAL reference selection for worker w.
            ref_ds_mod.get_worker_info = lambda w=w: type(
                "WI", (), {"id": w, "num_workers": _NUM_WORKERS}
            )()
            ds._active_chunks = None
            ds.current_streaming_chunk_idx = None
            ds.current_streaming_frame_idx = None
            # The real __getitem__ runs the reference's inline partition (populating
            # self._active_chunks) BEFORE the per-frame video decode; the decode may
            # fail in this standalone (no-DataLoader) context, but _active_chunks is
            # already set by the real selection path at that point.
            try:
                ds[0]
            except Exception:
                pass
            assert ds._active_chunks is not None, (
                "reference selection did not set _active_chunks"
            )
            active = [tuple(int(x) for x in c) for c in ds._active_chunks]
            rng = np.random.default_rng(_SEED + w)
            ci = rng.integers(0, len(active)).item()
            fi = active[ci][0]
            frames = []
            for _ in range(n_frames):
                if fi >= active[ci][1]:
                    ci = (ci + 1) % len(active)
                    fi = active[ci][0]
                frames.append((fi, active[ci]))
                fi += 1
            rows = identity(frames)
            per_worker[w] = rows
            hashes[str(w)] = _hash(rows)

        cols = (
            "rank,worker,batch_position,global_frame_idx,episode_index,frame_index,"
            "chunk_global_start,chunk_global_end,chunk_local_start,"
            "action_query_start,action_query_end,action_is_pad,prompt,skip_count"
        )
        with open(f"{out_dir}/ref_perworker_identity.csv", "w") as f:
            f.write(cols + "\n")
            for w in range(_NUM_WORKERS):
                for bp, r in enumerate(per_worker[w][:_BATCH]):
                    f.write(f"0,{w},{bp}," + ",".join(str(x) for x in r) + "\n")
        with open(f"{out_dir}/ref_identity_hashes.json", "w") as f:
            json.dump({"per_worker_sha256": hashes}, f, indent=2)
        result.update(ok=True, per_worker_sha256={k: v[:12] for k, v in hashes.items()})
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-800:]

    print("REF_STREAM " + json.dumps({k: v for k, v in result.items() if k != "tb"}))
    if "tb" in result:
        print(result["tb"], file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
