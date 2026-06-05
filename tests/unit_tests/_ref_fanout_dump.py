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

"""Mirror the REFERENCE trainer's rank-0 batch FANOUT to measure its effective batch.

Codex R33 review: the R32/R33 reference dump had every rank independently build
``create_behavior_data_loader_torch`` (which, with the same seed, replicates). But the
PRODUCTION reference trainer ``scripts/train_pytorch_new.py`` builds the loader ONLY on
rank 0 (line 866) and, each step, pulls ``world_size * gradient_accumulate`` SUCCESSIVE
``next(data_iter)`` micro-batches and fans them out -- one (block) to each rank (lines
952-961). So each rank trains on a DIFFERENT micro-batch -> RANK-DISJOINT -> effective
batch = world_size * local_batch = 256.

This single-process script mirrors that rank-0 fanout: it builds the reference loader at
the per-rank effective batch (``config.batch_size // world_size`` = 32) and, for each of
``n_steps`` global steps, pulls ``world_size`` successive micro-batches (the 8 ranks'
batches), hashing every frame. It then reports the UNIQUE frames per global step (256 if
the fanout is rank-disjoint, as the code says) -- measuring the reference's effective
per-step batch rather than assuming it.

    .venv/bin/python tests/unit_tests/_ref_fanout_dump.py <out_dir> <n_steps> <world_size>
"""

import dataclasses
import hashlib
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"  # GLOBAL batch 256
_NUM_WORKERS = 8


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _frame_hashes(observation, actions):
    state, act = _np(observation.state), _np(actions)
    tp = _np(getattr(observation, "tokenized_prompt", np.zeros((state.shape[0], 1))))
    out = []
    for i in range(state.shape[0]):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(state[i]).tobytes())
        h.update(np.ascontiguousarray(act[i]).tobytes())
        h.update(np.ascontiguousarray(tp[i]).tobytes())
        out.append(h.hexdigest()[:16])
    return out


def main(out_dir, n_steps, world_size):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

        cfg = _config.get_config(_CONFIG)
        global_batch = cfg.batch_size
        per_rank = global_batch // world_size  # the production effective_batch_size
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        # Single-process mirror of rank 0: the loader yields per-rank (32-frame) micro-batches.
        run_cfg = dataclasses.replace(
            cfg, data=data, batch_size=per_rank, num_workers=_NUM_WORKERS
        )
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))

        per_step_unique, per_step_rank_hashes = [], []
        for _ in range(n_steps):
            # Rank 0 pulls `world_size` successive micro-batches per step (one per rank).
            step_rank_hashes = []
            for _r in range(world_size):
                obs, actions = next(it)
                step_rank_hashes.append(_frame_hashes(obs, actions))
            uniq = len({f for rh in step_rank_hashes for f in rh})
            per_step_unique.append(uniq)
            per_step_rank_hashes.append(step_rank_hashes)

        result.update(
            ok=True,
            config=_CONFIG,
            global_batch=global_batch,
            world_size=world_size,
            per_rank_local_batch=per_rank,
            num_workers=_NUM_WORKERS,
            n_steps=n_steps,
            unique_frames_per_global_step={
                "mean": float(np.mean(per_step_unique)),
                "min": int(min(per_step_unique)),
                "max": int(max(per_step_unique)),
            },
            verdict="RANK-DISJOINT (effective batch = world_size*per_rank)"
            if np.mean(per_step_unique) > per_rank + 8
            else "RANK-REPLICATED (effective batch = per_rank)",
            per_rank_per_step_frame_hashes=per_step_rank_hashes,
        )
        with open(f"{out_dir}/ref_fanout_hashes.json", "w", newline="\n") as f:
            json.dump(result, f)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1000:]

    print(
        "REF_FANOUT "
        + json.dumps(
            {
                k: result[k]
                for k in ("ok", "err", "unique_frames_per_global_step", "verdict")
                if k in result
            }
        )
    )
    if "tb" in result and not result["ok"]:
        print(result["tb"])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(out, n, ws)
