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

"""Reference rank-0-fanout loader audit with the value-independent (episode, frame) id.

Builds the reference BEHAVIOR dataset (the raw ``BehaviorLeRobotDataset`` underneath
``create_behavior_dataset``) with ``id_only=True`` so each sample is the value-independent
``{episode_index, frame_index=round(timestamp*fps)}`` id taken BEFORE any transform/decode.
A DataLoader with the reference's local batch (= global / world) + ``num_workers`` mirrors
the per-rank micro-batch; rank-0 fanout pulls ``world_size`` successive micro-batches per
global step (256 ids). Emits the same per-rank-per-step id structure as the RLinf id audit.

Run in the reference venv::

    PYTHONPATH=<ref-src>:<ref-scripts>:<rlinf-repo> <ref-venv>/python \
        tests/unit_tests/_ref_loader_audit_ids.py <out_dir> <n_steps> <world_size>
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"  # GLOBAL batch 256
_NUM_WORKERS = 8


def _collate_ids(items):
    return [(int(it["episode_index"]), int(it["frame_index"])) for it in items]


def main(out_dir, n_steps, world_size):
    import os

    sys.path.insert(0, _REF_SRC)
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False, "repo": "reference", "world_size": world_size, "n_steps": n_steps}
    try:
        import torch

        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

        cfg = _config.get_config(_CONFIG)
        global_batch = cfg.batch_size
        per_rank = global_batch // world_size
        base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        run_cfg = dataclasses.replace(
            cfg, data=data, batch_size=per_rank, num_workers=_NUM_WORKERS
        )
        data_config = run_cfg.data.create(run_cfg.assets_dirs, run_cfg.model)

        # Raw BehaviorLeRobotDataset (underneath create_behavior_dataset's TransformedDataset),
        # switched to the id_only fast path -> yields {episode_index, frame_index} ids.
        transformed = _data_loader.create_behavior_dataset(
            data_config, action_horizon=run_cfg.model.action_horizon
        )
        raw = transformed._dataset  # the BehaviorLeRobotDataset
        raw.id_only = True

        loader = torch.utils.data.DataLoader(
            raw,
            batch_size=per_rank,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            collate_fn=_collate_ids,
            drop_last=True,
            persistent_workers=_NUM_WORKERS > 0,
        )
        it = iter(loader)

        per_step_set_hash, per_step_unique, per_step_global_set = [], [], []
        rolling = hashlib.sha256()
        for step in range(n_steps):
            # rank-0 fanout: world_size successive micro-batches per global step
            step_rank_ids = []
            for _r in range(world_size):
                try:
                    batch = next(it)
                except StopIteration:
                    it = iter(loader)  # epoch boundary: re-iterate (cyclic stream)
                    batch = next(it)
                step_rank_ids.append([f"{ep}:{fr}" for (ep, fr) in batch])
            union = sorted({f for rh in step_rank_ids for f in rh})
            sh = hashlib.sha256("".join(union).encode()).hexdigest()
            per_step_set_hash.append(sh)
            per_step_unique.append(len(union))
            per_step_global_set.append(union)
            rolling.update(f"{step}|{sh}".encode())

        result.update(
            ok=True,
            config=_CONFIG,
            global_batch=global_batch,
            per_rank_local_batch=per_rank,
            num_workers=_NUM_WORKERS,
            audited_steps=n_steps,
            per_step_global_set_hash=per_step_set_hash,
            per_step_unique=per_step_unique,
            rolling_hash=rolling.hexdigest(),
        )
        with open(os.path.join(out_dir, "ref_loader_audit.json"), "w", newline="\n") as f:
            json.dump(result, f)
            f.write("\n")
        with open(
            os.path.join(out_dir, "ref_loader_audit_detail.json"), "w", newline="\n"
        ) as f:
            json.dump({"per_step_global_set": per_step_global_set}, f)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1500:]
        with open(os.path.join(out_dir, "ref_loader_audit.json"), "w", newline="\n") as f:
            json.dump(result, f)
            f.write("\n")
    print("REF_LOADER_AUDIT_IDS", json.dumps({"ok": result["ok"], "err": result.get("err")}))


if __name__ == "__main__":
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(out, n, ws)
