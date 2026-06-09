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

"""B3 equivalence proof: the raw ``id_only`` DataLoader shortcut used by the reference id
audit emits the SAME frame-id stream as the PRODUCTION ``create_behavior_data_loader_torch``
DataLoader settings.

The reference's production wrapper (``src/openpi/training/data_loader.py:303-390``) builds a
``torch.utils.data.DataLoader`` with ``shuffle=True``, ``worker_init_fn=_worker_init_fn``,
``generator=manual_seed(seed)``, ``persistent_workers=True``, ``drop_last=True`` over the
SAME ``create_behavior_dataset`` (``shuffle=True, seed=42``). The streaming dataset ignores
``idx``, so the RandomSampler/generator only permute (ignored) idx values and ``worker_init_fn``
only sets JAX env vars — none touch the streamed content. This script proves that by running
the id_only dataset through (a) the EXACT production DataLoader kwargs and (b) the shortcut
(``shuffle=False``) and asserting the fanout id streams are byte-identical.

Run in the reference venv::

    PYTHONPATH=<ref-src>:<rlinf-repo> <ref-venv>/python \
        tests/unit_tests/_ref_loader_equiv_ids.py <out_dir> <n_steps> <world_size>
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_NUM_WORKERS = 8


def _collate_ids(items):
    return [(int(it["episode_index"]), int(it["frame_index"])) for it in items]


def _build_id_dataset(world_size):
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(_CONFIG)
    per_rank = cfg.batch_size // world_size
    base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
    assets = _config.AssetsConfig(
        assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
    )
    data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
    run_cfg = dataclasses.replace(cfg, data=data, batch_size=per_rank, num_workers=_NUM_WORKERS)
    data_config = run_cfg.data.create(run_cfg.assets_dirs, run_cfg.model)
    transformed = _data_loader.create_behavior_dataset(
        data_config, action_horizon=run_cfg.model.action_horizon
    )
    raw = transformed._dataset
    raw.id_only = True
    return raw, per_rank, cfg.seed


def _fanout_rolling(loader, n_steps, world_size):
    """Rolling hash over the rank-0-fanout per-step global id sets."""
    it = iter(loader)
    roll = hashlib.sha256()
    for step in range(n_steps):
        ids = []
        for _r in range(world_size):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader)
                b = next(it)
            ids.extend(f"{ep}:{fr}" for (ep, fr) in b)
        payload = json.dumps(sorted(set(ids)), separators=(",", ":"))
        roll.update(f"{step}|{hashlib.sha256(payload.encode()).hexdigest()}".encode())
    return roll.hexdigest()


def main(out_dir, n_steps, world_size):
    import os

    sys.path.insert(0, _REF_SRC)
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False, "n_steps": n_steps, "world_size": world_size}
    try:
        import torch

        import openpi.training.data_loader as _data_loader

        # (a) EXACT production DataLoader kwargs (shuffle=True + worker_init + generator).
        raw_prod, per_rank, seed = _build_id_dataset(world_size)
        gen = torch.Generator()
        gen.manual_seed(seed)
        prod_loader = torch.utils.data.DataLoader(
            raw_prod,
            batch_size=per_rank,
            shuffle=True,
            num_workers=_NUM_WORKERS,
            multiprocessing_context=torch.multiprocessing.get_context("spawn"),
            persistent_workers=True,
            collate_fn=_collate_ids,
            worker_init_fn=_data_loader._worker_init_fn,
            drop_last=True,
            generator=gen,
        )
        prod_rolling = _fanout_rolling(prod_loader, n_steps, world_size)
        del prod_loader

        # (b) the audit shortcut (shuffle=False, no generator/worker_init).
        raw_short, _, _ = _build_id_dataset(world_size)
        short_loader = torch.utils.data.DataLoader(
            raw_short,
            batch_size=per_rank,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            collate_fn=_collate_ids,
            drop_last=True,
            persistent_workers=True,
        )
        short_rolling = _fanout_rolling(short_loader, n_steps, world_size)

        result.update(
            ok=True,
            prod_settings_rolling=prod_rolling,
            shortcut_rolling=short_rolling,
            identical=prod_rolling == short_rolling,
            note="shuffle/sampler/worker_init are no-ops: the streaming dataset ignores idx",
        )
        with open(os.path.join(out_dir, "ref_loader_equiv.json"), "w", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1500:]
        with open(os.path.join(out_dir, "ref_loader_equiv.json"), "w", newline="\n") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print("REF_LOADER_EQUIV", json.dumps({"ok": result["ok"], "identical": result.get("identical"), "err": result.get("err")}))


if __name__ == "__main__":
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(out, n, ws)
