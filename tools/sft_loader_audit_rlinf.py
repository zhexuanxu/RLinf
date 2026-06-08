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

"""Data-only per-global-step frame-id dump for RLinf's 8-rank BEHAVIOR SFT loader.

Reconstructs the PRODUCTION per-rank streaming partition single-process by building one
``build_behavior_sft_dataloader`` per rank with EXPLICIT ``dist_rank``/``dist_world_size``
(the committed spawn-worker fix) and the production ``num_workers``, then iterating all
ranks in lockstep. For each global step it records every rank's ordered frame ids using
the value-independent ``(episode_index, frame_index)`` identity (taken BEFORE normalization/
tokenization, see ``id_only`` below) so the two repos are byte-comparable.

No model / no GPU. Output (per-rank-per-step frame hashes + per-step global-set hash +
distinct count) goes under a caller-provided scratch dir (point at /mnt/public/xzxuan/tmp).

    EMBODIED_PATH=.../examples/sft REPO_PATH=<repo> PYTHONPATH=<repo> \
        python tools/sft_loader_audit_rlinf.py <out_dir> <n_steps> [world_size]
"""

from __future__ import annotations

import hashlib
import json
import os
import sys


def main(out_dir, n_steps, world_size):
    os.makedirs(out_dir, exist_ok=True)
    result = {"ok": False, "repo": "rlinf", "world_size": world_size, "n_steps": n_steps}
    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
            build_behavior_sft_dataloader,
        )

        cfg_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "examples/sft/config",
        )
        with initialize_config_dir(version_base="1.1", config_dir=cfg_dir):
            cfg = compose(config_name="behavior_pi05_vla")
        OmegaConf.set_struct(cfg, False)
        data_paths = OmegaConf.select(cfg, "data.train_data_paths")

        # Each rank's stream is INDEPENDENT (its own dist_rank partition), so run the
        # ranks SEQUENTIALLY (one production loader at a time = prod num_workers, not
        # world_size*num_workers concurrent processes) and combine per-step afterward.
        # id_only=True: each sample is a value-independent (episode_index, frame_index)
        # id taken BEFORE normalization/tokenization, so the frame identity is byte-
        # comparable across repos AND no video decode is needed (the 30k pass is fast).
        # rank_step_ids[r][step] = that rank's 32 frame-id strings at that step.
        rank_step_ids = []
        num_workers = None
        for r in range(world_size):
            loader, _ = build_behavior_sft_dataloader(
                cfg, world_size=world_size, rank=r, data_paths=data_paths, id_only=True
            )
            if num_workers is None:
                num_workers = loader.torch_loader.num_workers
            it = iter(loader)
            steps_r = []
            for _ in range(n_steps):
                try:
                    batch = next(it)
                except StopIteration:
                    # Epoch boundary: re-iterate, exactly as the production SFT worker
                    # does (run_training resets data_iter on StopIteration).
                    it = iter(loader)
                    batch = next(it)
                steps_r.append([f"{ep}:{fr}" for (ep, fr) in batch])
            rank_step_ids.append(steps_r)
            del it, loader  # tear down this rank's workers before the next

        # Compact committed output: per-step global-set hash + distinct count + a rolling
        # hash over the whole schedule. The FULL per-step global sets (sorted ids) go to a
        # separate detail file (large for 30k -> NOT committed) so the comparator can
        # report concrete first-mismatch ids.
        per_step_set_hash, per_step_unique, per_step_global_set = [], [], []
        rolling = hashlib.sha256()
        for step in range(n_steps):
            step_rank_ids = [rank_step_ids[r][step] for r in range(world_size)]
            union = sorted({f for rh in step_rank_ids for f in rh})
            sh = hashlib.sha256("".join(union).encode()).hexdigest()
            per_step_set_hash.append(sh)
            per_step_unique.append(len(union))
            per_step_global_set.append(union)
            rolling.update(f"{step}|{sh}".encode())

        result.update(
            ok=True,
            seed=int(cfg.actor.get("seed", 42)),
            per_rank_local_batch=int(cfg.actor.micro_batch_size),
            num_workers=int(num_workers) if num_workers is not None else None,
            audited_steps=n_steps,
            per_step_global_set_hash=per_step_set_hash,
            per_step_unique=per_step_unique,
            rolling_hash=rolling.hexdigest(),
        )
        with open(os.path.join(out_dir, "rlinf_loader_audit.json"), "w", newline="\n") as f:
            json.dump(result, f)
            f.write("\n")
        with open(
            os.path.join(out_dir, "rlinf_loader_audit_detail.json"), "w", newline="\n"
        ) as f:
            json.dump({"per_step_global_set": per_step_global_set}, f)
            f.write("\n")
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1500:]
        with open(os.path.join(out_dir, "rlinf_loader_audit.json"), "w", newline="\n") as f:
            json.dump(result, f)
            f.write("\n")
    print("RLINF_LOADER_AUDIT", json.dumps({"ok": result["ok"], "err": result.get("err")}))


if __name__ == "__main__":
    out = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    ws = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    main(out, n, ws)
