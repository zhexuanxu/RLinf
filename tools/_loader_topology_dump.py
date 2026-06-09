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

"""torchrun worker: dump a loader's first-N per-rank micro-batch IDENTITY hashes at the
PRODUCTION TOPOLOGY (8 ranks x num_workers=8), for the R32 composition analysis.

Codex R31 review: the loader's per-step batch composition depends on the rank/worker-aware
partition, so the replay must use the real 8-rank / num_workers=8 topology, not world_size=1
/ num_workers=0. This worker runs under ``torchrun --nproc_per_node=8`` (gloo/CPU) so each
rank's loader sees the production ``rank``/``world_size``/``num_workers``, iterates the
first ``n_steps`` micro-batches, and computes a per-FRAME identity hash (sha256 of
state+actions+tokenized_prompt) for every frame. Rank 0 all-gathers the per-rank hashes and
writes ``<out>/<side>_topology_hashes.json`` -- the per-(rank, step) 32-frame-hash lists +
a manifest (side, config, world_size, num_workers, batch_size, seed, task, n_steps, src).

The R32 analysis then computes, per global step, how many UNIQUE frames the loader emits
across its 8 ranks (32 if the ranks are rank-REPLICATED, up to N*32 if rank-DISJOINT) --
proving each loader's effective per-step batch composition rather than assuming it.

    # RLinf (RLinf venv):
    torchrun --standalone --nproc_per_node=8 tools/_loader_topology_dump.py --side rlinf --out <dir>
    # reference (reference venv, with its src on path):
    <ref_venv>/python -m torch.distributed.run --standalone --nproc_per_node=8 \
        tools/_loader_topology_dump.py --side ref --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys

import numpy as np

_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_REF_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"  # GLOBAL batch 256
_TASK = "turning_on_radio"
_NUM_WORKERS = 8
_SEED = 42
_GLOBAL = 256
_MICRO = 32


def _git_rev(repo):
    try:
        return subprocess.check_output(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _frame_hashes(observation, actions):
    """Per-frame sha256[:16] of (state, actions, tokenized_prompt) -- a within-loader identity."""
    state = _np(observation.state)
    act = _np(actions)
    tp = _np(getattr(observation, "tokenized_prompt", np.zeros((state.shape[0], 1))))
    out = []
    for i in range(state.shape[0]):
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(state[i]).tobytes())
        h.update(np.ascontiguousarray(act[i]).tobytes())
        h.update(np.ascontiguousarray(tp[i]).tobytes())
        out.append(h.hexdigest()[:16])
    return out


def _build_rlinf(rank, world, nw):
    from rlinf.data.datasets.behavior import create_behavior_sft_data_loader

    return create_behavior_sft_data_loader(
        behavior_dataset_root=_DATA_ROOT,
        assets_dir=_ASSETS_DIR,
        asset_id="behavior-1k/2025-challenge-demos",
        repo_id="behavior-1k/2025-challenge-demos",
        tasks=[_TASK],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=_MICRO,  # per-rank micro_batch_size=32
        num_workers=nw,
        seed=_SEED,
        use_skill=False,
        # Mirror the production worker (build_behavior_sft_dataloader passes rank/world_size):
        # the explicit identity makes the spawned DataLoader workers rank-disjoint.
        dist_rank=rank,
        dist_world_size=world,
    )


def _build_ref(rank, world, nw):
    import dataclasses
    import sys

    sys.path.insert(0, _REF_SRC)
    import openpi.training.config as _config
    import openpi.training.data_loader as _data_loader

    cfg = _config.get_config(_REF_CONFIG)
    base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
    assets = _config.AssetsConfig(
        assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
    )
    data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
    # Keep the production num_workers + GLOBAL batch 256 (under dist -> local 256//world per rank).
    run_cfg = dataclasses.replace(cfg, data=data, batch_size=_GLOBAL, num_workers=nw)
    return iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--side", choices=["rlinf", "ref"], required=True)
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    ap.add_argument(
        "--with-images",
        action="store_true",
        help="rank 0 also saves its batches (npz) for the replay",
    )
    args = ap.parse_args()
    nw = args.num_workers

    import torch.distributed as dist

    dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()

    loader = (
        _build_rlinf(rank, world, nw)
        if args.side == "rlinf"
        else _build_ref(rank, world, nw)
    )
    it = iter(loader) if args.side == "rlinf" else loader

    _IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    per_step = []
    store = {}
    save_imgs = args.with_images and rank == 0
    for _ in range(args.n_steps):
        obs, actions = next(it)
        per_step.append(_frame_hashes(obs, actions))
        if (
            save_imgs
        ):  # rank 0 keeps its (effective, since rank-replicated) per-step batch
            for k in _IMG:
                store.setdefault(f"image__{k}", []).append(_np(obs.images[k]))
                store.setdefault(f"image_mask__{k}", []).append(_np(obs.image_masks[k]))
            store.setdefault("state", []).append(_np(obs.state))
            store.setdefault("tokenized_prompt", []).append(_np(obs.tokenized_prompt))
            store.setdefault("tokenized_prompt_mask", []).append(
                _np(obs.tokenized_prompt_mask)
            )
            store.setdefault("actions", []).append(_np(actions))
    if save_imgs:
        np.savez(
            f"{args.out}/{args.side}_rank0_batches{args.tag}.npz",
            **{k: np.stack(v) for k, v in store.items()},
        )
    local_batch = len(per_step[0]) if per_step else 0

    gathered = [None] * world
    dist.all_gather_object(
        gathered,
        {"rank": rank, "per_step_frame_hashes": per_step, "local_batch": local_batch},
    )

    if rank == 0:
        gathered.sort(key=lambda g: g["rank"])
        import torch

        is_ref = args.side == "ref"
        manifest = {
            "side": args.side,
            "config": _REF_CONFIG if is_ref else "create_behavior_sft_data_loader",
            "world_size": world,
            "num_workers": nw,
            "global_batch": _GLOBAL,
            "per_rank_local_batch": [g["local_batch"] for g in gathered],
            "seed": _SEED,
            "task": _TASK,
            "n_steps": args.n_steps,
            "data_root": _DATA_ROOT,
            "ref_src": _REF_SRC if is_ref else None,
            "provenance": {
                "command": f"torchrun --standalone --nproc_per_node={world} {sys.argv[0]} "
                + " ".join(sys.argv[1:]),
                "backend": "gloo",
                "rank_count": world,
                "num_workers": nw,
                "seed": _SEED,
                "task": _TASK,
                "data_root": _DATA_ROOT,
                "assets_dir": _ASSETS_DIR,
                "rlinf_repo": "/mnt/public/xzxuan/repos/RLinf_pi05",
                "rlinf_git_rev": _git_rev("/mnt/public/xzxuan/repos/RLinf_pi05"),
                "reference_repo": "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed"
                if is_ref
                else None,
                "reference_git_rev": _git_rev(
                    "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed"
                )
                if is_ref
                else None,
                "loader_builder": "openpi.training.data_loader.create_behavior_data_loader_torch"
                if is_ref
                else "rlinf.data.datasets.behavior.create_behavior_sft_data_loader",
                "torch_version": torch.__version__,
                "return_status": "rank0_wrote_manifest_ok",
                "output_reused": False,
                "with_images": args.with_images,
            },
            "per_rank_per_step_frame_hashes": [
                g["per_step_frame_hashes"] for g in gathered
            ],
        }
        out_path = f"{args.out}/{args.side}_topology_hashes{args.tag}.json"
        with open(out_path, "w", newline="\n") as f:
            json.dump(manifest, f)
            f.write("\n")
        print(
            f"LOADER_TOPOLOGY_DUMP side={args.side} world={world} local_batch={local_batch} wrote {out_path}",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
