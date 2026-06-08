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

"""Non-invasive step-0 instrumentation for the BEHAVIOR SFT training path.

This module observes (never alters) the production 8-GPU FSDP SFT run to produce a
trustworthy step-0 baseline:

* ``frame_hashes`` computes a per-frame identity (sha256 over state + actions +
  tokenized prompt) using the SAME algorithm as the reference fanout dump
  (``tests/unit_tests/_ref_fanout_dump.py``), so RLinf and the reference are directly
  comparable.
* ``capture_step0`` all-gathers each rank's step-0 frame-id set, and on rank 0 records
  the per-rank sets, the global union count (must be 256 = rank-disjoint effective
  batch), pairwise disjointness, and the step-0 loss / grad-norm / lr. This proves the
  rank/world-size streaming-loader fix was EXERCISED AT RUNTIME (not merely present in
  source) and pins the real step-0 numbers.
* ``dump_provenance`` records, on rank 0, the git rev + dirty status, the exact launch
  command, the fully-resolved config, package versions, the base-checkpoint and
  norm-stats sha256, the dataset path, seed, and the world/micro/global batch sizes.

Both functions are read-only with respect to the loader iterator / RNG / sampler state:
they hash already-materialized tensors and gather metadata. They are gated OFF by
default in the worker so production training is unaffected.

All artifacts are written under a caller-provided directory (the caller must point this
at ``/mnt/public/xzxuan/tmp``; the compact result is later committed under
``docs/evidence/``).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any

import numpy as np


def _np(x):
    # Preserve the native dtype bytes EXACTLY like the reference fanout helper
    # (tests/unit_tests/_ref_fanout_dump.py): integer token ids must hash as ints,
    # not be cast to float, or the frame hash is not cross-repo comparable.
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def frame_hashes(observation: Any, actions: Any) -> list[str]:
    """Per-frame identity for a micro-batch — identical algorithm to the reference
    fanout dump so the two repos' frame ids are byte-comparable."""
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


def _sha256_file(path: str) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_dir_manifest(path: str, *, max_files: int = 64) -> dict[str, Any] | None:
    """A directory fingerprint: per-file size + sha256 of up to ``max_files`` files,
    plus the count. Used for the base-checkpoint / norm-stats dir."""
    if not path or not os.path.isdir(path):
        return None
    entries = []
    n = 0
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            n += 1
            if len(entries) < max_files:
                fp = os.path.join(root, name)
                rel = os.path.relpath(fp, path)
                entries.append(
                    {"file": rel, "size": os.path.getsize(fp), "sha256": _sha256_file(fp)}
                )
    rollup = hashlib.sha256()
    for e in sorted(entries, key=lambda e: e["file"]):
        rollup.update(e["file"].encode())
        rollup.update(str(e["sha256"]).encode())
    return {"n_files": n, "rollup_sha256": rollup.hexdigest(), "files": entries}


def _git(args: list[str], repo: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo, *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return ""


def dump_provenance(
    cfg: Any,
    *,
    rank: int,
    world_size: int,
    out_dir: str,
    repo_root: str,
    resolved_config: dict[str, Any] | None = None,
    gradient_accumulation: int | None = None,
    reference_repo: str = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed",
    extra: dict[str, Any] | None = None,
) -> str | None:
    """Write a provenance JSON on rank 0. Returns the path (or None off rank 0)."""
    if rank != 0:
        return None
    os.makedirs(out_dir, exist_ok=True)

    def _g(path, default=None):
        cur = cfg
        for k in path.split("."):
            cur = getattr(cur, k, None) if not isinstance(cur, dict) else cur.get(k)
            if cur is None:
                return default
        return cur

    model_path = _g("actor.model.model_path")
    assets_dir = _g("actor.model.openpi.assets_dir")
    asset_id = _g("actor.model.openpi.asset_id")
    norm_stats_json = (
        os.path.join(assets_dir, asset_id, "norm_stats.json")
        if assets_dir and asset_id
        else None
    )
    num_workers = (
        _g("actor.num_workers")
        or _g("actor.model.openpi.num_workers")
        or _g("data.num_workers")
        or 8  # build_behavior_sft_dataloader default
    )
    pkg_versions = {}
    for pkg in ("torch", "ray", "numpy"):
        try:
            pkg_versions[pkg] = __import__(pkg).__version__
        except Exception:
            pkg_versions[pkg] = None

    # OpenPI / reference provenance: the reference repo has no installed package
    # __version__, so record the reference repo git rev + the import path of the
    # RLinf-vendored openpi_pytorch model package.
    try:
        import rlinf.models.embodiment.openpi_pytorch as _openpi_pkg

        openpi_import_path = os.path.dirname(os.path.abspath(_openpi_pkg.__file__))
    except Exception:
        openpi_import_path = None
    reference_provenance = {
        "reference_repo": reference_repo,
        "reference_git_rev": _git(["rev-parse", "HEAD"], reference_repo) or None,
        "openpi_vendored_import_path": openpi_import_path,
    }

    prov = {
        "schema_version": 1,
        "git": {
            "rev": _git(["rev-parse", "HEAD"], repo_root),
            "dirty": bool(_git(["status", "--porcelain"], repo_root)),
            "diff_stat": _git(["diff", "--stat"], repo_root)[:4000],
        },
        "launch_command": " ".join(os.environ.get("RLINF_LAUNCH_CMD", "").split())
        or None,
        "world_size": world_size,
        "num_workers": num_workers,
        "global_batch_size": _g("actor.global_batch_size"),
        "micro_batch_size": _g("actor.micro_batch_size"),
        "gradient_accumulation": gradient_accumulation,
        "seed": _g("actor.seed") or _g("runner.seed"),
        "dataset_path": _g("actor.train_data_paths") or _g("data.train_data_paths"),
        "optimizer": {
            "adam_beta1": _g("actor.optim.adam_beta1"),
            "adam_beta2": _g("actor.optim.adam_beta2"),
            "lr": _g("actor.optim.lr"),
        },
        "base_checkpoint": {
            "path": model_path,
            "fingerprint": _sha256_dir_manifest(model_path) if model_path else None,
        },
        "norm_stats": {
            "assets_dir": assets_dir,
            "asset_id": asset_id,
            "norm_stats_json": norm_stats_json,
            "norm_stats_sha256": _sha256_file(norm_stats_json)
            if norm_stats_json
            else None,
            "fingerprint": _sha256_dir_manifest(assets_dir) if assets_dir else None,
        },
        "optimizer_full": {
            "adam_beta1": _g("actor.optim.adam_beta1"),
            "adam_beta2": _g("actor.optim.adam_beta2"),
            "adam_eps": _g("actor.optim.adam_eps"),
            "weight_decay": _g("actor.optim.weight_decay"),
            "lr": _g("actor.optim.lr"),
            "lr_scheduler": _g("actor.optim.lr_scheduler"),
        },
        "package_versions": pkg_versions,
        "reference_provenance": reference_provenance,
        "resolved_config": resolved_config,
    }
    if extra:
        prov["extra"] = extra
    path = os.path.join(out_dir, "sft_step0_provenance.json")
    with open(path, "w") as f:
        json.dump(prov, f, indent=2, default=str)
    return path


def capture_step0(
    observation: Any,
    actions: Any,
    *,
    rank: int,
    world_size: int,
    out_dir: str,
    loss: float | None = None,
    grad_norm: float | None = None,
    lr: float | None = None,
    global_batch_size: int | None = None,
) -> dict[str, Any] | None:
    """All-gather per-rank step-0 frame ids; on rank 0 record union/disjointness +
    the step-0 loss/grad_norm/lr. Read-only w.r.t. the loader. Returns the artifact
    dict on rank 0 (also written to ``out_dir``), else None."""
    import torch
    import torch.distributed as dist

    local_frames = frame_hashes(observation, actions)
    if dist.is_available() and dist.is_initialized() and world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, local_frames)
    else:
        gathered = [local_frames]

    if rank != 0:
        return None

    os.makedirs(out_dir, exist_ok=True)
    artifact = summarize_step0_frames(gathered, global_batch_size=global_batch_size)
    artifact["step0"] = {
        "loss": None if loss is None else float(loss),
        "grad_norm": None if grad_norm is None else float(grad_norm),
        "lr": None if lr is None else float(lr),
    }
    path = os.path.join(out_dir, "sft_step0_capture.json")
    with open(path, "w") as f:
        json.dump(artifact, f, indent=2)
    return artifact


def summarize_step0_frames(
    gathered: list[list[str]], *, global_batch_size: int | None = None
) -> dict[str, Any]:
    """Pure summary of per-rank frame-id lists: union count, rank-disjointness, and
    whether the loader fix was exercised (rank-disjoint AND union == global batch).
    Separated from I/O / distributed code so it is unit-testable without a GPU."""
    per_rank_sets = [set(fr) for fr in gathered]
    union = set().union(*per_rank_sets) if per_rank_sets else set()
    # pairwise disjoint <=> sum of per-rank sizes == union size (frames distinct/rank)
    total = sum(len(s) for s in per_rank_sets)
    rank_disjoint = total == len(union)
    per_rank_hash = [
        hashlib.sha256("".join(sorted(fr)).encode()).hexdigest()[:16] for fr in gathered
    ]
    global_set_hash = hashlib.sha256("".join(sorted(union)).encode()).hexdigest()
    return {
        "schema_version": 1,
        "world_size": len(gathered),
        "per_rank_frame_count": [len(fr) for fr in gathered],
        "per_rank_frame_hash": per_rank_hash,
        "global_unique_frames": len(union),
        "expected_global_batch": global_batch_size,
        "rank_disjoint": bool(rank_disjoint),
        "global_set_hash": global_set_hash,
        "fix_exercised": bool(
            rank_disjoint
            and (global_batch_size is None or len(union) == global_batch_size)
        ),
    }
