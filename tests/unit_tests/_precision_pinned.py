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

"""Artifact-backed pinned input for the dual-repo precision dtype ledger.

Both precision-ledger harnesses load the same committed, one-step slice extracted
from the BEHAVIOR SFT pinned-batch format used by ``_ref_pinned_run.py`` and
``build_pinned_behavior_sft_dataloader``:

* ``ref_pinned_batches.npz``-style arrays store fanout chunks as
  ``(n_steps * world_size, micro, ...)``;
* ``ref_pinned_noise_time.npz``-style arrays store injected flow noise/time as
  ``(n_steps, world_size * micro, ...)``.

This helper deliberately has no synthetic fallback. If the evidence artifact is
missing or its manifest does not identify the pinned SFT source path, the ledger
fails instead of silently regenerating a Gaussian shape fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[2]
_EVIDENCE = _REPO / "docs" / "evidence"
_ARTIFACT_PATH = _EVIDENCE / "phase5_precision_pinned_artifact.json"
_BATCHES_NAME = "phase5_precision_pinned_batches.npz"
_NOISE_TIME_NAME = "phase5_precision_pinned_noise_time.npz"
_IMG_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_REQUIRED_BATCH_KEYS = {
    *{f"image__{k}" for k in _IMG_KEYS},
    *{f"image_mask__{k}" for k in _IMG_KEYS},
    "state",
    "tokenized_prompt",
    "tokenized_prompt_mask",
    "actions",
}
_ALLOWED_GENERATORS = (
    "tests/unit_tests/_ref_pinned_run.py",
    "tests/unit_tests/_precision_pinned_artifact_rlinf.py",
)


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_REPO))
    except ValueError:
        return str(path)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tensor(t) -> str:
    a = np.ascontiguousarray(
        t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    )
    return _sha_bytes(a.tobytes())


def _load_manifest(artifact_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"precision pinned artifact manifest is missing: {_rel(path)}"
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("precision pinned artifact schema_version must be 1")
    if manifest.get("format") != "ref_pinned_npz":
        raise ValueError("precision pinned artifact must use ref_pinned_npz format")
    source = manifest.get("source", {})
    generator = str(source.get("generator", ""))
    if not any(g in generator for g in _ALLOWED_GENERATORS):
        raise ValueError(
            "precision pinned artifact must be derived from the existing "
            "pinned SFT input/noise-time path"
        )
    if manifest.get("synthetic") is not False:
        raise ValueError("precision pinned artifact must explicitly set synthetic=false")
    return manifest


def _resolve_artifact_file(manifest_path: Path, manifest: dict[str, Any], key: str) -> Path:
    value = manifest.get(key)
    if not value:
        raise ValueError(f"precision pinned artifact missing {key}")
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"precision pinned artifact file missing: {_rel(path)}")
    expected = manifest.get("file_sha256", {}).get(key)
    if expected and file_sha256(path) != expected:
        raise ValueError(f"precision pinned artifact sha256 mismatch for {key}")
    return path


def _validate_source_npzs(batches_npz: Path, noise_time_npz: Path) -> None:
    with np.load(batches_npz) as batches:
        missing = sorted(_REQUIRED_BATCH_KEYS - set(batches.files))
        if missing:
            raise ValueError(f"pinned batches npz missing keys: {missing}")
        if batches["actions"].ndim != 4:
            raise ValueError("pinned actions must have shape (chunks, micro, horizon, dim)")
    with np.load(noise_time_npz) as nt:
        if set(nt.files) < {"noise", "time"}:
            raise ValueError("pinned noise/time npz must contain noise and time")
        if nt["noise"].ndim != 4 or nt["time"].ndim != 2:
            raise ValueError("pinned noise/time arrays have an unexpected rank")


def build_pinned(
    batches_npz: str | os.PathLike[str],
    noise_time_npz: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] = _ARTIFACT_PATH,
    source_command: str,
    source_ref_git_rev: str = "unknown",
    source_rlinf_git_rev: str = "unknown",
    source_generator: str = "tests/unit_tests/_ref_pinned_run.py",
    world_size: int = 2,
    micro: int = 1,
) -> str:
    """Extract a compact precision-ledger artifact from pinned SFT NPZ dumps.

    The source files are expected to be produced by ``_ref_pinned_run.py`` or an
    equivalent wrapper around the same pinned SFT input/noise-time path. The
    committed artifact keeps the first ``micro`` rows of each rank's first-step
    fanout chunk so the precision ledger can run cheaply while still consuming
    the same schema and injected noise/time protocol.
    """
    src_batches = Path(batches_npz)
    src_noise_time = Path(noise_time_npz)
    if world_size < 1 or micro < 1:
        raise ValueError("world_size and micro must be positive")
    _validate_source_npzs(src_batches, src_noise_time)

    manifest = Path(manifest_path)
    out_dir = manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_batches = out_dir / _BATCHES_NAME
    out_noise_time = out_dir / _NOISE_TIME_NAME

    with np.load(src_batches) as batches:
        if batches["actions"].shape[0] < world_size:
            raise ValueError("source batches do not contain one chunk per requested rank")
        if batches["actions"].shape[1] < micro:
            raise ValueError("source batches micro dimension is smaller than requested")
        compact_batches = {
            key: np.ascontiguousarray(batches[key][:world_size, :micro])
            for key in batches.files
        }

    with np.load(src_noise_time) as nt:
        global_needed = world_size * micro
        if nt["noise"].shape[0] < 1 or nt["noise"].shape[1] < global_needed:
            raise ValueError("source noise does not cover the requested first global step")
        compact_noise = np.ascontiguousarray(nt["noise"][:1, :global_needed])
        compact_time = np.ascontiguousarray(nt["time"][:1, :global_needed])

    np.savez_compressed(out_batches, **compact_batches)
    np.savez_compressed(out_noise_time, noise=compact_noise, time=compact_time)

    doc = {
        "schema_version": 1,
        "format": "ref_pinned_npz",
        "synthetic": False,
        "batches_npz": out_batches.name,
        "noise_time_npz": out_noise_time.name,
        "world_size": world_size,
        "micro_batch": micro,
        "source": {
            "generator": source_generator,
            "command": source_command,
            "source_batches_npz": str(src_batches),
            "source_noise_time_npz": str(src_noise_time),
            "ref_git_rev": source_ref_git_rev,
            "rlinf_git_rev": source_rlinf_git_rev,
        },
        "file_sha256": {
            "batches_npz": file_sha256(out_batches),
            "noise_time_npz": file_sha256(out_noise_time),
        },
    }
    manifest.write_text(json.dumps(doc, indent=2) + "\n")
    return str(manifest)


def load_pinned(device, artifact_path: str | os.PathLike[str] = _ARTIFACT_PATH):
    """Load this rank's committed pinned batch and injected noise/time."""
    manifest_path = Path(artifact_path)
    manifest = _load_manifest(manifest_path)
    batches_path = _resolve_artifact_file(manifest_path, manifest, "batches_npz")
    noise_time_path = _resolve_artifact_file(manifest_path, manifest, "noise_time_npz")

    world_size = int(os.environ.get("WORLD_SIZE", manifest["world_size"]))
    rank = int(os.environ.get("RANK", 0))
    artifact_world_size = int(manifest["world_size"])
    micro = int(manifest["micro_batch"])
    if world_size != artifact_world_size:
        raise ValueError(
            f"precision pinned artifact world_size {artifact_world_size} != runtime {world_size}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} outside world_size {world_size}")

    with np.load(batches_path) as batches:
        missing = sorted(_REQUIRED_BATCH_KEYS - set(batches.files))
        if missing:
            raise ValueError(f"precision pinned batches missing keys: {missing}")
        chunk = rank
        images_np = {k: np.ascontiguousarray(batches[f"image__{k}"][chunk]) for k in _IMG_KEYS}
        image_masks_np = {
            k: np.ascontiguousarray(batches[f"image_mask__{k}"][chunk]) for k in _IMG_KEYS
        }
        state_np = np.ascontiguousarray(batches["state"][chunk])
        actions_np = np.ascontiguousarray(batches["actions"][chunk])
        tok_np = np.ascontiguousarray(batches["tokenized_prompt"][chunk])
        tok_mask_np = np.ascontiguousarray(batches["tokenized_prompt_mask"][chunk])

    lo = rank * micro
    with np.load(noise_time_path) as nt:
        noise_np = np.ascontiguousarray(nt["noise"][0, lo : lo + micro])
        time_np = np.ascontiguousarray(nt["time"][0, lo : lo + micro])

    def t(x):
        return torch.from_numpy(x).to(device)

    hashes = {
        **{f"image__{k}": sha_tensor(v) for k, v in images_np.items()},
        **{f"image_mask__{k}": sha_tensor(v) for k, v in image_masks_np.items()},
        "state": sha_tensor(state_np),
        "actions": sha_tensor(actions_np),
        "noise": sha_tensor(noise_np),
        "time": sha_tensor(time_np),
        "tokenized_prompt": sha_tensor(tok_np),
        "tokenized_prompt_mask": sha_tensor(tok_mask_np),
    }
    shapes = {
        "image": list(next(iter(images_np.values())).shape),
        "state": list(state_np.shape),
        "actions": list(actions_np.shape),
        "noise": list(noise_np.shape),
        "time": list(time_np.shape),
        "tokenized_prompt": list(tok_np.shape),
    }
    return {
        "spec_path": _rel(manifest_path),
        "artifact": {
            "manifest": _rel(manifest_path),
            "batches_npz": _rel(batches_path),
            "noise_time_npz": _rel(noise_time_path),
            "format": manifest["format"],
            "synthetic": manifest["synthetic"],
            "source": manifest["source"],
            "file_sha256": manifest["file_sha256"],
            "world_size": artifact_world_size,
            "micro_batch": micro,
            "rank": rank,
        },
        "images": {k: t(v) for k, v in images_np.items()},
        "image_masks": {k: t(v).bool() for k, v in image_masks_np.items()},
        "state": t(state_np).float(),
        "actions": t(actions_np).float(),
        "noise": t(noise_np).float(),
        "time": t(time_np).float(),
        "tokenized_prompt": t(tok_np).long(),
        "tokenized_prompt_mask": t(tok_mask_np).bool(),
        "hashes": hashes,
        "shapes": shapes,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Use build_pinned(<ref_pinned_batches.npz>, <ref_pinned_noise_time.npz>, "
        "source_command=...) from a Python shell or helper script."
    )
