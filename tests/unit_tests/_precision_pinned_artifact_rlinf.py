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

"""Generate the compact pinned-input artifact consumed by the precision ledger.

This wrapper uses RLinf's BEHAVIOR SFT loader against the same local task/data
configuration and writes the same two-NPZ fanout format used by
``_ref_pinned_run.py``. It exists as a practical artifact-generation path when
the reference venv cannot import its loader, while still avoiding the old
shape-only synthetic fixture.

Run from the repo root:

    python tests/unit_tests/_precision_pinned_artifact_rlinf.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from _precision_pinned import build_pinned

_REPO = Path("/mnt/public/xzxuan/repos/RLinf_pi05")
sys.path.insert(0, str(_REPO))
_OUT = _REPO / "docs" / "evidence" / "phase5_precision_pinned_artifact.json"
_TMP = Path("/mnt/public/xzxuan/tmp/phase5_precision_pinned_artifact")
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_NOISE_SEED = 1234


def _git_rev(repo: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _first_rank_batch(cfg, world_size: int, rank: int):
    from rlinf.data.datasets.behavior import build_behavior_sft_dataloader

    loader, _ = build_behavior_sft_dataloader(
        cfg, world_size, rank, cfg.data.train_data_paths, False
    )
    batch = next(iter(loader))
    if isinstance(batch, dict):
        return batch["observation"], batch["actions"]
    return batch


def _source_npzs(world_size: int, micro: int, tmp_dir: Path) -> tuple[Path, Path]:
    import torch
    from hydra import compose, initialize_config_dir

    os.environ.setdefault("EMBODIED_PATH", str(_REPO / "examples/embodiment"))
    os.environ.setdefault("REPO_PATH", str(_REPO))
    cfg_dir = _REPO / "examples/sft/config"
    with initialize_config_dir(version_base="1.1", config_dir=str(cfg_dir)):
        cfg = compose(
            config_name="behavior_pi05_vla",
            overrides=[f"actor.micro_batch_size={micro}", "+data.num_workers=0"],
        )

    tmp_dir.mkdir(parents=True, exist_ok=True)
    store: dict[str, list[np.ndarray]] = {}
    actions_shape = None
    for rank in range(world_size):
        observation, actions = _first_rank_batch(cfg, world_size, rank)
        actions_shape = tuple(actions.shape)
        for key in _IMG:
            store.setdefault(f"image__{key}", []).append(_np(observation.images[key]))
            store.setdefault(f"image_mask__{key}", []).append(
                _np(observation.image_masks[key])
            )
        store.setdefault("state", []).append(_np(observation.state))
        store.setdefault("tokenized_prompt", []).append(_np(observation.tokenized_prompt))
        store.setdefault("tokenized_prompt_mask", []).append(
            _np(observation.tokenized_prompt_mask)
        )
        store.setdefault("actions", []).append(_np(actions))

    batches_npz = tmp_dir / "ref_pinned_batches.npz"
    np.savez_compressed(batches_npz, **{k: np.stack(v) for k, v in store.items()})

    if actions_shape is None:
        raise RuntimeError("no pinned SFT batch was produced")
    _, horizon, action_dim = actions_shape
    rs = np.random.RandomState(_NOISE_SEED)
    noise = rs.randn(1, world_size * micro, horizon, action_dim).astype(np.float32)
    time = (
        rs.beta(1.5, 1.0, size=(1, world_size * micro)).astype(np.float32) * 0.999
        + 0.001
    )
    noise_time_npz = tmp_dir / "ref_pinned_noise_time.npz"
    np.savez_compressed(noise_time_npz, noise=noise, time=time)

    # Keep torch imported until after DataLoader teardown.
    _ = torch
    return batches_npz, noise_time_npz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--micro", type=int, default=1)
    parser.add_argument("--out", default=str(_OUT))
    parser.add_argument("--tmp-dir", default=str(_TMP))
    args = parser.parse_args()

    batches_npz, noise_time_npz = _source_npzs(
        world_size=args.world_size,
        micro=args.micro,
        tmp_dir=Path(args.tmp_dir),
    )
    command = " ".join(
        [
            "python",
            "tests/unit_tests/_precision_pinned_artifact_rlinf.py",
            f"--world-size={args.world_size}",
            f"--micro={args.micro}",
        ]
    )
    manifest = build_pinned(
        batches_npz,
        noise_time_npz,
        manifest_path=args.out,
        source_command=command,
        source_rlinf_git_rev=_git_rev(_REPO),
        source_ref_git_rev=_git_rev(Path("/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed")),
        source_generator="tests/unit_tests/_precision_pinned_artifact_rlinf.py",
        world_size=args.world_size,
        micro=args.micro,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
