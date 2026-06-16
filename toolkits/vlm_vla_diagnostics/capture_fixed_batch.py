# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture a deterministic real BEHAVIOR SFT batch for VLM-VLA diagnostics.

The saved payload is directly consumable by ``grad_norm_audit.py --batch-path``:
top-level ``observation`` and ``actions`` contain the ``vlm_vla`` transformed
batch. It also stores matched ``vla`` and ``vlm_vla`` variants built from the
same raw frames so action-quality diagnostics can compare checkpoints without a
frame-distribution confound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import sys
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (  # noqa: E402
    BehaviorSftDataset,
    BehaviorSftTransform,
    collate_behavior_sft_items,
    validate_behavior_data_config_migration,
)
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id  # noqa: E402
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (  # noqa: E402
    load_norm_stats,
)


def _resolve_config(config_name: str, overrides: list[str]):
    config_dir = REPO_ROOT / "examples" / "sft" / "config"
    os.environ.setdefault("EMBODIED_PATH", str(REPO_ROOT / "examples" / "sft"))
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _build_level1_dataset(cfg, *, shuffle: bool, seed: int, rank: int, world_size: int):
    data_cfg = cfg.data
    model_cfg = cfg.actor.model
    validate_behavior_data_config_migration(data_cfg)

    data_path = resolve_lerobot_repo_id(data_cfg.train_data_paths)
    if data_path is None:
        raise ValueError("data.train_data_paths must resolve to a BEHAVIOR dataset.")

    tasks = list(data_cfg.tasks)
    if len(tasks) != 1:
        raise ValueError(f"Expected one BEHAVIOR task for diagnostics, got {tasks}.")
    labels = data_cfg.task_subtasks.get(tasks[0])
    if not labels:
        raise ValueError(f"Missing data.task_subtasks.{tasks[0]} labels.")
    subtask_labels = {i: str(label) for i, label in enumerate(labels)}

    return BehaviorSftDataset(
        repo_id=str(data_cfg.repo_id),
        root=str(data_cfg.behavior_dataset_root),
        tolerance_s=float(data_cfg.tolerance_s),
        tasks=tasks,
        modalities=list(data_cfg.modalities),
        local_only=True,
        delta_timestamps={
            "action": [t / 30.0 for t in range(int(model_cfg.num_action_chunks))]
        },
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=1,
        subtask_labels=subtask_labels,
        enable_gap=bool(data_cfg.get("enable_gap", True)),
        dist_rank=rank,
        dist_world_size=world_size,
    )


def _build_transform(cfg, *, vlm_vla: bool) -> BehaviorSftTransform:
    model_cfg = cfg.actor.model
    norm_stats = load_norm_stats(model_cfg.openpi.assets_dir, model_cfg.openpi.asset_id)
    return BehaviorSftTransform(
        norm_stats=norm_stats,
        tokenizer_path=str(model_cfg.openpi.paligemma_tokenizer),
        action_dim=int(model_cfg.openpi.model_action_dim),
        max_token_len=int(model_cfg.openpi.max_token_len),
        vlm_vla=vlm_vla,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _sample_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "episode_index",
        "frame_index",
        "timestamp",
        "task_index",
        "prompt",
        "task",
        "response",
    )
    return {key: _jsonable(frame[key]) for key in keys if key in frame}


def _update_hash_from_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    h.update(str(array.shape).encode("utf-8"))
    h.update(str(array.dtype).encode("utf-8"))
    h.update(array.tobytes())


def _batch_fingerprint(observation, actions: torch.Tensor) -> str:
    h = hashlib.sha256()
    for key in sorted(observation.images):
        _update_hash_from_tensor(h, observation.images[key])
    for key in sorted(observation.image_masks):
        _update_hash_from_tensor(h, observation.image_masks[key])
    _update_hash_from_tensor(h, observation.state)
    _update_hash_from_tensor(h, actions)
    return h.hexdigest()


def _variant_payload(observation, actions: torch.Tensor) -> dict[str, Any]:
    return {
        "observation": observation.to_dict(),
        "actions": actions,
        "fingerprint": _batch_fingerprint(observation, actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="behavior_pi05_vlm_vla")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/public/xzxuan/tmp/vlm_vla_fixed_batch.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Match training's shuffled chunk order; default is deterministic order.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides applied while composing the SFT config.",
    )
    args = parser.parse_args()

    cfg = _resolve_config(args.config_name, args.overrides)
    dataset = _build_level1_dataset(
        cfg,
        shuffle=args.shuffle,
        seed=args.seed,
        rank=args.rank,
        world_size=args.world_size,
    )
    vla_transform = _build_transform(cfg, vlm_vla=False)
    vlm_vla_transform = _build_transform(cfg, vlm_vla=True)

    raw_frames = [dataset[i] for i in range(args.batch_size)]
    vla_items = [vla_transform(frame) for frame in raw_frames]
    vlm_vla_items = [vlm_vla_transform(frame) for frame in raw_frames]
    vla_observation, vla_actions = collate_behavior_sft_items(vla_items)
    vlm_vla_observation, vlm_vla_actions = collate_behavior_sft_items(vlm_vla_items)

    vla_payload = _variant_payload(vla_observation, vla_actions)
    vlm_vla_payload = _variant_payload(vlm_vla_observation, vlm_vla_actions)
    if vla_payload["fingerprint"] != vlm_vla_payload["fingerprint"]:
        raise RuntimeError("Matched batch variants do not share frame/action data.")

    metadata = {
        "config_name": args.config_name,
        "overrides": args.overrides,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "rank": args.rank,
        "world_size": args.world_size,
        "data_path": str(cfg.data.train_data_paths),
        "behavior_dataset_root": str(cfg.data.behavior_dataset_root),
        "repo_id": str(cfg.data.repo_id),
        "tasks": list(cfg.data.tasks),
        "enable_gap": bool(cfg.data.get("enable_gap", True)),
        "tokenizer_path": str(cfg.actor.model.openpi.paligemma_tokenizer),
        "assets_dir": str(cfg.actor.model.openpi.assets_dir),
        "asset_id": str(cfg.actor.model.openpi.asset_id),
        "fingerprint": vlm_vla_payload["fingerprint"],
        "samples": [_sample_metadata(frame) for frame in raw_frames],
    }
    try:
        import subprocess

        metadata["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception as exc:  # pragma: no cover - diagnostic metadata only
        metadata["git_commit_error"] = str(exc)

    payload = {
        "metadata": metadata,
        "observation": vlm_vla_payload["observation"],
        "actions": vlm_vla_payload["actions"],
        "variants": {
            "vla": vla_payload,
            "vlm_vla": vlm_vla_payload,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved_batch={args.output}")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
