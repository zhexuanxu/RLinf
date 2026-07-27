# Copyright 2026 The RLinf Authors.
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

"""Compute OpenPI normalization assets for BEHAVIOR control modes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

if __package__:
    from rlinf.data.datasets.openpi_pytorch.behavior.convert_control_mode import (
        _iter_jsonlines,
        _select_records,
        action_statistics,
        aggregate_feature_statistics,
    )
else:
    # The toolkit wrapper executes this file directly so Python does not import
    # the eager ``rlinf.data.datasets`` package (and its training dependencies)
    # before a conversion-only CLI can even parse ``--help``.
    from convert_control_mode import (  # type: ignore[import-not-found]
        _iter_jsonlines,
        _select_records,
        action_statistics,
        aggregate_feature_statistics,
    )
from rlinf.envs.behavior.control_modes import (
    CONTROL_MODE_ACTION_DIMS,
    CONTROL_MODES,
    STATE_TOKEN_DIMS,
    STATE_TOKENS,
    extract_state_from_proprio,
    norm_stats_asset_path,
    state_layout_for_token,
    validate_control_mode,
    validate_control_mode_dataset,
    validate_state_token,
)

_NORM_STAT_KEYS = ("mean", "std", "q01", "q99")


def _selected_episode_statistics(
    dataset_root: Path,
    episode_indices: Iterable[int],
    *,
    include_state: bool,
) -> list[dict[str, Any]]:
    requested = set(map(int, episode_indices))
    by_episode = {}
    for record in _iter_jsonlines(dataset_root / "meta/episodes_stats.jsonl"):
        episode_index = int(record["episode_index"])
        if episode_index in requested:
            source_stats = record["stats"]
            selected_stats = {"action": source_stats["action"]}
            if include_state:
                selected_stats["observation.state"] = source_stats["observation.state"]
            by_episode[episode_index] = selected_stats
            if len(by_episode) == len(requested):
                break
    selected = []
    for episode_index in episode_indices:
        if episode_index not in by_episode:
            raise ValueError(
                f"{dataset_root / 'meta/episodes_stats.jsonl'} has no episode "
                f"{episode_index}."
            )
        selected.append(by_episode[episode_index])
    return selected


def _raw_state_statistics(
    dataset_root: Path,
    info: dict[str, Any],
    episode_indices: Iterable[int],
    state_token: str,
) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    chunks_size = int(info.get("chunks_size", 10000))
    data_template = str(info["data_path"])
    per_episode = []
    for episode_index in episode_indices:
        relative_path = data_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
        )
        state_column = pq.read_table(
            dataset_root / relative_path, columns=["observation.state"]
        )["observation.state"]
        proprio = np.stack(
            [np.asarray(value, dtype=np.float64) for value in state_column.to_pylist()]
        )
        extracted = extract_state_from_proprio(proprio, state_token)
        per_episode.append(action_statistics(extracted))
    return aggregate_feature_statistics(per_episode)


def _summary_state_statistics(
    episode_statistics: list[dict[str, Any]], state_token: str
) -> dict[str, np.ndarray]:
    aggregated = aggregate_feature_statistics(
        record["observation.state"] for record in episode_statistics
    )
    return {
        key: np.asarray(
            extract_state_from_proprio(aggregated[key], state_token),
            dtype=np.float64,
        )
        for key in _NORM_STAT_KEYS
    }


def _pad(vector: np.ndarray, target_dim: int) -> list[float]:
    values = np.asarray(vector, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(
            f"normalization vector must be one-dimensional: {values.shape}."
        )
    if values.shape[0] > target_dim:
        raise ValueError(f"cannot pad a {values.shape[0]}-dim vector to {target_dim}.")
    output = np.zeros(target_dim, dtype=np.float64)
    output[: values.shape[0]] = values
    return output.tolist()


def compute_norm_stats(
    behavior_dataset_root: str | Path,
    control_mode: str,
    state_token: str,
    *,
    model_action_dim: int = 32,
    task_names: list[str] | None = None,
    episode_indices: list[int] | None = None,
    raw_state_stats: bool = False,
) -> dict[str, Any]:
    """Compute a padded OpenPI norm-stats payload from BEHAVIOR metadata.

    The default fast path count-aggregates ``meta/episodes_stats.jsonl``. For
    ``abs_eef`` state, quaternion-to-axis-angle is nonlinear, so raw Parquet
    states are always scanned before aggregation. ``raw_state_stats=True`` also
    enables that exact frame path for joint state layouts. Without that flag,
    joint-state index channels and means are exact, while collapsed-gripper
    standard deviations and quantiles inherit the source metadata's summary
    approximation.
    """
    mode = validate_control_mode(control_mode)
    token = validate_state_token(state_token)
    if model_action_dim < max(CONTROL_MODE_ACTION_DIMS[mode], STATE_TOKEN_DIMS[token]):
        raise ValueError(
            f"model_action_dim={model_action_dim} cannot hold {mode}/{token} "
            "state and actions."
        )

    root = Path(behavior_dataset_root).expanduser().resolve()
    validate_control_mode_dataset(root, mode, task_names)
    info = json.loads((root / "meta/info.json").read_text())
    chunks_size = int(info.get("chunks_size", 10000))
    selected_tasks, selected_episodes = _select_records(
        root, task_names, episode_indices, chunks_size
    )
    selected_episode_indices = [
        int(record["episode_index"]) for record in selected_episodes
    ]
    scan_raw_state = raw_state_stats or token == "abs_eef"
    episode_statistics = _selected_episode_statistics(
        root,
        selected_episode_indices,
        include_state=not scan_raw_state,
    )

    action_stats = aggregate_feature_statistics(
        record["action"] for record in episode_statistics
    )
    meaningful_action_dim = np.asarray(action_stats["mean"]).shape[-1]
    expected_action_dim = CONTROL_MODE_ACTION_DIMS[mode]
    if meaningful_action_dim != expected_action_dim:
        raise ValueError(
            f"{root} has {meaningful_action_dim}-dim action stats, expected "
            f"{expected_action_dim} for control_mode={mode!r}."
        )

    if scan_raw_state:
        state_stats = _raw_state_statistics(root, info, selected_episode_indices, token)
    else:
        state_stats = _summary_state_statistics(episode_statistics, token)

    norm_stats = {
        "state": {
            key: _pad(np.asarray(state_stats[key]), model_action_dim)
            for key in _NORM_STAT_KEYS
        },
        "actions": {
            key: _pad(np.asarray(action_stats[key]), model_action_dim)
            for key in _NORM_STAT_KEYS
        },
    }
    return {
        "norm_stats": norm_stats,
        "metadata": {
            "schema_version": 1,
            "control_mode": mode,
            "action_env_dim": expected_action_dim,
            "model_action_dim": model_action_dim,
            "state_token": token,
            "state_layout": state_layout_for_token(token),
            "state_dim": STATE_TOKEN_DIMS[token],
            "tasks": [str(record["task_name"]) for record in selected_tasks],
            "episodes": selected_episode_indices,
            "dataset_root": str(root),
            "raw_state_stats": bool(scan_raw_state),
        },
    }


def write_norm_stats(
    payload: dict[str, Any], assets_dir: str | Path, asset_id: str
) -> Path:
    """Write ``norm_stats.json`` where generic OpenPI asset fields resolve it."""
    path = norm_stats_asset_path(assets_dir, asset_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BEHAVIOR OpenPI norm_stats.json for one selected state/action "
            "representation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--behavior-dataset-root",
        required=True,
        help="Dataset selected by the generic data.behavior_dataset_root field.",
    )
    parser.add_argument(
        "--assets-dir",
        required=True,
        help="Generic actor.model.openpi.assets_dir value.",
    )
    parser.add_argument(
        "--asset-id",
        required=True,
        help="Generic actor.model.openpi.asset_id value below assets_dir.",
    )
    parser.add_argument("--control-mode", required=True, choices=CONTROL_MODES)
    parser.add_argument("--state-token", required=True, choices=STATE_TOKENS)
    parser.add_argument("--model-action-dim", type=int, default=32)
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to aggregate. Omit for every task in the dataset.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Absolute LeRobot episode_index values. Omit for all selected tasks.",
    )
    parser.add_argument(
        "--raw-state-stats",
        action="store_true",
        help=(
            "Scan raw frames for exact state stats. abs_eef always does this "
            "because quaternion conversion is nonlinear."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Run the control-mode norm-stat CLI."""
    args = _parse_args()
    payload = compute_norm_stats(
        args.behavior_dataset_root,
        args.control_mode,
        args.state_token,
        model_action_dim=args.model_action_dim,
        task_names=args.tasks,
        episode_indices=args.episodes,
        raw_state_stats=args.raw_state_stats,
    )
    path = write_norm_stats(payload, args.assets_dir, args.asset_id)
    print(f"Wrote norm stats to {path}.")


if __name__ == "__main__":
    main()
