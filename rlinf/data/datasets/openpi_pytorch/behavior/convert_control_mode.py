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

"""Write a BEHAVIOR LeRobot dataset in a selected OpenPI control mode.

Large video files and action-independent annotation trees are symlinked. Parquet
files are rewritten because their action column changes, and every metadata file
is filtered to the selected tasks/episodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.behavior.control_modes import (
    CONTROL_MODE_ACTION_DIMS,
    CONTROL_MODE_MANIFEST,
    CONTROL_MODES,
    convert_episode_actions,
    read_control_mode_manifest,
    validate_control_mode,
    validate_control_mode_dataset,
    write_control_mode_manifest,
)


def action_statistics(actions: np.ndarray) -> dict[str, list[float] | list[int]]:
    """Return LeRobot-compatible statistics for one episode's actions."""
    values = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(
            f"actions must be a non-empty (frames, dim) array; got {values.shape}."
        )
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def aggregate_feature_statistics(
    statistics: Iterable[dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Count-aggregate per-episode LeRobot feature statistics.

    The mean/variance combination and percentile-of-episode-quantiles match the
    BEHAVIOR metadata aggregation used to create the original assets.
    """
    records = list(statistics)
    if not records:
        raise ValueError("cannot aggregate an empty statistics sequence.")
    means = np.stack(
        [np.asarray(record["mean"], dtype=np.float64) for record in records]
    )
    variances = np.stack(
        [np.asarray(record["std"], dtype=np.float64) ** 2 for record in records]
    )
    counts = np.asarray(
        [
            float(np.asarray(record["count"], dtype=np.float64).reshape(-1)[0])
            for record in records
        ],
        dtype=np.float64,
    )
    if np.any(counts <= 0):
        raise ValueError("every statistics record must have a positive count.")
    expanded_counts = counts.reshape((-1,) + (1,) * (means.ndim - 1))
    total_count = counts.sum()
    total_mean = (means * expanded_counts).sum(axis=0) / total_count
    delta = means - total_mean
    total_variance = ((variances + delta**2) * expanded_counts).sum(
        axis=0
    ) / total_count
    q01 = np.stack([np.asarray(record["q01"], dtype=np.float64) for record in records])
    q99 = np.stack([np.asarray(record["q99"], dtype=np.float64) for record in records])
    return {
        "min": np.min(
            np.stack(
                [np.asarray(record["min"], dtype=np.float64) for record in records]
            ),
            axis=0,
        ),
        "max": np.max(
            np.stack(
                [np.asarray(record["max"], dtype=np.float64) for record in records]
            ),
            axis=0,
        ),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "q01": np.percentile(q01, 1, axis=0),
        "q99": np.percentile(q99, 99, axis=0),
        "count": np.asarray([int(total_count)], dtype=np.int64),
    }


def _iter_jsonlines(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _load_jsonlines(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonlines(path))


def _write_jsonlines(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _index_episode_statistics(path: Path, episode_indices: set[int]) -> dict[int, int]:
    """Index selected JSONL records by seek offset without retaining payloads."""
    offsets: dict[int, int] = {}
    with path.open() as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            episode_index = int(json.loads(line)["episode_index"])
            if episode_index in episode_indices:
                offsets[episode_index] = offset
                if len(offsets) == len(episode_indices):
                    break
    missing = sorted(episode_indices - offsets.keys())
    if missing:
        raise ValueError(f"{path} has no episode statistics for {missing}.")
    return offsets


def _validate_roots(source: Path, destination: Path) -> None:
    source = source.resolve()
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(
            f"source dataset has no metadata: {source / 'meta/info.json'}."
        )
    if destination.is_symlink():
        raise ValueError(f"destination must not be a symlink: {destination}.")
    destination = destination.resolve(strict=False)
    try:
        common = Path(os.path.commonpath([source, destination]))
    except ValueError:
        common = None
    if destination == source or common in (source, destination):
        raise ValueError(
            "source and destination dataset trees must be disjoint; in-place, "
            "nested, and ancestor destinations are refused "
            f"({source} -> {destination})."
        )


def _validate_converter_owned_destination(destination: Path) -> None:
    """Refuse to recursively replace a directory not owned by this converter."""
    manifest_path = destination / CONTROL_MODE_MANIFEST
    if manifest_path.is_symlink():
        raise ValueError(
            f"refusing to overwrite destination with a symlinked ownership "
            f"manifest: {manifest_path}."
        )
    manifest = read_control_mode_manifest(destination)
    if manifest is None or int(manifest.get("schema_version", -1)) != 1:
        raise ValueError(
            f"refusing to overwrite {destination}: it is not a converter-owned "
            f"dataset with a schema-version-1 {CONTROL_MODE_MANIFEST}."
        )
    try:
        mode = validate_control_mode(str(manifest.get("control_mode")))
        validate_control_mode_dataset(destination, mode)
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise ValueError(
            f"refusing to overwrite {destination}: its converter ownership "
            "metadata is invalid."
        ) from exc


def _select_records(
    source: Path,
    task_names: list[str] | None,
    episode_indices: list[int] | None,
    chunks_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    task_records = _load_jsonlines(source / "meta/tasks.jsonl")
    task_by_name = {record["task_name"]: record for record in task_records}
    requested_tasks = (
        list(task_by_name) if task_names is None else list(dict.fromkeys(task_names))
    )
    missing_tasks = sorted(set(requested_tasks) - set(task_by_name))
    if missing_tasks:
        raise ValueError(
            f"unknown task names {missing_tasks}; inspect {source / 'meta/tasks.jsonl'}."
        )
    requested_task_indices = {
        int(task_by_name[name]["task_index"]) for name in requested_tasks
    }

    requested_episodes = (
        None if episode_indices is None else set(map(int, episode_indices))
    )
    episode_records = _load_jsonlines(source / "meta/episodes.jsonl")
    selected_episodes = [
        record
        for record in episode_records
        if int(record["episode_index"]) // chunks_size in requested_task_indices
        and (
            requested_episodes is None
            or int(record["episode_index"]) in requested_episodes
        )
    ]
    if requested_episodes is not None:
        found = {int(record["episode_index"]) for record in selected_episodes}
        missing_episodes = sorted(requested_episodes - found)
        if missing_episodes:
            raise ValueError(
                f"episode ids {missing_episodes} were not found under the selected "
                "tasks. --episodes uses absolute LeRobot episode_index values."
            )
    if not selected_episodes:
        raise ValueError("no episodes matched the requested tasks/episode ids.")

    actual_task_indices = {
        int(record["episode_index"]) // chunks_size for record in selected_episodes
    }
    selected_tasks = [
        record
        for record in task_records
        if int(record["task_index"]) in actual_task_indices
    ]
    return selected_tasks, selected_episodes


def _link_shared_tree(source: Path, destination: Path, name: str) -> None:
    source_path = source / name
    if source_path.exists():
        (destination / name).symlink_to(source_path, target_is_directory=True)


def _link_episode_metadata(
    source: Path,
    destination: Path,
    info: dict[str, Any],
    episode_records: list[dict[str, Any]],
    chunks_size: int,
) -> None:
    template = info.get("metainfo_path")
    if not template:
        return
    for record in episode_records:
        episode_index = int(record["episode_index"])
        relative_path = Path(
            template.format(
                episode_chunk=episode_index // chunks_size,
                episode_index=episode_index,
            )
        )
        source_path = source / relative_path
        if source_path.is_file():
            output_path = destination / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.symlink_to(source_path)


def _rewrite_episode(
    source_path: Path,
    destination_path: Path,
    control_mode: str,
) -> tuple[np.ndarray, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(source_path)
    for required_column in ("action", "observation.state"):
        if required_column not in table.column_names:
            raise ValueError(f"{source_path} has no {required_column!r} column.")
    actions = np.stack(
        [np.asarray(value, dtype=np.float64) for value in table["action"].to_pylist()]
    )
    states = np.stack(
        [
            np.asarray(value, dtype=np.float64)
            for value in table["observation.state"].to_pylist()
        ]
    )
    converted = convert_episode_actions(actions, states, control_mode).astype(
        np.float32
    )
    action_column = pa.array(converted.tolist(), type=pa.list_(pa.float32()))
    action_index = table.schema.get_field_index("action")
    table = table.set_column(action_index, "action", action_column)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination_path)
    return converted, table.num_rows


def _build_dataset(
    source: Path,
    destination: Path,
    control_mode: str,
    selected_tasks: list[dict[str, Any]],
    selected_episodes: list[dict[str, Any]],
    info: dict[str, Any],
    model_action_dim: int,
    progress_every: int,
    converter: Callable[[Path, Path, str], tuple[np.ndarray, int]],
) -> dict[str, Any]:
    mode = validate_control_mode(control_mode)
    chunks_size = int(info.get("chunks_size", 10000))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "meta").mkdir()
    (destination / "data").mkdir()
    for shared_tree in ("videos", "annotations", "orchestrators"):
        _link_shared_tree(source, destination, shared_tree)
    _link_episode_metadata(source, destination, info, selected_episodes, chunks_size)

    selected_episode_ids = {
        int(record["episode_index"]) for record in selected_episodes
    }
    source_stats_path = source / "meta/episodes_stats.jsonl"
    stats_offsets = _index_episode_statistics(source_stats_path, selected_episode_ids)
    output_stats_path = destination / "meta/episodes_stats.jsonl"
    total_frames = 0
    start_time = time.monotonic()
    data_template = str(info["data_path"])
    with (
        source_stats_path.open() as source_stats_stream,
        output_stats_path.open("w") as output_stats_stream,
    ):
        for position, episode_record in enumerate(selected_episodes, start=1):
            episode_index = int(episode_record["episode_index"])
            relative_path = Path(
                data_template.format(
                    episode_chunk=episode_index // chunks_size,
                    episode_index=episode_index,
                )
            )
            converted, frame_count = converter(
                source / relative_path, destination / relative_path, mode
            )
            total_frames += frame_count
            source_stats_stream.seek(stats_offsets[episode_index])
            stats_record = json.loads(source_stats_stream.readline())
            stats_record["stats"]["action"] = action_statistics(converted)
            output_stats_stream.write(json.dumps(stats_record) + "\n")
            if progress_every and (
                position % progress_every == 0 or position == len(selected_episodes)
            ):
                elapsed = max(time.monotonic() - start_time, 1e-9)
                rate = position / elapsed
                remaining = (len(selected_episodes) - position) / rate
                print(
                    f"Converted {position}/{len(selected_episodes)} episodes "
                    f"({total_frames} frames, {rate:.2f} episodes/s, "
                    f"ETA {remaining / 60:.1f} min).",
                    flush=True,
                )

    _write_jsonlines(destination / "meta/tasks.jsonl", selected_tasks)
    _write_jsonlines(destination / "meta/episodes.jsonl", selected_episodes)

    output_info = dict(info)
    output_info["features"] = {
        key: dict(value) for key, value in info["features"].items()
    }
    output_info["features"]["action"]["shape"] = [CONTROL_MODE_ACTION_DIMS[mode]]
    output_info["total_episodes"] = len(selected_episodes)
    output_info["total_frames"] = total_frames
    output_info["total_tasks"] = len(selected_tasks)
    output_info["total_chunks"] = len(selected_tasks)
    output_info["splits"] = {"train": f"0:{len(selected_episodes)}"}
    if "total_videos" in output_info:
        video_count = sum(
            feature.get("dtype") == "video"
            for feature in output_info["features"].values()
        )
        output_info["total_videos"] = len(selected_episodes) * video_count
    (destination / "meta/info.json").write_text(
        json.dumps(output_info, indent=2) + "\n"
    )

    task_names = [str(record["task_name"]) for record in selected_tasks]
    task_indices = {
        str(record["task_name"]): int(record["task_index"]) for record in selected_tasks
    }
    episode_indices = [int(record["episode_index"]) for record in selected_episodes]
    manifest = {
        "schema_version": 1,
        "control_mode": mode,
        "action_env_dim": CONTROL_MODE_ACTION_DIMS[mode],
        "model_action_dim": model_action_dim,
        "source_action_dim": 23,
        "tasks": task_names,
        "task_indices": task_indices,
        "episodes": episode_indices,
        "source_root": str(source),
        "conversion": {
            "abs_joint": "recorded absolute joint targets",
            "delta_joint": (
                "achieved arm qpos at t+1 minus achieved arm qpos at t; "
                "base, trunk, and grippers pass through"
            ),
            "abs_eef": (
                "achieved base-frame EEF position and axis-angle at t+1; "
                "base, trunk, and grippers pass through"
            ),
            "delta_eef": (
                "achieved base-frame EEF position and relative-rotation delta "
                "from t to t+1; base, trunk, and grippers pass through"
            ),
        }[mode],
        "orientation": "xyzw quaternion converted with scipy Rotation",
    }
    write_control_mode_manifest(destination, manifest)
    return {
        "control_mode": mode,
        "episodes": len(selected_episodes),
        "frames": total_frames,
        "tasks": task_names,
    }


def convert_behavior_dataset(
    source_root: str | Path,
    destination_root: str | Path,
    control_mode: str,
    *,
    task_names: list[str] | None = None,
    episode_indices: list[int] | None = None,
    model_action_dim: int = 32,
    overwrite: bool = False,
    progress_every: int = 100,
) -> dict[str, Any]:
    """Convert and atomically publish a selected BEHAVIOR dataset subset.

    Args:
        source_root: Native 23-dimensional BEHAVIOR LeRobot dataset.
        destination_root: New dataset root. It must be outside ``source_root``.
        control_mode: One exact value from :data:`CONTROL_MODES`.
        task_names: Optional task-name subset. Omit to consider every task.
        episode_indices: Optional absolute LeRobot episode ids. Omit to convert
            every episode belonging to ``task_names``.
        model_action_dim: Padded action width recorded in provenance.
        overwrite: Replace an existing destination only after conversion succeeds.
        progress_every: Print progress every N episodes; zero disables progress.

    Returns:
        Conversion counts and the published ``destination_root``.
    """
    mode = validate_control_mode(control_mode)
    if model_action_dim < CONTROL_MODE_ACTION_DIMS[mode]:
        raise ValueError(
            f"model_action_dim={model_action_dim} is smaller than the "
            f"{CONTROL_MODE_ACTION_DIMS[mode]}-dim {mode} action."
        )
    source = Path(source_root).expanduser().resolve()
    destination_arg = Path(destination_root).expanduser().absolute()
    if destination_arg.is_symlink():
        raise ValueError(f"destination must not be a symlink: {destination_arg}.")
    destination = destination_arg.resolve(strict=False)
    _validate_roots(source, destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"destination exists: {destination}; pass --overwrite to replace it."
        )
    if destination.exists():
        _validate_converter_owned_destination(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    info = json.loads((source / "meta/info.json").read_text())
    source_action_dim = int(info["features"]["action"]["shape"][-1])
    if source_action_dim != 23:
        raise ValueError(
            "the converter requires the native 23-dimensional abs_joint "
            f"dataset; {source} declares {source_action_dim}."
        )
    # Width alone cannot distinguish native abs_joint data from converted
    # delta_joint data. Validate provenance before applying any conversion so a
    # same-width source can never be silently relabeled.
    validate_control_mode_dataset(source, "abs_joint")
    chunks_size = int(info.get("chunks_size", 10000))
    selected_tasks, selected_episodes = _select_records(
        source, task_names, episode_indices, chunks_size
    )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        result = _build_dataset(
            source,
            temporary,
            mode,
            selected_tasks,
            selected_episodes,
            info,
            model_action_dim,
            progress_every,
            _rewrite_episode,
        )
        if destination.exists():
            # Recheck immediately before the destructive overwrite in case a
            # destination path was replaced while conversion was staged.
            _validate_roots(source, destination)
            _validate_converter_owned_destination(destination)
            shutil.rmtree(destination)
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    result["destination_root"] = str(destination)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BEHAVIOR OpenPI SFT data to one exact control mode. The "
            "source dataset is never modified."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dataset-root",
        required=True,
        help="Native 23-dimensional BEHAVIOR LeRobot dataset.",
    )
    parser.add_argument(
        "--output-dataset-root",
        required=True,
        help="Destination root outside the source dataset.",
    )
    parser.add_argument(
        "--control-mode",
        required=True,
        choices=CONTROL_MODES,
        help="Target action representation.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task names to convert. Omit to include every task.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Absolute LeRobot episode_index values. Omit to include every "
            "episode of the selected tasks."
        ),
    )
    parser.add_argument(
        "--model-action-dim",
        type=int,
        default=32,
        help="Padded OpenPI action width recorded in provenance.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace an existing destination only when its canonical manifest "
            "proves it was written by this converter."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N episodes; use zero to silence.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the control-mode dataset converter CLI."""
    args = _parse_args()
    result = convert_behavior_dataset(
        args.source_dataset_root,
        args.output_dataset_root,
        args.control_mode,
        task_names=args.tasks,
        episode_indices=args.episodes,
        model_action_dim=args.model_action_dim,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(
        f"Wrote {result['episodes']} episodes / {result['frames']} frames in "
        f"{result['control_mode']} mode to {result['destination_root']}."
    )


if __name__ == "__main__":
    main()
