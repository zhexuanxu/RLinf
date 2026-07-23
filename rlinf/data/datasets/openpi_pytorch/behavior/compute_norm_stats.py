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

"""Compute BEHAVIOR-1K state/action normalization statistics for the pi05 path.

This produces the ``norm_stats.json`` that the openpi_pytorch BEHAVIOR pipeline
consumes at both SFT time (``behavior_sft_data_loader``) and eval time
(``processing.BehaviorEvalProcessor``), via
:func:`rlinf.models.embodiment.openpi_pytorch.utils.normalize.load_norm_stats`.

How it works
------------
The BEHAVIOR-1K LeRobot dataset ships precomputed *per-episode* statistics in
``meta/episodes_stats.jsonl`` (``mean``/``std``/``q01``/``q99`` for the 256-dim
``observation.state`` and the 23-dim ``action``). Rather than iterating raw
frames, this script:

1. loads and count-aggregates those per-episode stats via
   :class:`~rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset.BehaviorSftDatasetMetadata`
   (which internally uses OmniGibson's ``aggregate_stats``);
2. maps the 256-dim ``observation.state`` stats down to the 23-dim policy state
   with the same
   :func:`~rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy.extract_state_from_proprio`
   used at train/eval time — so the stats are, by construction, in the exact
   channel order the model sees. ``--state-order`` selects that ordering:
   ``comet`` (reference / pretrained-checkpoint order, both grippers at the
   tail) or ``align`` (action-aligned, left gripper at index 14). The two
   orderings are NOT interchangeable, so ``--state-order`` here MUST match
   ``actor.model.openpi.state_order`` in the training/eval config;
3. zero-pads both the 23-dim state and the 23-dim action stats to ``--action-dim``
   (the model action dim, e.g. 32) and writes them as ``norm_stats.json``.

.. note::
   Applying ``extract_state_from_proprio`` to *statistics* is exact for the
   index-selected channels (base/trunk/arms) and for each gripper **mean**
   (the mean of a sum equals the sum of the means), but only approximate for the
   gripper ``std``/``q01``/``q99`` (the two finger channels are summed). This is
   the same trade-off the upstream openpi producer accepts to build the canonical
   asset. The ``abs_eef`` state layout is different: its arm channels are
   ``quat2axisangle(quat)``, a NONLINEAR map, so mapping the aggregated summary
   stats would be meaningless. For ``--state-token abs_eef`` the STATE stats are
   therefore computed from RAW frames (extract each frame, then aggregate with the
   same count-weighted method as the actions); the action stats are unaffected.

Environment
-----------
Run inside the BEHAVIOR venv (e.g. ``/mnt/public/xzxuan/.venv_pi``): building the
metadata triggers a lazy OmniGibson import for ``aggregate_stats``. Only the
dataset ``meta/**`` is read — no scene/asset load and no GPU are required.

Output layout
-------------
The file is written to ``<output-dir>/norm_stats.json``. The loader reads it from
``{assets_dir}/{asset_id}/norm_stats.json``, so point ``--output-dir`` at that
same directory, i.e. ``--output-dir <assets_dir>/<asset_id>``.

Examples
--------
Single task, action-aligned order, pad to the pi05 model action dim (32),
written where the loader expects ``asset_id = turn_on_radio_reorder``::

    python rlinf/data/datasets/openpi_pytorch/behavior/compute_norm_stats.py \\
        --dataset-root /mnt/public/xzxuan/data/2025-challenge-demos \\
        --repo-id behavior-1k/2025-challenge-demos \\
        --tasks turning_on_radio \\
        --action-dim 32 \\
        --state-order align \\
        --output-dir /mnt/public/xzxuan/repos/RLinf/outputs/norm_stats/turn_on_radio_reorder

All tasks in the dataset, reference (comet) order — omit ``--tasks`` and
``--state-order`` (comet is the default)::

    python rlinf/data/datasets/openpi_pytorch/behavior/compute_norm_stats.py \\
        --dataset-root /mnt/public/xzxuan/data/2025-challenge-demos \\
        --output-dir /path/to/assets/behavior-1k/2025-challenge-demos

Restrict aggregation to an explicit episode subset::

    python rlinf/data/datasets/openpi_pytorch/behavior/compute_norm_stats.py \\
        --dataset-root /mnt/public/xzxuan/ci_behavior/dataset \\
        --tasks turning_on_radio \\
        --episodes 10 11 12 13 \\
        --output-dir /tmp/behavior_norm_stats_check
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    extract_state_from_proprio,
)

# LeRobot stats key -> norm_stats.json key. State is reordered/sliced by
# extract_state_from_proprio; action passes through (already 23-dim, env order).
_STATE_SRC_KEY = "observation.state"
_ACTION_SRC_KEY = "action"
_STAT_KEYS = ("mean", "std", "q01", "q99")


def _pad_to_dim(x: np.ndarray, target_dim: int, value: float = 0.0) -> np.ndarray:
    """Right-pad a 1-D stat vector to ``target_dim`` with ``value``.

    Inlined here (rather than imported from behavior_sft_data_loader) so the
    ``--from-episodes-stats`` fast path does not drag in torch at import time.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] > target_dim:
        raise ValueError(
            f"cannot pad length-{x.shape[-1]} vector down to {target_dim}."
        )
    out = np.full(target_dim, value, dtype=np.float64)
    out[: x.shape[-1]] = x
    return out


def compute_norm_stats(
    *,
    dataset_root: str,
    repo_id: str,
    tasks: list[str] | None,
    episodes: list[int] | None,
    action_dim: int,
    state_order: str = "comet",
    control_mode: str = "joint_absolute",
    from_episodes_stats: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """Aggregate per-episode LeRobot stats into padded state/action norm stats.

    Returns the ``norm_stats`` payload (without the top-level ``"norm_stats"``
    wrapper): ``{"state": {mean/std/q01/q99}, "actions": {mean/std/q01/q99}}``,
    each value a list of length ``action_dim``. ``state_order`` selects the
    proprio->state channel ordering (``"comet"`` or ``"align"``) and MUST match
    the ordering used at train/eval time (``actor.model.openpi.state_order``).

    ``control_mode`` selects the ACTION semantics. ``joint_absolute`` uses the
    23-dim recorded joint action as-is (the original behavior). ``eef_delta_pose``
    expects ``dataset_root`` to be a converted delta-EEF dataset whose
    ``episodes_stats.jsonl`` already carries 21-dim delta-EEF action stats; those
    are aggregated and passed through (the state stats are still the recorded
    proprio, unchanged). The action's meaningful length is validated against the
    control mode before padding.

    ``from_episodes_stats`` aggregates ``meta/episodes_stats.jsonl`` directly with
    a faithful numpy re-implementation of OmniGibson's ``aggregate_stats`` (no
    OmniGibson/Isaac import), aggregating ALL episodes in the file. Use it for a
    converted delta-EEF dataset (whose episodes_stats is already filtered to the
    converted tasks) to avoid the heavy metadata path.
    """
    from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
        CONTROL_MODE_ACTION_ENV_DIM,
        CONTROL_MODES,
        resolve_state_token,
    )

    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"control_mode must be one of {CONTROL_MODES}, got {control_mode!r}."
        )

    if from_episodes_stats:
        from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef_delta import (
            aggregate_episode_stats_from_jsonl,
        )

        if episodes is not None:
            raise ValueError("--episodes is not supported with --from-episodes-stats.")
        stats = aggregate_episode_stats_from_jsonl(
            f"{dataset_root}/meta/episodes_stats.jsonl"
        )
    else:
        # Heavy path: import the OmniGibson-backed metadata only when actually
        # aggregating via meta.stats (the fast path above never needs it).
        from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
            BehaviorSftDatasetMetadata,
            _omnigibson_utils,
        )

        meta = BehaviorSftDatasetMetadata(
            repo_id=repo_id,
            root=dataset_root,
            tasks=tasks,
            modalities=[],
            cameras=[],
        )

        if episodes:
            # Aggregate over an explicit episode subset (mirrors BehaviorSftDataset's
            # per-episode-subset aggregation) rather than all selected-task episodes.
            _, aggregate_stats, _, _ = _omnigibson_utils()
            try:
                subset = [meta.episodes_stats[ep] for ep in episodes]
            except KeyError as exc:
                raise KeyError(
                    f"episode {exc} is not among the selected tasks' episodes; "
                    f"check --tasks and --episodes."
                ) from exc
            stats = aggregate_stats(subset)
        else:
            stats = meta.stats

    expected_action_dim = CONTROL_MODE_ACTION_ENV_DIM[control_mode]
    raw_action = np.asarray(stats[_ACTION_SRC_KEY]["mean"])
    if raw_action.shape[-1] != expected_action_dim:
        raise ValueError(
            f"control_mode={control_mode!r} expects {expected_action_dim}-dim "
            f"action stats, but {dataset_root}/meta/episodes_stats.jsonl has "
            f"{raw_action.shape[-1]}-dim action stats. For eef_delta_pose, point "
            f"--dataset-root at the converted delta-EEF dataset."
        )
    if expected_action_dim > action_dim:
        raise ValueError(
            f"action_dim (pad target {action_dim}) must be >= the meaningful "
            f"action length {expected_action_dim} for control_mode={control_mode!r}."
        )

    norm_stats: dict[str, dict[str, list[float]]] = {"state": {}, "actions": {}}
    if resolve_state_token(state_order) == "abs_eef":
        # abs_eef maps each arm quaternion through quat2axisangle -- a NONLINEAR
        # transform -- so mapping the AGGREGATED 256-dim summary stats through
        # extract_state_from_proprio (quat2axisangle(mean_quat), and axis-angle of a
        # std/quantile quaternion) is meaningless. Compute the STATE stats from RAW
        # frames instead (extract per frame, then aggregate); the ACTION stats still
        # come from the converted-dataset episode/meta aggregation above.
        from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef_delta import (
            compute_extracted_state_stats_from_frames,
        )

        state_frame_stats = compute_extracted_state_stats_from_frames(
            dataset_root, state_order, tasks=tasks, episodes=episodes
        )
        for key in _STAT_KEYS:
            state_vec = np.asarray(state_frame_stats[key])
            action_vec = np.asarray(stats[_ACTION_SRC_KEY][key])
            norm_stats["state"][key] = _pad_to_dim(state_vec, action_dim).tolist()
            norm_stats["actions"][key] = _pad_to_dim(action_vec, action_dim).tolist()
        return norm_stats

    # Linear joint layouts (abs_joint_old / abs_joint): index-selection + gripper
    # sum, so mapping the aggregated summary stats is exact (mean) / the documented
    # gripper approximation (std/q01/q99); no raw-frame read needed.
    for key in _STAT_KEYS:
        state_vec = extract_state_from_proprio(
            np.asarray(stats[_STATE_SRC_KEY][key]), state_order
        )
        action_vec = np.asarray(stats[_ACTION_SRC_KEY][key])
        norm_stats["state"][key] = _pad_to_dim(state_vec, action_dim).tolist()
        norm_stats["actions"][key] = _pad_to_dim(action_vec, action_dim).tolist()
    return norm_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute BEHAVIOR-1K state/action norm stats (norm_stats.json) for "
            "the openpi_pytorch pi05 path."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="LeRobot dataset root (the behavior_dataset_root), e.g. "
        "/mnt/public/xzxuan/data/2025-challenge-demos.",
    )
    parser.add_argument(
        "--repo-id",
        default="behavior-1k/2025-challenge-demos",
        help="LeRobot repo id.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Task name(s) to include, e.g. turning_on_radio. Omit for all tasks.",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        type=int,
        default=None,
        help="Optional explicit episode indices to aggregate over. Omit to use "
        "all episodes of the selected tasks.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=32,
        help="Pad target (= model action dim); state and actions are zero-padded "
        "to this length.",
    )
    parser.add_argument(
        "--state-order",
        "--state-token",
        dest="state_token",
        default="abs_joint_old",
        help="Proprio->state channel layout (canonical state_token, legacy "
        "state_order aliases resolve): 'abs_joint_old'(==comet) both grippers at "
        "tail; 'abs_joint'(==align) action-aligned; 'abs_eef' arms as base-frame "
        "EEF [pos, axisangle]. Also accepts 'comet'/'align'. MUST match "
        "actor.model.openpi.state_token.",
    )
    parser.add_argument(
        "--control-mode",
        choices=("joint_absolute", "absolute_eef", "delta_eef", "eef_delta_pose"),
        default="joint_absolute",
        help="Action space. 'joint_absolute' uses the recorded 23-dim joint "
        "action; the EEF modes (absolute_eef / delta_eef / legacy eef_delta_pose) "
        "expect --dataset-root to be a converted 21-dim EEF dataset.",
    )
    parser.add_argument(
        "--from-episodes-stats",
        action="store_true",
        help="Aggregate meta/episodes_stats.jsonl directly with a faithful numpy "
        "re-implementation of OmniGibson's aggregate_stats (no OmniGibson import). "
        "Aggregates ALL episodes in the file. Recommended for a converted "
        "EEF dataset (its episodes_stats is already filtered to the "
        "converted tasks).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write into; norm_stats.json is created inside it. "
        "Point this at {assets_dir}/{asset_id} so load_norm_stats finds it.",
    )
    return parser.parse_args()


def _dataset_task_names(dataset_root: str) -> list[str] | None:
    """Task names from ``dataset_root/meta/tasks.jsonl`` (in file order).

    Used to self-document the norm-stats manifest when the caller aggregates a
    whole converted dataset (``--from-episodes-stats`` without ``--tasks``), so
    the manifest records the actual covered tasks instead of ``null``. Returns
    ``None`` if the file is absent (the manifest then records ``null``, matching
    the prior behavior rather than failing stats generation).
    """
    tasks_path = pathlib.Path(dataset_root).expanduser() / "meta" / "tasks.jsonl"
    if not tasks_path.is_file():
        return None
    names: list[str] = []
    for line in tasks_path.read_text().splitlines():
        line = line.strip()
        if line:
            names.append(json.loads(line)["task_name"])
    return names or None


def main() -> None:
    args = _parse_args()
    from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
        CONTROL_MODE_ACTION_ENV_DIM,
    )

    norm_stats = compute_norm_stats(
        dataset_root=args.dataset_root,
        repo_id=args.repo_id,
        tasks=args.tasks,
        episodes=args.episodes,
        action_dim=args.action_dim,
        state_order=args.state_token,
        control_mode=args.control_mode,
        from_episodes_stats=args.from_episodes_stats,
    )
    out_path = pathlib.Path(args.output_dir).expanduser() / "norm_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Record the covered tasks explicitly. When aggregating a whole converted
    # dataset (--from-episodes-stats without --tasks), derive the list from the
    # dataset's meta/tasks.jsonl so the manifest documents the real coverage
    # (e.g. all 50) instead of null.
    manifest_tasks = args.tasks
    if manifest_tasks is None and args.from_episodes_stats:
        manifest_tasks = _dataset_task_names(args.dataset_root)
    # Manifest documents what this asset was built for so the loader/eval can
    # reject a stats/dim/mode mismatch instead of silently normalizing a 21-dim
    # action against a 23-dim joint stats file.
    manifest = {
        "control_mode": args.control_mode,
        "action_env_dim": CONTROL_MODE_ACTION_ENV_DIM[args.control_mode],
        "model_action_dim": args.action_dim,
        "state_token": args.state_token,
        "tasks": manifest_tasks,
        "dataset_root": args.dataset_root,
    }
    out_path.write_text(
        json.dumps({"norm_stats": norm_stats, "metadata": manifest}, indent=2)
    )
    print(f"Wrote norm stats to: {out_path}")


if __name__ == "__main__":
    main()
