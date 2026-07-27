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

"""Shared BEHAVIOR state and action representations for OpenPI models.

The public configuration surface intentionally has four action modes and four
state-token choices. This module is the single source of truth for their names,
dimensions, conversion math, and on-disk dataset provenance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_MODES = ("abs_joint", "delta_joint", "abs_eef", "delta_eef")
"""Exact values accepted by ``actor.model.openpi.control_mode``."""

CONTROL_MODE_ACTION_DIMS = {
    "abs_joint": 23,
    "delta_joint": 23,
    "abs_eef": 21,
    "delta_eef": 21,
}
"""Meaningful environment action width before OpenPI pads to its model width."""

OPENPI_MODEL_ACTION_DIM = 32
"""Fixed padded action/state width of the migrated π₀.₅ checkpoints."""

STATE_TOKENS = ("none", "abs_joint_old", "abs_joint", "abs_eef")
"""Exact values accepted by ``actor.model.openpi.state_token``."""

STATE_TOKEN_DIMS = {
    "none": 23,
    "abs_joint_old": 23,
    "abs_joint": 23,
    "abs_eef": 21,
}

CONTROL_MODE_MANIFEST = Path("meta/control_mode.json")

# R1Pro fields in the 256-dimensional ``observation.state`` vector.
R1PRO_PROPRIO_INDICES = {
    "arm_left_qpos": np.s_[158:165],
    "eef_left_pos": np.s_[186:189],
    "eef_left_quat": np.s_[189:193],
    "gripper_left_qpos": np.s_[193:195],
    "arm_right_qpos": np.s_[197:204],
    "eef_right_pos": np.s_[225:228],
    "eef_right_quat": np.s_[228:232],
    "gripper_right_qpos": np.s_[232:234],
    "trunk_qpos": np.s_[236:240],
    "base_qvel": np.s_[253:256],
}

_ACTION_SLICES = {
    "base": np.s_[0:3],
    "trunk": np.s_[3:7],
    "arm_left": np.s_[7:14],
    "gripper_left": 14,
    "arm_right": np.s_[15:22],
    "gripper_right": 22,
}


def validate_control_mode(value: str) -> str:
    """Validate and return an exact public control-mode name."""
    if value not in CONTROL_MODES:
        raise ValueError(
            f"control_mode must be one of {CONTROL_MODES}; got {value!r}. "
            "Legacy aliases are not supported."
        )
    return value


def validate_state_token(value: str) -> str:
    """Validate and return an exact public state-token name."""
    if value not in STATE_TOKENS:
        raise ValueError(
            f"state_token must be one of {STATE_TOKENS}; got {value!r}. "
            "Legacy aliases are not supported."
        )
    return value


def state_layout_for_token(state_token: str) -> str:
    """Return the continuous-state layout used by a prompt state selector.

    ``none`` suppresses prompt injection but keeps the same checkpoint-facing
    continuous layout as ``abs_joint_old``.
    """
    token = validate_state_token(state_token)
    return "abs_joint_old" if token == "none" else token


def _state_layout_from_metadata(metadata: dict[str, Any]) -> str | None:
    """Resolve canonical or legacy on-disk state-layout metadata.

    Legacy names are accepted only while reading provenance. They remain
    invalid values for the public ``state_token`` selector.
    """
    value = metadata.get("state_token", metadata.get("state_order"))
    if value is None:
        return None
    aliases = {"comet": "abs_joint_old", "align": "abs_joint"}
    token = aliases.get(str(value), str(value))
    return state_layout_for_token(validate_state_token(token))


def norm_stats_asset_path(assets_dir: str | Path, asset_id: str) -> Path:
    """Resolve one generic OpenPI norm-stat asset below ``assets_dir``."""
    relative_asset = Path(asset_id)
    if relative_asset.is_absolute() or ".." in relative_asset.parts:
        raise ValueError(
            "asset_id must be a relative path below assets_dir without '..'; "
            f"got {asset_id!r}."
        )
    return Path(assets_dir).expanduser() / relative_asset / "norm_stats.json"


def validate_norm_stats_asset(
    assets_dir: str | Path,
    asset_id: str,
    control_mode: str,
    state_token: str,
    *,
    model_action_dim: int | None = None,
    tasks: list[str] | None = None,
) -> dict[str, Any] | None:
    """Reject normalization assets built for another state/action layout.

    Legacy assets without canonical metadata are accepted only for native
    ``abs_joint`` with the original state layout (or state-free π₀.₅).
    Converted modes require exact provenance.
    """
    mode = validate_control_mode(control_mode)
    token = validate_state_token(state_token)
    path = norm_stats_asset_path(assets_dir, asset_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata")
    metadata_dict = metadata if isinstance(metadata, dict) else None
    canonical_metadata = (
        metadata
        if isinstance(metadata, dict) and metadata.get("control_mode") is not None
        else None
    )
    if canonical_metadata is None:
        if mode != "abs_joint":
            raise ValueError(
                f"{path} has no canonical control-mode metadata and can only be "
                "used with the native abs_joint mode."
            )
        asset_layout = (
            _state_layout_from_metadata(metadata_dict)
            if metadata_dict is not None
            else None
        )
        if token != "none":
            requested_layout = state_layout_for_token(token)
            if asset_layout is None and requested_layout != "abs_joint_old":
                raise ValueError(
                    f"{path} has no state-layout metadata and is only "
                    "checkpoint-compatible with state_token='none' or "
                    f"'abs_joint_old', not {token!r}."
                )
            if asset_layout is not None and asset_layout != requested_layout:
                raise ValueError(
                    f"{path} records legacy state layout {asset_layout!r}, "
                    f"not the requested {requested_layout!r}."
                )
    else:
        metadata = canonical_metadata
        if validate_control_mode(str(metadata.get("control_mode"))) != mode:
            raise ValueError(
                f"{path} was built for control_mode="
                f"{metadata.get('control_mode')!r}, not {mode!r}."
            )
        asset_state_token = validate_state_token(str(metadata.get("state_token")))
        inferred_asset_layout = state_layout_for_token(asset_state_token)
        asset_layout = str(metadata.get("state_layout", inferred_asset_layout))
        if asset_layout != inferred_asset_layout:
            raise ValueError(
                f"{path} has inconsistent state_token/state_layout metadata: "
                f"{asset_state_token!r} -> {asset_layout!r}."
            )
        # A state-free π₀.₅ prefix does not consume the normalized state.
        # Continue to materialize the old continuous tensor for checkpoint
        # compatibility, but do not reject an otherwise matching action asset
        # solely because its unused state statistics have another layout.
        if token != "none" and asset_layout != state_layout_for_token(token):
            raise ValueError(
                f"{path} was built for state_token="
                f"{metadata.get('state_token')!r}, not {token!r}."
            )
        if int(metadata.get("action_env_dim", -1)) != CONTROL_MODE_ACTION_DIMS[mode]:
            raise ValueError(f"{path} has inconsistent action_env_dim metadata.")
        if model_action_dim is not None and int(
            metadata.get("model_action_dim", -1)
        ) != int(model_action_dim):
            raise ValueError(
                f"{path} was padded to model_action_dim="
                f"{metadata.get('model_action_dim')}, not {model_action_dim}."
            )
    if tasks and metadata_dict is not None and "tasks" in metadata_dict:
        covered_tasks = set(metadata_dict.get("tasks") or [])
        missing_tasks = sorted(set(tasks) - covered_tasks)
        if missing_tasks:
            raise ValueError(
                f"{path} does not cover requested tasks {missing_tasks}. "
                "Compute norm stats from the same task set used for SFT."
            )
    elif tasks and canonical_metadata is not None:
        raise ValueError(f"{path} has canonical metadata but no task coverage.")

    if model_action_dim is not None:
        try:
            for feature in ("state", "actions"):
                for key in ("mean", "std", "q01", "q99"):
                    vector = payload["norm_stats"][feature][key]
                    if len(vector) != model_action_dim:
                        raise ValueError(
                            f"{path} {feature}.{key} has length {len(vector)}, "
                            f"expected {model_action_dim}."
                        )
        except KeyError as exc:
            raise ValueError(f"{path} is missing norm_stats field {exc}.") from exc
    return canonical_metadata


def quaternion_to_axis_angle(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternions to robust axis-angle vectors.

    SciPy uses the same convention as OmniGibson and remains stable around the
    pi singularity common in the R1Pro's downward-facing gripper pose.
    """
    from scipy.spatial.transform import Rotation

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape[-1] != 4:
        raise ValueError(
            f"quaternion must have trailing dimension 4; got {quaternion.shape}."
        )
    original_shape = quaternion.shape[:-1]
    return (
        Rotation.from_quat(quaternion.reshape(-1, 4))
        .as_rotvec()
        .reshape(*original_shape, 3)
    )


def relative_rotation_axis_angle(
    current_xyzw: np.ndarray, target_xyzw: np.ndarray
) -> np.ndarray:
    """Return the base-frame left-multiplicative rotation from current to target."""
    from scipy.spatial.transform import Rotation

    current = np.asarray(current_xyzw, dtype=np.float64)
    target = np.asarray(target_xyzw, dtype=np.float64)
    if current.shape != target.shape or current.shape[-1] != 4:
        raise ValueError(
            "current and target quaternions must have the same (..., 4) shape; "
            f"got {current.shape} and {target.shape}."
        )
    original_shape = current.shape[:-1]
    current_rotation = Rotation.from_quat(current.reshape(-1, 4))
    target_rotation = Rotation.from_quat(target.reshape(-1, 4))
    return (
        (target_rotation * current_rotation.inv())
        .as_rotvec()
        .reshape(*original_shape, 3)
    )


def extract_state_from_proprio(
    proprio: np.ndarray, state_token: str = "abs_joint_old"
) -> np.ndarray:
    """Extract the continuous policy state selected by ``state_token``.

    ``none`` disables state injection into the π₀.₅ language prefix. The data
    pipeline still materializes the checkpoint-compatible ``abs_joint_old``
    layout before normalization and padding, even though π₀.₅ does not consume
    it as a suffix token.
    """
    token = validate_state_token(state_token)
    state = np.asarray(proprio)
    if state.shape[-1] < 256:
        raise ValueError(
            f"R1Pro proprio must have at least 256 values; got {state.shape}."
        )

    base = state[..., R1PRO_PROPRIO_INDICES["base_qvel"]]
    trunk = state[..., R1PRO_PROPRIO_INDICES["trunk_qpos"]]
    left_arm = state[..., R1PRO_PROPRIO_INDICES["arm_left_qpos"]]
    right_arm = state[..., R1PRO_PROPRIO_INDICES["arm_right_qpos"]]
    left_gripper = state[..., R1PRO_PROPRIO_INDICES["gripper_left_qpos"]].sum(
        axis=-1, keepdims=True
    )
    right_gripper = state[..., R1PRO_PROPRIO_INDICES["gripper_right_qpos"]].sum(
        axis=-1, keepdims=True
    )

    if token == "abs_eef":
        left_eef = np.concatenate(
            [
                state[..., R1PRO_PROPRIO_INDICES["eef_left_pos"]],
                quaternion_to_axis_angle(
                    state[..., R1PRO_PROPRIO_INDICES["eef_left_quat"]]
                ),
            ],
            axis=-1,
        )
        right_eef = np.concatenate(
            [
                state[..., R1PRO_PROPRIO_INDICES["eef_right_pos"]],
                quaternion_to_axis_angle(
                    state[..., R1PRO_PROPRIO_INDICES["eef_right_quat"]]
                ),
            ],
            axis=-1,
        )
        return np.concatenate(
            [
                base,
                trunk,
                left_eef,
                left_gripper,
                right_eef,
                right_gripper,
            ],
            axis=-1,
        )

    if token in ("none", "abs_joint_old"):
        return np.concatenate(
            [base, trunk, left_arm, right_arm, left_gripper, right_gripper],
            axis=-1,
        )
    return np.concatenate(
        [base, trunk, left_arm, left_gripper, right_arm, right_gripper],
        axis=-1,
    )


def _read_eef_pose(state: np.ndarray, arm: str) -> tuple[np.ndarray, np.ndarray]:
    return (
        state[..., R1PRO_PROPRIO_INDICES[f"eef_{arm}_pos"]],
        state[..., R1PRO_PROPRIO_INDICES[f"eef_{arm}_quat"]],
    )


def _assemble_eef_action(
    source_action: np.ndarray, left_eef: np.ndarray, right_eef: np.ndarray
) -> np.ndarray:
    output = np.empty(21, dtype=np.float64)
    output[0:3] = source_action[_ACTION_SLICES["base"]]
    output[3:7] = source_action[_ACTION_SLICES["trunk"]]
    output[7:13] = left_eef
    output[13] = source_action[_ACTION_SLICES["gripper_left"]]
    output[14:20] = right_eef
    output[20] = source_action[_ACTION_SLICES["gripper_right"]]
    return output


def convert_action(
    action: np.ndarray,
    state: np.ndarray,
    next_state: np.ndarray,
    control_mode: str,
) -> np.ndarray:
    """Convert one recorded absolute-joint action to ``control_mode``.

    Delta labels and absolute EEF labels use achieved state-to-next-state
    motion. At an episode's last frame the caller passes the current state as
    ``next_state``, producing a hold target for both delta modes.
    """
    mode = validate_control_mode(control_mode)
    source_action = np.asarray(action, dtype=np.float64)
    current = np.asarray(state, dtype=np.float64)
    target = np.asarray(next_state, dtype=np.float64)
    if source_action.shape != (23,):
        raise ValueError(
            f"source action must have shape (23,); got {source_action.shape}."
        )
    if current.shape != (256,) or target.shape != (256,):
        raise ValueError(
            "current and next R1Pro states must have shape (256,); got "
            f"{current.shape} and {target.shape}."
        )

    if mode == "abs_joint":
        return source_action.copy()

    if mode == "delta_joint":
        output = source_action.copy()
        output[_ACTION_SLICES["arm_left"]] = (
            target[R1PRO_PROPRIO_INDICES["arm_left_qpos"]]
            - current[R1PRO_PROPRIO_INDICES["arm_left_qpos"]]
        )
        output[_ACTION_SLICES["arm_right"]] = (
            target[R1PRO_PROPRIO_INDICES["arm_right_qpos"]]
            - current[R1PRO_PROPRIO_INDICES["arm_right_qpos"]]
        )
        return output

    eef_actions = []
    for arm in ("left", "right"):
        current_position, current_quaternion = _read_eef_pose(current, arm)
        target_position, target_quaternion = _read_eef_pose(target, arm)
        if mode == "abs_eef":
            eef_actions.append(
                np.concatenate(
                    [
                        target_position,
                        quaternion_to_axis_angle(target_quaternion),
                    ]
                )
            )
        else:
            eef_actions.append(
                np.concatenate(
                    [
                        target_position - current_position,
                        relative_rotation_axis_angle(
                            current_quaternion, target_quaternion
                        ),
                    ]
                )
            )
    return _assemble_eef_action(source_action, eef_actions[0], eef_actions[1])


def convert_episode_actions(
    actions: np.ndarray, states: np.ndarray, control_mode: str
) -> np.ndarray:
    """Convert every action in an episode and preserve the final-frame hold."""
    mode = validate_control_mode(control_mode)
    source_actions = np.asarray(actions, dtype=np.float64)
    proprio = np.asarray(states, dtype=np.float64)
    if (
        source_actions.ndim != 2
        or source_actions.shape[1] != 23
        or proprio.ndim != 2
        or proprio.shape[1] != 256
        or source_actions.shape[0] != proprio.shape[0]
    ):
        raise ValueError(
            "episode arrays must have shapes (frames, 23) and (frames, 256) "
            f"with equal frame counts; got {source_actions.shape} and {proprio.shape}."
        )
    if source_actions.shape[0] == 0:
        raise ValueError("cannot convert an empty episode.")

    output = np.empty(
        (source_actions.shape[0], CONTROL_MODE_ACTION_DIMS[mode]), dtype=np.float64
    )
    for index in range(source_actions.shape[0]):
        next_index = min(index + 1, source_actions.shape[0] - 1)
        output[index] = convert_action(
            source_actions[index], proprio[index], proprio[next_index], mode
        )
    return output


def read_control_mode_manifest(dataset_root: str | Path) -> dict[str, Any] | None:
    """Read a converted dataset's control-mode manifest, if present."""
    path = Path(dataset_root).expanduser() / CONTROL_MODE_MANIFEST
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def write_control_mode_manifest(
    dataset_root: str | Path, manifest: dict[str, Any]
) -> Path:
    """Write a control-mode manifest after validating its canonical fields."""
    mode = validate_control_mode(str(manifest.get("control_mode")))
    expected_dim = CONTROL_MODE_ACTION_DIMS[mode]
    if int(manifest.get("action_env_dim", -1)) != expected_dim:
        raise ValueError(
            "control-mode manifest action_env_dim does not match its mode: "
            f"{manifest.get('action_env_dim')} != {expected_dim}."
        )
    path = Path(dataset_root).expanduser() / CONTROL_MODE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def validate_control_mode_dataset(
    dataset_root: str | Path,
    control_mode: str,
    tasks: list[str] | None = None,
) -> dict[str, Any] | None:
    """Validate dataset width and provenance against an exact control mode.

    A manifest-less 23-dimensional dataset is accepted only for ``abs_joint``
    because that is the native BEHAVIOR representation. Every converted mode
    requires ``meta/control_mode.json``.
    """
    mode = validate_control_mode(control_mode)
    root = Path(dataset_root).expanduser()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise ValueError(f"BEHAVIOR dataset has no metadata file: {info_path}.")
    info = json.loads(info_path.read_text())
    try:
        action_dim = int(info["features"]["action"]["shape"][-1])
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"{info_path} has no valid action feature shape.") from exc
    expected_dim = CONTROL_MODE_ACTION_DIMS[mode]
    if action_dim != expected_dim:
        raise ValueError(
            f"control_mode={mode!r} requires {expected_dim}-dim actions, but "
            f"{info_path} declares {action_dim}."
        )

    manifest = read_control_mode_manifest(root)
    if manifest is None:
        legacy_manifest = root / "meta/eef_delta_provenance.json"
        if legacy_manifest.is_file():
            raise ValueError(
                f"{root} carries the legacy provenance file {legacy_manifest}; "
                "reconvert it to write the exact-mode meta/control_mode.json "
                "manifest before using this migration."
            )
        if mode != "abs_joint":
            raise ValueError(
                f"control_mode={mode!r} requires {root / CONTROL_MODE_MANIFEST}; "
                "convert the source dataset with the control-mode converter."
            )
    else:
        manifest_mode = validate_control_mode(str(manifest.get("control_mode")))
        if manifest_mode != mode:
            raise ValueError(
                f"{root / CONTROL_MODE_MANIFEST} was built for "
                f"{manifest_mode!r}, not {mode!r}."
            )
        if int(manifest.get("action_env_dim", -1)) != expected_dim:
            raise ValueError(
                f"{root / CONTROL_MODE_MANIFEST} declares action_env_dim="
                f"{manifest.get('action_env_dim')}, expected {expected_dim}."
            )
        if tasks:
            covered_tasks = set(manifest.get("tasks") or [])
            missing = sorted(set(tasks) - covered_tasks)
            if missing:
                raise ValueError(
                    f"converted dataset does not cover requested tasks {missing}."
                )

    # Sample the physical action column so stale metadata cannot hide a mismatch.
    sample_paths = sorted((root / "data").glob("*/*.parquet"))
    if sample_paths:
        import pyarrow.parquet as pq

        action_column = pq.read_table(sample_paths[0], columns=["action"])["action"]
        if len(action_column):
            row_dim = len(action_column[0].as_py())
            if row_dim != expected_dim:
                raise ValueError(
                    f"{sample_paths[0]} stores {row_dim}-dim actions, expected "
                    f"{expected_dim} for {mode!r}."
                )
    return manifest
