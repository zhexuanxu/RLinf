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

"""Deterministic dataset-action replay for the BEHAVIOR environment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.envs.behavior.control_modes import (
    CONTROL_MODE_ACTION_DIMS,
    validate_control_mode,
    validate_control_mode_dataset,
)


@dataclass(frozen=True)
class BehaviorReplayEpisode:
    """Explicit pairing of one cached task instance and one dataset episode."""

    activity_instance_id: int
    episode_index: int


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative, got {result}.")
    return result


def resolve_replay_episode(
    replay_cfg: Any,
    task_cfg: Any,
) -> BehaviorReplayEpisode:
    """Validate and resolve the explicit replay instance/episode pair.

    Replay deliberately does not infer an episode from an instance ID. Dataset
    episode numbering is not a stable public contract, so both identifiers must
    be selected explicitly. The BEHAVIOR loader must also be in ``disabled``
    resampling mode; otherwise it could reset to a random cached instance while
    replaying the configured episode.
    """

    instance_id = _nonnegative_integer(
        replay_cfg.get("activity_instance_id"),
        "env.eval.replay.activity_instance_id",
    )
    episode_index = _nonnegative_integer(
        replay_cfg.get("episode_index"),
        "env.eval.replay.episode_index",
    )
    configured_instance_id = _nonnegative_integer(
        task_cfg.get("activity_instance_id"),
        "env.eval.omni_config.task.activity_instance_id",
    )
    if configured_instance_id != instance_id:
        raise ValueError(
            "Replay instance mismatch: env.eval.replay.activity_instance_id="
            f"{instance_id}, but env.eval.omni_config.task.activity_instance_id="
            f"{configured_instance_id}."
        )
    resample_mode = task_cfg.get("instance_resample_mode")
    if resample_mode != "disabled":
        raise ValueError(
            "BEHAVIOR replay requires "
            "env.eval.omni_config.task.instance_resample_mode=disabled; "
            f"got {resample_mode!r}."
        )
    if bool(task_cfg.get("online_object_sampling", False)):
        raise ValueError(
            "BEHAVIOR replay requires "
            "env.eval.omni_config.task.online_object_sampling=false."
        )
    return BehaviorReplayEpisode(
        activity_instance_id=instance_id,
        episode_index=episode_index,
    )


class BehaviorReplayActionSource:
    """Load and chunk one episode from a BEHAVIOR LeRobot dataset.

    Args:
        dataset_root: Root containing ``meta/info.json`` and episode Parquet
            files.
        control_mode: Exact public control mode used by the BEHAVIOR controller.
        num_action_chunks: Number of recorded control steps sent per env chunk.
        num_envs: Number of environments receiving the trajectory.
        activity_name: BEHAVIOR activity expected for the selected episode.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        control_mode: str,
        num_action_chunks: int,
        num_envs: int,
        activity_name: str,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.control_mode = validate_control_mode(control_mode)
        self.action_env_dim = CONTROL_MODE_ACTION_DIMS[self.control_mode]
        self.num_action_chunks = _nonnegative_integer(
            num_action_chunks, "num_action_chunks"
        )
        self.num_envs = _nonnegative_integer(num_envs, "num_envs")
        if self.num_action_chunks == 0:
            raise ValueError("num_action_chunks must be greater than zero.")
        if self.num_envs == 0:
            raise ValueError("num_envs must be greater than zero.")
        self.activity_name = str(activity_name)
        if not self.activity_name:
            raise ValueError("activity_name must not be empty.")

        validate_control_mode_dataset(
            self.dataset_root,
            self.control_mode,
            tasks=[self.activity_name],
        )
        info_path = self.dataset_root / "meta/info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        try:
            self.data_path_template = str(info["data_path"])
            self.chunks_size = int(info.get("chunks_size", 10000))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{info_path} has invalid episode path metadata.") from exc
        if self.chunks_size <= 0:
            raise ValueError(f"{info_path} declares invalid chunks_size.")

    def _parquet_path(self, episode_index: int) -> Path:
        episode_chunk = episode_index // self.chunks_size
        relative_path = Path(
            self.data_path_template.format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
            )
        )
        if relative_path.is_absolute():
            raise ValueError(
                f"Replay data_path must be relative, got {self.data_path_template!r}."
            )
        parquet_path = (self.dataset_root / relative_path).resolve()
        if not parquet_path.is_relative_to(self.dataset_root):
            raise ValueError(
                f"Replay data_path escapes dataset_root: {self.data_path_template!r}."
            )
        return parquet_path

    def _validate_episode_activity(self, episode_index: int) -> None:
        tasks_path = self.dataset_root / "meta/tasks.jsonl"
        episodes_path = self.dataset_root / "meta/episodes.jsonl"
        if not tasks_path.is_file() or not episodes_path.is_file():
            raise ValueError(
                "Replay requires meta/tasks.jsonl and meta/episodes.jsonl to "
                "verify the selected episode's activity."
            )

        expected_labels = {self.activity_name}
        with tasks_path.open(encoding="utf-8") as tasks_file:
            for line in tasks_file:
                record = json.loads(line)
                if record.get("task_name") == self.activity_name:
                    task_prompt = record.get("task")
                    if task_prompt:
                        expected_labels.add(str(task_prompt))
                    break
            else:
                raise ValueError(
                    f"{tasks_path} has no activity {self.activity_name!r}."
                )

        selected_episode = None
        with episodes_path.open(encoding="utf-8") as episodes_file:
            for line in episodes_file:
                record = json.loads(line)
                if record.get("episode_index") == episode_index:
                    selected_episode = record
                    break
        if selected_episode is None:
            raise ValueError(f"{episodes_path} has no episode_index={episode_index}.")

        episode_tasks = {str(value) for value in selected_episode.get("tasks", [])}
        if expected_labels.isdisjoint(episode_tasks):
            raise ValueError(
                f"Episode {episode_index} belongs to {sorted(episode_tasks)}, "
                f"not activity {self.activity_name!r}."
            )

    def load_episode(self, episode_index: int) -> np.ndarray:
        """Load an explicitly selected episode's actions.

        The cached simulator instance and demonstration episode are an explicit
        caller-selected pair. Replay does not restore the demonstration's
        recorded observation state, so loading its proprio column would neither
        initialize nor validate the cached scene.
        """

        episode_index = _nonnegative_integer(episode_index, "episode_index")
        self._validate_episode_activity(episode_index)
        parquet_path = self._parquet_path(episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(
                f"Replay episode {episode_index} does not exist: {parquet_path}."
            )

        import pyarrow.parquet as pq

        schema_names = pq.read_schema(parquet_path).names
        if "action" not in schema_names:
            raise ValueError(f"{parquet_path} has no action column.")
        table = pq.read_table(parquet_path, columns=["action"])
        if table.num_rows == 0:
            raise ValueError(f"Replay episode {episode_index} is empty.")
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
        expected_shape = (table.num_rows, self.action_env_dim)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Episode {episode_index} stores action shape {actions.shape}; "
                f"expected {expected_shape} for control_mode={self.control_mode!r}."
            )

        return actions

    def episode_to_chunks(
        self,
        actions: np.ndarray,
        n_eval_chunk_steps: int,
    ) -> np.ndarray:
        """Return replay chunks shaped ``[steps, envs, horizon, action_dim]``.

        Episodes longer than the evaluation budget are truncated. Shorter
        episodes hold their final recorded target for the remaining steps.
        """

        n_eval_chunk_steps = _nonnegative_integer(
            n_eval_chunk_steps, "n_eval_chunk_steps"
        )
        if n_eval_chunk_steps == 0:
            raise ValueError("n_eval_chunk_steps must be greater than zero.")
        episode_actions = np.asarray(actions, dtype=np.float32)
        if episode_actions.ndim != 2 or episode_actions.shape[1] != self.action_env_dim:
            raise ValueError(
                "actions must have shape "
                f"(frames, {self.action_env_dim}); got {episode_actions.shape}."
            )
        if episode_actions.shape[0] == 0:
            raise ValueError("Cannot replay an empty action sequence.")

        total_steps = n_eval_chunk_steps * self.num_action_chunks
        if episode_actions.shape[0] < total_steps:
            hold_action = episode_actions[-1:].copy()
            # R1Pro base channels are velocity commands in every control mode.
            # Replaying the final recorded velocity would keep driving after the
            # demonstration ends; zero only those channels while holding all
            # position/pose/gripper targets representation-appropriately.
            hold_action[:, :3] = 0.0
            padding = np.repeat(
                hold_action,
                total_steps - episode_actions.shape[0],
                axis=0,
            )
            episode_actions = np.concatenate([episode_actions, padding], axis=0)
        else:
            episode_actions = episode_actions[:total_steps]
        chunks = episode_actions.reshape(
            n_eval_chunk_steps,
            self.num_action_chunks,
            self.action_env_dim,
        )
        return np.broadcast_to(
            chunks[:, None],
            (
                n_eval_chunk_steps,
                self.num_envs,
                self.num_action_chunks,
                self.action_env_dim,
            ),
        ).copy()
