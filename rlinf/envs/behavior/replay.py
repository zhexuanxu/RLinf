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

"""Dataset-action replay source for the BEHAVIOR env.

Feeds recorded LeRobot dataset actions into the environment instead of
actor-generated actions, so we can verify that the recorded (``joint_absolute``,
23-dim) or converted (``eef_delta_pose``, 21-dim) actions reproduce the
demonstrated task when executed by the *real* OmniGibson controller (JointController
or InverseKinematicsController). The env worker resets to the matching cached
``tro_state`` initial state and this source supplies the episode's action chunks.

Instance <-> episode mapping: the 2025-challenge demos were cached as per-instance
``tro_state`` files with ``activity_instance_id = k`` corresponding to
``episode_index = k * episode_stride`` (default 10), i.e. instance 1 -> episode
``episode_00000010``, instance 2 -> ``episode_00000020``, and so on.

Pure numpy + pyarrow (no torch / OmniGibson / Isaac imports), so importing this
module is cheap and side-effect-free.
"""

from __future__ import annotations

import json
import os

import numpy as np


class BehaviorReplayActionSource:
    """Load recorded episode actions from a LeRobot dataset and hand them to the
    env worker as ``[num_envs, num_action_chunks, action_env_dim]`` chunks.

    Args:
        dataset_root: LeRobot dataset root (original 23-dim joint dataset for
            joint replay, or the converted 21-dim delta-EEF dataset for delta
            replay).
        action_env_dim: Expected meaningful action width (23 or 21); validated
            against the dataset's ``meta/info.json`` and each episode.
        num_action_chunks: Env chunk length (pi05 = 32).
        num_envs: Number of envs per stage (replay uses 1).
        episode_stride: ``episode_index = instance_id * episode_stride``.
    """

    def __init__(
        self,
        dataset_root: str,
        action_env_dim: int,
        num_action_chunks: int,
        num_envs: int,
        episode_stride: int = 10,
    ):
        self.dataset_root = os.path.abspath(dataset_root)
        self.action_env_dim = int(action_env_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_envs = int(num_envs)
        self.episode_stride = int(episode_stride)

        info_path = os.path.join(self.dataset_root, "meta", "info.json")
        if not os.path.isfile(info_path):
            raise FileNotFoundError(
                f"replay dataset_root has no meta/info.json: {self.dataset_root}"
            )
        info = json.load(open(info_path))
        self.data_path_template = info["data_path"]
        self.chunks_size = int(info.get("chunks_size", 10000))
        ds_width = int(info["features"]["action"]["shape"][-1])
        if ds_width != self.action_env_dim:
            raise ValueError(
                f"replay dataset action width {ds_width} != action_env_dim "
                f"{self.action_env_dim} (dataset_root={self.dataset_root})."
            )

    def episode_index(self, instance_id: int) -> int:
        return int(instance_id) * self.episode_stride

    def _parquet_path(self, episode_index: int) -> str:
        chunk = episode_index // self.chunks_size
        rel = self.data_path_template.format(
            episode_chunk=chunk, episode_index=episode_index
        )
        return os.path.join(self.dataset_root, rel)

    def load_episode(self, instance_id: int):
        """Return ``(actions[T, action_env_dim] float32, first_state or None)`` for
        the episode mapped from ``instance_id``."""
        import pyarrow.parquet as pq

        ei = self.episode_index(instance_id)
        path = self._parquet_path(ei)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"replay episode parquet not found for instance {instance_id} "
                f"(episode_index {ei}): {path}"
            )
        schema_names = pq.read_schema(path).names
        want = [c for c in ("action", "observation.state") if c in schema_names]
        pdf = pq.read_table(path, columns=want).to_pandas()
        actions = np.stack(
            [np.asarray(a, dtype=np.float32) for a in pdf["action"]]
        )
        if actions.ndim != 2 or actions.shape[-1] != self.action_env_dim:
            raise ValueError(
                f"episode {ei} action shape {actions.shape} incompatible with "
                f"action_env_dim {self.action_env_dim}."
            )
        first_state = None
        if "observation.state" in pdf:
            first_state = np.asarray(pdf["observation.state"].iloc[0], dtype=np.float64)
        return actions, first_state

    def episode_to_chunks(
        self, actions: np.ndarray, n_eval_chunk_steps: int
    ) -> np.ndarray:
        """Reshape ``[T, dim]`` into ``[n_eval_chunk_steps, num_envs, num_action_chunks, dim]``.

        The tail is padded by repeating the last recorded action (hold pose) when
        the episode is shorter than ``n_eval_chunk_steps * num_action_chunks``;
        longer episodes are truncated. The same trajectory is broadcast to all
        envs (replay uses a single env).
        """
        total = n_eval_chunk_steps * self.num_action_chunks
        t = actions.shape[0]
        if t < total:
            pad = np.repeat(actions[-1:], total - t, axis=0)
            actions = np.concatenate([actions, pad], axis=0)
        else:
            actions = actions[:total]
        chunked = actions.reshape(
            n_eval_chunk_steps, self.num_action_chunks, self.action_env_dim
        )
        chunked = np.broadcast_to(
            chunked[:, None],
            (n_eval_chunk_steps, self.num_envs, self.num_action_chunks, self.action_env_dim),
        ).copy()
        return chunked.astype(np.float32)
