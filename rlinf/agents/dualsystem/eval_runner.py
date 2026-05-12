# Copyright 2025 The RLinf Authors.
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

"""Dual-system eval runner with full trajectory logging.

Inherits from :class:`EmbodiedEvalRunner` and adds:
  - Per-step trajectory logging to a ``.jsonl`` file (subtasks, actions, rewards).
  - Observation image saving to ``.jpg`` files with configurable ``k``
    (number of images per trajectory to save; -1 for all, default 10).
"""

import json
import os
from pathlib import Path

import numpy as np
import torch

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics


class DualSystemEvalRunner(EmbodiedEvalRunner):
    """Eval runner that logs full trajectories for the dual-system pipeline.

    Extra config keys (under ``cfg.vlm``):
        - ``trajectory_log_dir``: directory for ``.jsonl`` and images.
          Defaults to ``{log_path}/trajectories``.
        - ``k_images``: number of observation images to save per trajectory.
          ``-1`` saves all steps, ``0`` saves none. Default ``10``.
    """

    def __init__(self, cfg, rollout, env, run_timer=None):
        super().__init__(cfg, rollout, env, run_timer=run_timer)

        vlm_cfg = cfg.get("vlm", {})
        log_path = cfg.runner.logger.log_path
        self.trajectory_log_dir = str(
            vlm_cfg.get("trajectory_log_dir", os.path.join(log_path, "trajectories"))
        )
        self.k_images = int(vlm_cfg.get("k_images", 10))

    def init_workers(self):
        rollout_handle = self.rollout.init_worker()
        self.env.init_worker().wait()
        rollout_handle.wait()

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        env_results = env_handle.wait()
        rollout_results = rollout_handle.wait()

        # Standard eval metrics from env.
        eval_metrics_list = [r for r in env_results if r is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)

        # Trajectory logging from rollout workers.
        trajectory_data = [r for r in rollout_results if r is not None]
        if trajectory_data:
            self._log_trajectories(trajectory_data)

        return eval_metrics

    # ------------------------------------------------------------------
    # Trajectory logging
    # ------------------------------------------------------------------

    def _log_trajectories(self, trajectory_data: list):
        """Save per-env trajectory data from all rollout workers to disk.

        Each rollout worker returns a flat list of step records (one per
        chunk-step) where each record's batch dim spans multiple envs and
        possibly multiple trajectories. We expand them here so the JSONL
        contains one record per (worker_idx, env_idx, traj_idx, step_in_traj).

        Args:
            trajectory_data: List of per-worker trajectory lists. Each worker
                returns a ``list[dict]`` with keys ``obs``, ``actions``,
                ``subtasks``, ``vlm_inputs``, ``vlm_outputs``,
                ``env_traj_idx`` (list[int], one per env in batch),
                ``env_step_in_traj`` (list[int], one per env in batch).
        """
        log_dir = Path(self.trajectory_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        img_dir = log_dir / "images"
        if self.k_images != 0:
            img_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = log_dir / "trajectory.jsonl"

        # Collect all records first so we can sort them by
        # (worker_idx, env_idx, traj_idx, step_in_traj).
        all_records: list[dict] = []

        for worker_idx, worker_traj in enumerate(trajectory_data):
            if not isinstance(worker_traj, list) or not worker_traj:
                continue

            # First pass: count how many step records each (env_idx,
            # traj_idx) pair contains, so we can sample image indices
            # uniformly within each trajectory.
            steps_per_traj: dict[tuple[int, int], int] = {}
            for step in worker_traj:
                env_traj_idx = step.get("env_traj_idx") or []
                for env_i, traj_i in enumerate(env_traj_idx):
                    key = (env_i, int(traj_i))
                    steps_per_traj[key] = steps_per_traj.get(key, 0) + 1

            save_set: set[tuple[int, int, int]] = set()
            for (env_i, traj_i), n in steps_per_traj.items():
                indices = self._select_image_indices(n)
                for s in indices:
                    save_set.add((env_i, traj_i, s))

            for step in worker_traj:
                env_traj_idx = step.get("env_traj_idx") or []
                env_step_in_traj = step.get("env_step_in_traj") or []
                batch_size = len(env_traj_idx)
                if batch_size == 0:
                    continue

                vlm_inputs = step.get("vlm_inputs") or {}
                vlm_outputs = step.get("vlm_outputs") or {}
                obs = step.get("obs") or {}
                subtasks_field = step.get("subtasks") or []

                for env_i in range(batch_size):
                    traj_i = int(env_traj_idx[env_i])
                    s_in_traj = int(env_step_in_traj[env_i]) if env_i < len(env_step_in_traj) else 0

                    # Save observation image if selected.
                    image_path = None
                    if (env_i, traj_i, s_in_traj) in save_set:
                        saved = self._save_obs_image(
                            obs, img_dir, worker_idx, env_i, traj_i, s_in_traj
                        )
                        image_path = str(saved) if saved else None

                    record = {
                        "worker_idx": worker_idx,
                        "env_idx": env_i,
                        "traj_idx": traj_i,
                        "step_in_traj": s_in_traj,
                        # VLM turn
                        "vlm_inputs": {
                            "task_descriptions": _index_or_none(
                                vlm_inputs.get("task_descriptions"), env_i
                            ),
                            "input_memories": _index_or_none(
                                vlm_inputs.get("input_memories"), env_i
                            ),
                            "image_path": image_path,
                            "skip": bool(vlm_inputs.get("skip", False)),
                        },
                        "vlm_outputs": {
                            "raw_outputs": _index_or_none(
                                vlm_outputs.get("raw_outputs"), env_i
                            ),
                            "subtasks": _index_or_none(
                                vlm_outputs.get("subtasks"), env_i
                            ),
                            "output_memories": _index_or_none(
                                vlm_outputs.get("output_memories"), env_i
                            ),
                            "skip": bool(vlm_outputs.get("skip", False)),
                        },
                        # VLA turn
                        "vla_inputs": {
                            "subtasks": _index_or_none(subtasks_field, env_i),
                        },
                        "vla_outputs": {
                            "actions": "skip now, it's too long",
                        },
                    }

                    all_records.append(record)

        # Sort by (worker_idx, env_idx, traj_idx, step_in_traj) so each
        # trajectory's steps are contiguous and easy to read.
        all_records.sort(
            key=lambda r: (r["worker_idx"], r["env_idx"], r["traj_idx"], r["step_in_traj"])
        )

        with open(jsonl_path, "a") as f:
            for record in all_records:
                f.write(json.dumps(record) + "\n")

        self.logger.info(
            "Trajectory logged to %s (%d workers)", jsonl_path, len(trajectory_data)
        )

    def _select_image_indices(self, n_steps: int) -> set[int]:
        """Return the step indices whose images should be saved."""
        if self.k_images == 0:
            return set()
        if self.k_images < 0 or self.k_images >= n_steps:
            return set(range(n_steps))
        # Uniformly sample k indices across the trajectory.
        indices = np.linspace(0, n_steps - 1, self.k_images, dtype=int)
        return set(indices.tolist())

    @staticmethod
    def _actions_to_list(actions) -> list | None:
        """Convert action tensor/array to a JSON-serializable list."""
        if actions is None:
            return None
        if isinstance(actions, torch.Tensor):
            return actions.tolist()
        if isinstance(actions, np.ndarray):
            return actions.tolist()
        return actions

    @staticmethod
    def _save_obs_image(
        obs: dict | None,
        img_dir: Path,
        worker_idx: int,
        env_idx: int,
        traj_idx: int,
        step_in_traj: int,
    ) -> Path | None:
        """Save the main observation image for one env as a .jpg file.

        Filename format: ``w{worker}_e{env}_t{traj}_s{step:04d}.jpg``.
        """
        if obs is None:
            return None
        main_images = obs.get("main_images")
        if main_images is None:
            return None

        from PIL import Image

        # Slice the env at index ``env_idx`` from a [B, H, W, C] batch.
        if isinstance(main_images, torch.Tensor):
            if env_idx >= main_images.shape[0]:
                return None
            img_arr = main_images[env_idx].cpu().numpy()
        elif isinstance(main_images, np.ndarray):
            if env_idx >= main_images.shape[0]:
                return None
            img_arr = main_images[env_idx]
        else:
            return None

        if img_arr.dtype != np.uint8:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
            else:
                img_arr = img_arr.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(img_arr)
        filename = f"w{worker_idx}_e{env_idx}_t{traj_idx}_s{step_in_traj:04d}.jpg"
        save_path = img_dir / filename
        img.save(save_path)
        return save_path


def _index_or_none(value, idx: int):
    """Safely extract ``value[idx]`` from a list-like, returning ``None``
    when ``value`` is ``None`` or the index is out of bounds.

    Used to slice batched VLM I/O fields down to a single env for the JSONL.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if 0 <= idx < len(value):
            return value[idx]
        return None
    return value
