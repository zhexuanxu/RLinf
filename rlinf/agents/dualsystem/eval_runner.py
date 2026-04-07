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
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
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
        """Save trajectory data from all rollout workers to disk.

        Args:
            trajectory_data: List of per-worker trajectory lists.  Each worker
                returns a ``list[dict]`` with keys ``obs``, ``actions``,
                ``subtasks``.
        """
        log_dir = Path(self.trajectory_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        img_dir = log_dir / "images"
        if self.k_images != 0:
            img_dir.mkdir(parents=True, exist_ok=True)

        jsonl_path = log_dir / "trajectory.jsonl"

        with open(jsonl_path, "a") as f:
            for worker_idx, worker_traj in enumerate(trajectory_data):
                if not isinstance(worker_traj, list):
                    continue
                n_steps = len(worker_traj)

                # Decide which step indices get images saved.
                save_indices = self._select_image_indices(n_steps)

                for step_idx, step in enumerate(worker_traj):
                    # Save observation image if selected.
                    image_path = None
                    if step_idx in save_indices:
                        saved = self._save_obs_image(
                            step.get("obs"), img_dir, worker_idx, step_idx
                        )
                        image_path = str(saved) if saved else None

                    # --- VLM inputs / outputs ---
                    vlm_inputs = step.get("vlm_inputs") or {}
                    vlm_outputs = step.get("vlm_outputs") or {}

                    record = {
                        "worker": worker_idx,
                        "step": step_idx,
                        # VLM turn
                        "vlm_inputs": {
                            "task_descriptions": vlm_inputs.get("task_descriptions"),
                            "input_memories": vlm_inputs.get("input_memories"),
                            "image_path": image_path,
                        },
                        "vlm_outputs": {
                            "raw_outputs": vlm_outputs.get("raw_outputs"),
                            "subtasks": vlm_outputs.get("subtasks"),
                            "output_memories": vlm_outputs.get("output_memories"),
                        },
                        # VLA turn
                        "vla_inputs": {
                            "subtasks": step.get("subtasks"),
                        },
                        "vla_outputs": {
                            "actions": "skip now, it's too long" # self._actions_to_list(step.get("actions"))
                        },
                    }

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
        step_idx: int,
    ) -> Path | None:
        """Save the main observation image as a .jpg file."""
        if obs is None:
            return None
        main_images = obs.get("main_images")
        if main_images is None:
            return None

        from PIL import Image

        # Take the first env in the batch: [B, H, W, C] -> [H, W, C]
        if isinstance(main_images, torch.Tensor):
            img_arr = main_images[0].cpu().numpy()
        elif isinstance(main_images, np.ndarray):
            img_arr = main_images[0]
        else:
            return None

        if img_arr.dtype != np.uint8:
            if img_arr.max() <= 1.0:
                img_arr = (img_arr * 255).clip(0, 255).astype(np.uint8)
            else:
                img_arr = img_arr.clip(0, 255).astype(np.uint8)

        img = Image.fromarray(img_arr)
        filename = f"w{worker_idx}_s{step_idx:04d}.jpg"
        save_path = img_dir / filename
        img.save(save_path)
        return save_path
