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

"""Lightweight rollout worker that injects recorded BEHAVIOR actions."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.envs.behavior.replay import (
    BehaviorReplayActionSource,
    resolve_replay_episode,
)
from rlinf.scheduler import Channel, Worker


class BehaviorReplayRolloutWorker(Worker):
    """Maintain eval channel lifecycle without constructing a policy model."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if not bool(OmegaConf.select(cfg, "env.eval.replay.enabled", default=False)):
            raise ValueError(
                "BehaviorReplayRolloutWorker requires env.eval.replay.enabled=true."
            )
        if not bool(cfg.runner.get("only_eval", False)):
            raise ValueError("BEHAVIOR replay is supported only for standalone eval.")
        if cfg.env.eval.env_type != "behavior":
            raise ValueError("Dataset-action replay is supported only for BEHAVIOR.")
        if cfg.runner.get("enable_decoupled_mode", False):
            raise ValueError("BEHAVIOR replay does not support decoupled mode.")
        if int(cfg.rollout.pipeline_stage_num) != 1:
            raise ValueError("BEHAVIOR replay requires rollout.pipeline_stage_num=1.")
        if int(cfg.env.eval.total_num_envs) != 1:
            raise ValueError("BEHAVIOR replay requires env.eval.total_num_envs=1.")
        if self._world_size != 1:
            raise ValueError("BEHAVIOR replay requires exactly one rollout worker.")

        self.model_cfg = cfg.rollout.model
        self.eval_rollout_epoch = int(cfg.env.eval.rollout_epoch)
        self.eval_batch_size = 1
        self.n_eval_chunk_steps = int(cfg.env.eval.max_steps_per_rollout_epoch) // int(
            self.model_cfg.num_action_chunks
        )
        self._replay_chunks: np.ndarray | None = None

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict) -> int:
        obs = obs_batch.get("obs", obs_batch)
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, (np.ndarray, torch.Tensor, list)):
                return len(value)
        raise ValueError("Cannot infer batch size from BEHAVIOR observations.")

    def init_worker(self) -> None:
        """Load replay actions without loading OpenPI or normalization assets."""

        replay_cfg = self.cfg.env.eval.replay
        task_cfg = self.cfg.env.eval.omni_config.task
        episode = resolve_replay_episode(replay_cfg, task_cfg)
        control_mode = str(self.model_cfg.openpi.control_mode)
        action_source = BehaviorReplayActionSource(
            dataset_root=replay_cfg.dataset_root,
            control_mode=control_mode,
            num_action_chunks=int(self.model_cfg.num_action_chunks),
            num_envs=self.eval_batch_size,
            activity_name=str(task_cfg.activity_name),
        )
        if int(self.model_cfg.action_dim) != action_source.action_env_dim:
            raise ValueError(
                f"rollout.model.action_dim={self.model_cfg.action_dim} does not "
                f"match control_mode={control_mode!r} replay width "
                f"{action_source.action_env_dim}."
            )
        actions = action_source.load_episode(episode.episode_index)
        self._replay_chunks = action_source.episode_to_chunks(
            actions,
            self.n_eval_chunk_steps,
        )
        self.log_info(
            "Loaded BEHAVIOR replay pair "
            f"(activity_instance_id={episode.activity_instance_id}, "
            f"episode_index={episode.episode_index}) from "
            f"{action_source.dataset_root}: {actions.shape[0]} recorded steps, "
            f"{self.n_eval_chunk_steps} action chunks."
        )

    async def evaluate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ) -> None:
        """Consume observations and respond with the next recorded action chunk."""

        if self._replay_chunks is None:
            raise RuntimeError("init_worker() must run before replay evaluation.")
        for _ in range(self.eval_rollout_epoch):
            for eval_step in range(self.n_eval_chunk_steps):
                await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                    infer_batch_size_fn=self._infer_env_batch_size,
                ).async_wait()
                actions = torch.from_numpy(self._replay_chunks[eval_step])
                self.send_to(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=actions,
                    tag="eval_rollout_results",
                    route_key=0,
                    async_op=True,
                    batch_size=self.eval_batch_size,
                )
