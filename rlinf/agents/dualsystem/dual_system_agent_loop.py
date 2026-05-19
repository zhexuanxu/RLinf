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

"""Dual-system agentloop coordination logic for embodied eval.

This module coordinates the high-level VLM and low-level VLA in a two-turn
pipeline:

  Turn 1 — VLM: obs + task_description (+ memory)  ->  subtask (+ updated memory)
  Turn 2 — VLA: obs + subtask                      ->  action tensor

Memory module (optional)
~~~~~~~~~~~~~~~~~~~~~~~~
When ``enable_memory=True``, the agentloop maintains a per-environment
language memory following the MEM formulation:

    pi_HL(l_{t+1}, m_{t+1} | o_t, m_t, g)

At each step the VLM receives the task goal *g*, previous memory *m_t*,
and current observation *o_t*, and jointly produces the next subtask
*l_{t+1}* and an updated memory *m_{t+1}*.  The memory is a compressed
natural-language summary of all semantically relevant events so far.

When ``enable_memory=False`` (default), the VLM simply outputs a subtask
string with no memory state — identical to the original memoryless mode.
"""

import logging
from typing import Any, Literal

import numpy as np
import torch
import copy
from rlinf.agents.dualsystem.prompts import (
    build_vlm_user_text,
    parse_vlm_output,
)

logger = logging.getLogger(__name__)


class DualSystemAgentLoop:
    """Coordinates VLM and VLA for one agentloop step.

    Args:
        vlm_model: VLM exposing ``generate_subtask(obs, prompt, **kwargs)``.
        vla_model: VLA exposing ``predict_action_batch(env_obs, **kwargs)``.
        log_subtasks: Log the first subtask at each step.
        enable_reasoning: Expect ``<think>`` block in VLM output.
        enable_memory: Include Old Memory in prompt and expect ``<memory>``
            block in VLM output.
        vlm_sampling_params: Generation kwargs forwarded to the VLM.
    """

    def __init__(
        self,
        vlm_model,
        vla_model,
        log_subtasks: bool = False,
        enable_reasoning: bool = False,
        enable_memory: bool = False,
        vlm_sampling_params: dict | None = None,
        frequency: int = 1,
    ):
        self.vlm_model = vlm_model
        self.vla_model = vla_model
        self.log_subtasks = log_subtasks
        self.enable_reasoning = enable_reasoning
        self.enable_memory = enable_memory
        self.vlm_sampling_params = vlm_sampling_params or {}
        # How many run_step() calls between successive VLM invocations.
        # ``frequency=1`` (default) calls the VLM every step (original behavior).
        # ``frequency=N`` calls the VLM once every N steps and reuses the
        # cached subtask for the remaining (N-1) steps.
        self.frequency = max(1, int(frequency))

        # Per-environment memory strings.  Initialised lazily on the first
        # call to run_step (when batch size is known) or via reset_memory().
        self._memories: list[str] | None = None

        # Counter and per-env caches for VLM-frequency control.
        self._vlm_step_counter: int = 0
        self._cached_subtasks: list[str] | None = None
        self._cached_raw_outputs: list[str] | None = None
        self._cached_input_memories: list[str] | None = None
        self._cached_prompts: list[str] | None = None

        # Per-environment trajectory tracking (used by the eval runner to
        # tag JSONL records). Lazily initialised in ``run_step`` once the
        # batch size is known. ``_traj_idx[i]`` is the current trajectory
        # index for env i within this rollout worker; it is bumped by
        # ``reset_memory`` (epoch boundary, all envs) and
        # ``reset_memory_for_envs`` (auto-reset, only the affected envs).
        # ``_step_in_traj[i]`` counts steps emitted for env i in its current
        # trajectory and is incremented at the end of each ``run_step``.
        self._traj_idx: list[int] | None = None
        self._step_in_traj: list[int] | None = None

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def reset_memory(self, batch_size: int | None = None):
        """Reset the memory state and cached VLM outputs for all environments.

        Call this between evaluation epochs / episodes so that memory and
        cached subtasks from a previous episode do not leak into the next one.

        Side effect: when the trajectory tracker has already been initialised
        (i.e. this is not the very first reset), every env's ``traj_idx`` is
        bumped by 1 and ``step_in_traj`` is zeroed. This is how an
        ``auto_reset=False`` epoch boundary is recorded as a new trajectory.
        """
        if batch_size is not None:
            self._memories = ["" for _ in range(batch_size)]
        elif self._memories is not None:
            self._memories = ["" for _ in range(len(self._memories))]
        else:
            self._memories = None

        # Invalidate the VLM-frequency cache so the next run_step re-plans.
        self._vlm_step_counter = 0
        self._cached_subtasks = None
        self._cached_raw_outputs = None
        self._cached_input_memories = None
        self._cached_prompts = None

        # Bump the trajectory index for every env IF the tracker has been
        # initialised. The first reset (before any run_step) leaves
        # ``_traj_idx`` as None and is handled lazily on first run_step.
        if self._traj_idx is not None:
            for i in range(len(self._traj_idx)):
                self._traj_idx[i] += 1
                self._step_in_traj[i] = 0

    def reset_memory_for_envs(self, done_mask: torch.Tensor):
        """Selectively reset memory and cached VLM outputs for done envs.

        This is called on each eval step when the env auto-resets done
        episodes, so that memory and cached subtasks from a finished episode
        do not bleed into the new episode that starts in the same batch slot.

        Side effect: every env marked done in ``done_mask`` has its
        ``traj_idx`` incremented and ``step_in_traj`` zeroed, so the next
        ``run_step`` records the upcoming chunk under a new trajectory.

        Args:
            done_mask: Boolean tensor of shape ``[B]`` (or ``[B, C]`` where
                the last column ``[:, -1]`` indicates the final done state).
                ``True`` entries have their memory and cache cleared.
        """
        # Handle [B, C] dones — use the last chunk-step column.
        if done_mask.dim() > 1:
            done_mask = done_mask[:, -1]

        if self._memories is not None:
            for i in range(min(len(self._memories), done_mask.shape[0])):
                if done_mask[i].item():
                    self._memories[i] = ""

        # Mark the cached subtask as stale for done envs by setting it to None.
        # ``run_step`` will detect this and force a fresh VLM call next step.
        if self._cached_subtasks is not None:
            for i in range(min(len(self._cached_subtasks), done_mask.shape[0])):
                if done_mask[i].item():
                    self._cached_subtasks[i] = None
                    if self._cached_raw_outputs is not None and i < len(self._cached_raw_outputs):
                        self._cached_raw_outputs[i] = None
                    if self._cached_input_memories is not None and i < len(self._cached_input_memories):
                        self._cached_input_memories[i] = None
                    if self._cached_prompts is not None and i < len(self._cached_prompts):
                        self._cached_prompts[i] = None

        # Bump trajectory index for envs that just finished. The next
        # ``run_step`` will record the upcoming chunk under traj_idx+1.
        if self._traj_idx is not None:
            for i in range(min(len(self._traj_idx), done_mask.shape[0])):
                if done_mask[i].item():
                    self._traj_idx[i] += 1
                    self._step_in_traj[i] = 0

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run_step(
        self,
        obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        vla_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute one dual-system step.

        Args:
            obs: Environment observation dict.
            mode: Passed through to VLA when ``vla_kwargs`` is ``None``.
            vla_kwargs: If provided, forwarded to
                ``vla_model.predict_action_batch()`` instead of ``mode``.

        Returns:
            ``(actions, result)`` where ``result`` contains at least
            ``"subtasks"`` and, when memory is enabled, ``"memories"``.
        """
        assert obs['wrist_images'].shape[0] == obs['main_images'].shape[0], "Batch size of wrist_images and main_images must match."
        batch_size = obs['main_images'].shape[0]
        task_descriptions = obs["task_descriptions"]
        assert len(task_descriptions) == batch_size, "Length of task_descriptions must match batch size."

        # Lazy-init the per-env trajectory tracker on the first run_step call
        # (or whenever the batch size changes). The first call always records
        # under traj_idx=0; subsequent epoch boundaries / auto-resets bump it
        # via reset_memory / reset_memory_for_envs.
        if self._traj_idx is None or len(self._traj_idx) != batch_size:
            self._traj_idx = [0] * batch_size
            self._step_in_traj = [0] * batch_size

        # Decide whether this step needs a fresh VLM call. The VLM is called
        # whenever (a) the global step counter hits a frequency boundary, OR
        # (b) the cache is empty / has the wrong batch size, OR (c) any env's
        # cached subtask was invalidated by a memory reset (set to None).
        cache_valid = (
            self._cached_subtasks is not None
            and len(self._cached_subtasks) == batch_size
            and all(s is not None for s in self._cached_subtasks)
        )
        on_frequency_boundary = (self._vlm_step_counter % self.frequency == 0)
        call_vlm = on_frequency_boundary or not cache_valid
        self._vlm_step_counter += 1

        input_memories: list[str] | None = None
        prompts: list[str] | None = None
        raw_outputs: list[str] = []
        subtasks: list[str]

        if call_vlm:
            # ----- Build per-sample prompts (unified via build_vlm_user_text) ----- #
            if self.enable_memory:
                if self._memories is None or len(self._memories) != batch_size:
                    self._memories = ["" for _ in range(batch_size)]
                input_memories = list(self._memories)
            else:
                input_memories = None

            prompts = []
            for i in range(batch_size):
                td = task_descriptions[i] if i < len(task_descriptions) else ""
                mem = self._memories[i] if self._memories else ""
                prompts.append(build_vlm_user_text(
                    task_description=td,
                    memory=mem,
                    enable_memory=self.enable_memory,
                ))

            # ----- Turn 1: VLM generates subtask (+ optional reasoning/memory) ----- #
            raw_outputs = self.vlm_model.generate_subtask(
                obs, prompt=prompts, **self.vlm_sampling_params
            )

            # ----- Parse outputs (unified via parse_vlm_output) ----- #
            subtasks = []
            reasonings = []
            for i, raw in enumerate(raw_outputs):
                reasoning, memory, subtask = parse_vlm_output(raw)
                subtasks.append(subtask)
                reasonings.append(reasoning)
                if self.enable_memory and memory:
                    self._memories[i] = memory

            # Refresh the per-env cache so subsequent skipped turns can reuse.
            self._cached_subtasks = list(subtasks)
            self._cached_raw_outputs = list(raw_outputs)
            self._cached_input_memories = list(input_memories) if input_memories is not None else None
            self._cached_prompts = list(prompts)
            skip = False
        else:
            # Reuse the cached VLM output. We still emit one VLM-style
            # entry into the result so the JSONL records have one VLM turn
            # per agentloop step (with skip=True).
            subtasks = list(self._cached_subtasks)
            raw_outputs = list(self._cached_raw_outputs) if self._cached_raw_outputs is not None else []
            reasonings = []
            input_memories = (
                list(self._cached_input_memories)
                if self._cached_input_memories is not None
                else None
            )
            prompts = list(self._cached_prompts) if self._cached_prompts is not None else None
            skip = True

        if self.log_subtasks and subtasks:
            logger.info(
                "[DualSystem] VLM subtask (skip=%s): %s", skip, subtasks[0]
            )
            if self.enable_memory and self._memories:
                logger.info("[DualSystem] Memory: %s", self._memories[0][:200])

        # ----- Turn 2: VLA generates actions ----- #
        obs_with_subtask = copy.deepcopy(dict(obs))
        obs_with_subtask["task_descriptions"] = subtasks

        if vla_kwargs is not None:
            actions, vla_result = self.vla_model.predict_action_batch(
                env_obs=obs_with_subtask, **vla_kwargs
            )
        else:
            actions, vla_result = self.vla_model.predict_action_batch(
                env_obs=obs_with_subtask, mode=mode
            )

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        # ----- Assemble result with detailed logging ----- #
        vla_result["subtasks"] = subtasks

        vla_result["vlm_inputs"] = {
            "task_descriptions": list(task_descriptions),
            "prompts": prompts,
            "input_memories": input_memories,
            "skip": skip,
        }

        vla_result["vlm_outputs"] = {
            "raw_outputs": list(raw_outputs),
            "subtasks": list(subtasks),
            "output_memories": list(self._memories) if self.enable_memory else None,
            "reasonings": reasonings if reasonings else None,
            "skip": skip,
        }

        if self.enable_memory:
            vla_result["memories"] = list(self._memories)

        # Snapshot per-env trajectory tracking BEFORE bumping step_in_traj
        # so the recorded value matches what the eval runner needs.
        vla_result["env_traj_idx"] = list(self._traj_idx)
        vla_result["env_step_in_traj"] = list(self._step_in_traj)

        # Bump step_in_traj for every env now that this step has been
        # recorded. The next run_step will see step_in_traj+1.
        for i in range(len(self._step_in_traj)):
            self._step_in_traj[i] += 1

        return actions, vla_result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_batch_size(obs: dict[str, Any]) -> int:
        for key in ("main_images", "states", "task_descriptions"):
            val = obs.get(key)
            if isinstance(val, torch.Tensor):
                return val.shape[0]
            if isinstance(val, (list, tuple)):
                return len(val)
        raise ValueError("Cannot infer batch size from obs dict.")

    @staticmethod
    def _slice_obs(obs: dict[str, Any], idx: int) -> dict[str, Any]:
        """Slice a single sample from a batched obs dict."""
        single: dict[str, Any] = {}
        for key, val in obs.items():
            if isinstance(val, torch.Tensor):
                single[key] = val[idx : idx + 1]
            elif isinstance(val, (list, tuple)):
                single[key] = [val[idx]]
            elif isinstance(val, np.ndarray):
                single[key] = val[idx : idx + 1]
            else:
                single[key] = val
        return single
