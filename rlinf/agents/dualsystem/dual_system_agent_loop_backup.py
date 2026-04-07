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

from rlinf.agents.dualsystem.prompts import (
    DEFAULT_VLM_PROMPT,
    MEMORY_VLM_PROMPT,
    parse_subtask_and_memory,
    parse_subtask_only,
)

logger = logging.getLogger(__name__)


class DualSystemAgentLoop:
    """Coordinates VLM and VLA for one agentloop step.

    Args:
        vlm_model: VLM exposing ``generate_subtask(obs, prompt, **kwargs)``.
        vla_model: VLA exposing ``predict_action_batch(env_obs, **kwargs)``.
        log_subtasks: Log the first subtask at each step.
        enable_memory: Enable the MEM-style language memory module.
        vlm_sampling_params: Generation kwargs forwarded to the VLM
            (``temperature``, ``top_p``, ``top_k``, ``max_new_tokens``, etc.).
    """

    def __init__(
        self,
        vlm_model,
        vla_model,
        log_subtasks: bool = False,
        enable_memory: bool = False,
        vlm_sampling_params: dict | None = None,
    ):
        self.vlm_model = vlm_model
        self.vla_model = vla_model
        self.log_subtasks = log_subtasks
        self.enable_memory = enable_memory
        self.vlm_sampling_params = vlm_sampling_params or {}

        # Per-environment memory strings.  Initialised lazily on the first
        # call to run_step (when batch size is known) or via reset_memory().
        self._memories: list[str] | None = None

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def reset_memory(self, batch_size: int | None = None):
        """Reset the memory state for all environments.

        Call this between evaluation epochs / episodes so that memory from
        a previous episode does not leak into the next one.
        """
        if batch_size is not None:
            self._memories = ["" for _ in range(batch_size)]
        elif self._memories is not None:
            self._memories = ["" for _ in range(len(self._memories))]
        else:
            self._memories = None

    def reset_memory_for_envs(self, done_mask: torch.Tensor):
        """Selectively reset memory for environments that have terminated.

        This is called on each eval step when the env auto-resets done
        episodes, so that memory from a finished episode does not bleed
        into the new episode that starts in the same batch slot.

        Args:
            done_mask: Boolean tensor of shape ``[B]`` (or ``[B, C]`` where
                the last column ``[:, -1]`` indicates the final done state).
                ``True`` entries have their memory cleared.
        """
        if self._memories is None:
            return
        # Handle [B, C] dones — use the last chunk-step column.
        if done_mask.dim() > 1:
            done_mask = done_mask[:, -1]
        for i in range(min(len(self._memories), done_mask.shape[0])):
            if done_mask[i].item():
                self._memories[i] = ""

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
        batch_size = self._infer_batch_size(obs)
        task_descriptions = obs.get("task_descriptions") or [""] * batch_size

        # ----- Build per-sample prompts ----- #
        input_memories: list[str] | None = None
        if self.enable_memory:
            # Lazy-init memory.
            if self._memories is None or len(self._memories) != batch_size:
                self._memories = ["" for _ in range(batch_size)]
            # Snapshot AFTER init but BEFORE VLM call (for logging).
            input_memories = list(self._memories)

            prompts = [
                MEMORY_VLM_PROMPT.format(
                    task_description=task_descriptions[i]
                    if i < len(task_descriptions)
                    else "",
                    memory=self._memories[i] or "(no memory yet)",
                )
                for i in range(batch_size)
            ]
        else:
            prompts = [
                DEFAULT_VLM_PROMPT.format(
                    task_description=task_descriptions[i]
                    if i < len(task_descriptions)
                    else "",
                )
                for i in range(batch_size)
            ]

        # ----- Turn 1: VLM generates subtask (+ memory) ----- #
        # Call generate_subtask once per sample (prompt differs per sample
        # because task_description / memory differ).  For batch efficiency
        # we pass the first prompt when all prompts are identical, otherwise
        # loop per sample.
        all_same_prompt = len(set(prompts)) == 1
        if all_same_prompt:
            raw_outputs = self.vlm_model.generate_subtask(
                obs, prompt=prompts[0], **self.vlm_sampling_params
            )
        else:
            # Per-sample generation (prompts differ across batch).
            raw_outputs = []
            for i in range(batch_size):
                single_obs = self._slice_obs(obs, i)
                out = self.vlm_model.generate_subtask(
                    single_obs, prompt=prompts[i], **self.vlm_sampling_params
                )
                raw_outputs.append(out[0])

        # ----- Parse outputs ----- #
        subtasks: list[str] = []
        if self.enable_memory:
            new_memories: list[str] = []
            for i, raw in enumerate(raw_outputs):
                sub, mem = parse_subtask_and_memory(raw)
                subtasks.append(sub)
                # If parsing failed to produce a new memory, keep the old one.
                new_memories.append(mem if mem else self._memories[i])
            # Update memory state — strictly aligned per batch index.
            self._memories = new_memories
        else:
            for raw in raw_outputs:
                sub, _ = parse_subtask_only(raw)
                subtasks.append(sub)

        if self.log_subtasks and subtasks:
            logger.info("[DualSystem] VLM subtask: %s", subtasks[0])
            if self.enable_memory and self._memories:
                logger.info("[DualSystem] Memory: %s", self._memories[0][:200])

        # ----- Turn 2: VLA generates actions ----- #
        obs_with_subtask = dict(obs)
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

        # VLM inputs: what the VLM saw this turn.
        vla_result["vlm_inputs"] = {
            "task_descriptions": list(task_descriptions),
            "prompts": prompts,
            "input_memories": input_memories,  # None when memory disabled
        }

        # VLM outputs: what the VLM produced this turn.
        vla_result["vlm_outputs"] = {
            "raw_outputs": list(raw_outputs),
            "subtasks": list(subtasks),
            "output_memories": list(self._memories) if self.enable_memory else None,
        }

        if self.enable_memory:
            vla_result["memories"] = list(self._memories)

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
