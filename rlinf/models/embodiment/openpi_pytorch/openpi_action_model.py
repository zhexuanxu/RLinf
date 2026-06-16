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

"""Entry point for the self-contained PyTorch OpenPI 0.5 BEHAVIOR model.

This model focuses on exactly two responsibilities: eval action sampling and
SFT flow-matching loss.
Its high-level interface mirrors the old ``OpenPi0ForRLActionPrediction`` so the eval
rollout worker can call it unchanged via the OpenPI dispatch path:

    actions, result = model.predict_action_batch(env_obs=env_obs, mode="eval")

``actions`` has shape ``[B, action_chunk, action_env_dim]`` (e.g. ``[B, 32, 23]``).
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn

from rlinf.data.datasets.openpi_pytorch.eval_processor import EvalProcessor
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    normalize_quantile,
)

logger = logging.getLogger(__name__)

# The first few eval batches log their full generated subtask text (to eyeball
# the reasoning-then-acting flow); after that, a compact structured summary
# (EOS rate, token-length stats, and the most common generated strings) is
# logged every _GENERATION_SUMMARY_INTERVAL batches instead of flooding the log
# with per-batch text. All summary bookkeeping is best-effort and never affects
# the returned actions.
_GENERATION_LOG_BATCHES = 3
_GENERATION_SUMMARY_INTERVAL = 20
_INT_MAX = 2**63 - 1


class OpenPiPytorchActionModel(nn.Module):
    """Wrap the vendored ``Pi0`` model with BEHAVIOR eval pre/post-processing."""

    def __init__(
        self,
        pi0_model: Pi0,
        processor: EvalProcessor | None,
        *,
        num_steps: int,
        action_chunk: int,
        action_env_dim: int,
    ):
        super().__init__()
        self.model = pi0_model
        self.processor = processor
        self.num_steps = num_steps
        self.action_chunk = action_chunk
        self.action_env_dim = action_env_dim
        self._generation_batches_logged = 0

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions for a batch of observations (eval / action generation).

        When ``noise`` (shape ``[B, action_horizon, model_action_dim]``) and/or a
        ``rng`` generator are provided, they are passed verbatim to the
        flow-matching sampler so the same observation yields the same actions —
        used for deterministic / paired evaluation. The default (both ``None``)
        preserves the stochastic production sampling unchanged.
        """
        if self.processor is None:
            raise RuntimeError(
                "predict_action_batch requires an eval processor; "
                "the current model was built for SFT training only."
            )
        observation = self.processor.build_observation(env_obs, self.device)
        generation = None
        if getattr(self.model, "vlm_vla", False):
            # Reasoning first: the VLM generates the subtask text, then the
            # action expert denoises against the generation's KV cache.
            model_actions, generation = self.model.reason_and_sample_actions(
                observation,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                num_steps=self.num_steps,
                noise=noise,
                rng=rng,
            )
        else:
            model_actions = self.model.sample_actions(
                observation, num_steps=self.num_steps, noise=noise, rng=rng
            )
        actions = self.processor.postprocess_actions(model_actions).to(self.device)

        batch = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(batch, -1).contiguous(),
                "model_action": model_actions.reshape(batch, -1).contiguous(),
            },
        }
        if generation is not None:
            tokens = generation["tokens"]
            eos_steps = generation["eos_steps"]
            texts = []
            for row in range(batch):
                row_ids = tokens[row, : int(eos_steps[row].item())]
                texts.append(self.processor.tokenizer.decode(row_ids.tolist()))
            result["generated_text"] = texts
            result["generation_terminated"] = generation["terminated"].cpu()
            result["forward_inputs"]["generated_token_ids"] = tokens.contiguous()
            # Best-effort logging only: wrapped so a logging/stat bug can never
            # disturb the returned actions or the eval metric.
            try:
                self._log_generation(texts, eos_steps, generation["terminated"])
            except Exception:  # noqa: BLE001 - diagnostics must never break eval
                logger.debug("vlm_vla generation logging skipped", exc_info=True)
        return actions, result

    def _log_generation(self, texts, eos_steps, terminated) -> None:
        """Accumulate generation stats and log verbose text then summaries.

        The first few batches log full generated text; afterwards a compact
        summary (EOS rate, token-length min/mean/max, top generated strings) is
        logged periodically so the eval log shows whether the VLM produces
        varied, terminating subtasks without per-batch flooding.
        """
        if not hasattr(self, "_generation_stats"):
            self._generation_stats = {
                "rows": 0, "eos": 0, "len_sum": 0,
                "len_min": _INT_MAX, "len_max": 0, "texts": {},
            }
        if not hasattr(self, "_generation_batches_logged"):
            self._generation_batches_logged = 0
        stats = self._generation_stats
        for row, text in enumerate(texts):
            stats["rows"] += 1
            if bool(terminated[row]):
                stats["eos"] += 1
            length = int(eos_steps[row].item())
            stats["len_sum"] += length
            stats["len_min"] = min(stats["len_min"], length)
            stats["len_max"] = max(stats["len_max"], length)
            stats["texts"][text] = stats["texts"].get(text, 0) + 1

        n = self._generation_batches_logged
        self._generation_batches_logged += 1
        if n < _GENERATION_LOG_BATCHES:
            logger.info(
                "vlm_vla subtask generation before denoise (batch of %d): %s",
                len(texts),
                [
                    f"{text[:60]!r}{' [EOS]' if bool(terminated[row]) else ' [no EOS]'}"
                    for row, text in enumerate(texts)
                ],
            )
        elif (n + 1) % _GENERATION_SUMMARY_INTERVAL == 0:
            rows = max(stats["rows"], 1)
            top = sorted(stats["texts"].items(), key=lambda kv: -kv[1])[:5]
            logger.info(
                "vlm_vla generation summary over %d samples: eos_rate=%.3f, "
                "token_len[min/mean/max]=%d/%.1f/%d, %d unique strings; top: %s",
                stats["rows"],
                stats["eos"] / rows,
                stats["len_min"] if stats["len_min"] != _INT_MAX else 0,
                stats["len_sum"] / rows,
                stats["len_max"],
                len(stats["texts"]),
                [f"{t[:40]!r}x{c}" for t, c in top],
            )

    # --- SFT training (BEHAVIOR supervised fine-tuning) ---
    def forward(self, forward_type: ForwardType = ForwardType.SFT, **kwargs):
        """Dispatch a training forward pass.

        Eval/action-generation goes through :meth:`predict_action_batch`; the
        SFT runner calls this with ``forward_type=ForwardType.SFT, data=batch``.
        """
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            "OpenPiPytorchActionModel supports eval (predict_action_batch) and "
            f"SFT (ForwardType.SFT); got forward_type={forward_type!r}."
        )

    def sft_forward(self, data: Any) -> torch.Tensor | dict[str, torch.Tensor]:
        """Compute the SFT loss for one batch.

        ``data`` is either a ``(observation, actions)`` tuple or a dict with
        ``observation`` and ``actions`` (the dataloader already normalizes and
        pads actions to the model action dim). In action-only mode this returns
        the scalar mean of the ``(B, action_horizon)`` per-timestep
        flow-matching loss; with VLM token output the model returns a dict
        whose ``loss`` combines the flow and language CE terms (plus detached
        per-component metrics), passed through to the SFT worker unchanged.
        """
        observation, actions = self._unpack_sft_batch(data)
        observation = self._observation_to_device(observation)
        actions = self._actions_to_device(actions)
        output = self.model.compute_loss(observation, actions, train=True)
        if isinstance(output, dict):
            return output
        return output.mean()

    def compute_loss(self, data: Any) -> torch.Tensor:
        """Alias kept for interface parity with the old action model."""
        return self.sft_forward(data)

    @staticmethod
    def _unpack_sft_batch(data: Any) -> tuple[Any, Any]:
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    "SFT batch tuple must be (observation, actions); "
                    f"got length {len(data)}."
                )
            observation, actions = data
        elif isinstance(data, dict):
            if "observation" not in data or "actions" not in data:
                raise ValueError(
                    "SFT batch dict must contain 'observation' and 'actions'; "
                    f"got keys {sorted(data)}."
                )
            observation, actions = data["observation"], data["actions"]
        else:
            raise TypeError(f"Unsupported SFT batch type: {type(data)!r}.")
        if observation is None or actions is None:
            raise ValueError("SFT batch is missing observation or actions.")
        return observation, actions

    def _observation_to_device(self, observation: Any) -> Observation:
        if isinstance(observation, dict):
            observation = Observation.from_dict(observation)
        if not isinstance(observation, Observation):
            raise TypeError(
                f"SFT observation must be an Observation or dict; "
                f"got {type(observation)!r}."
            )
        device = self.device

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        return Observation(
            images={k: _move(v) for k, v in observation.images.items()},
            image_masks={k: _move(v) for k, v in observation.image_masks.items()},
            state=_move(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            token_kv_cache_mask=_move(observation.token_kv_cache_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )

    def _actions_to_device(self, actions: Any) -> torch.Tensor:
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        model_action_dim = self.model.action_dim
        if actions.dim() != 3:
            raise ValueError(
                "SFT actions must have shape [B, action_horizon, D]; "
                f"got {tuple(actions.shape)}."
            )
        if actions.shape[-1] == model_action_dim:
            return actions.to(device=self.device, dtype=torch.float32)
        if actions.shape[-1] == self.action_env_dim:
            return self._normalize_and_pad_env_actions(actions)
        raise ValueError(
            "SFT actions must have shape [B, action_horizon, "
            f"{model_action_dim}] (normalized + padded) or [B, action_horizon, "
            f"{self.action_env_dim}] (raw env actions); got {tuple(actions.shape)}."
        )

    def _normalize_and_pad_env_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if self.processor is None:
            raise ValueError(
                "SFT actions were provided at env action dim, but this model "
                "has no action normalization stats. Supply normalized/padded "
                "actions from the dataloader or build the model with a processor."
            )
        actions_np = actions.detach().cpu().numpy()
        normalized = normalize_quantile(
            actions_np.astype(np.float32), self.processor.action_stats
        )
        pad_width = [(0, 0)] * normalized.ndim
        pad_width[-1] = (0, self.model.action_dim - normalized.shape[-1])
        padded = np.pad(normalized, pad_width, constant_values=0.0)
        return torch.as_tensor(padded, device=self.device, dtype=torch.float32)

    # --- Gradient checkpointing pass-through (used by the FSDP training path) ---
    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_disable()
