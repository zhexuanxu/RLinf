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

"""Qwen3-VL-Thinking policy wrapper for the dual-system embodied agentloop.

Uses ``Qwen3VLForConditionalGeneration`` with thinking mode **always enabled**.
The chat template unconditionally inserts ``<think>`` at the start of the
assistant turn; after generation, the thinking content (between ``<think>``
and ``</think>``) is stripped so that only the final response is returned.

Primary interface: ``generate_subtask(obs, prompt, **kwargs) -> list[str]``.
"""

import re
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from rlinf.models.embodiment.base_policy import BasePolicy

# Token IDs for the thinking delimiters in Qwen3-VL.
_THINK_END_TOKEN_ID = 151668  # </think>


class Qwen3_VLPolicy(nn.Module, BasePolicy):
    """Qwen3-VL-Thinking wrapper with thinking mode enabled.

    Thinking mode is **unconditionally** enabled by the Qwen3-VL-Thinking chat
    template — every generation starts with ``<think>`` and the model produces
    a chain-of-thought reasoning block before the final response.

    The thinking content is automatically stripped from the returned strings
    so the agentloop only sees the actionable subtask / JSON output.
    """

    def __init__(self, cfg: DictConfig, torch_dtype=None):
        nn.Module.__init__(self)

        self.model_path = cfg.model_path
        self.min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
        self.max_pixels = int(cfg.get("max_pixels", 1280 * 28 * 28))

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.model.eval()

    # ------------------------------------------------------------------
    # Primary interface for the dual-system agentloop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_subtask(
        self,
        obs: dict[str, Any],
        prompt: str | list[str] = "",
        **generate_kwargs,
    ) -> list[str]:
        """Generate a text response for each environment in the batch.

        Thinking content (``<think>...</think>``) is stripped from the output.

        Args:
            obs: Environment observation dict.
            prompt: Pre-formatted prompt string shared across the batch,
                or a ``list[str]`` of length B with per-sample prompts.
            **generate_kwargs: Forwarded to ``model.generate()``.

        Returns:
            List of generated strings (thinking stripped), length B.
        """
        main_images: Optional[torch.Tensor] = obs.get("main_images")
        task_descriptions: Optional[list] = obs.get("task_descriptions")
        batch_size = self._infer_batch_size(obs)

        # Build per-sample messages.
        messages_batch: list[list[dict]] = []

        for i in range(batch_size):
            prompt_text = prompt[i] if isinstance(prompt, list) else prompt
            if not prompt_text:
                task_desc = ""
                if task_descriptions is not None and i < len(task_descriptions):
                    task_desc = str(task_descriptions[i])
                prompt_text = (
                    f"Task: {task_desc}. Based on the image, describe the "
                    "immediate next subtask for the robot arm in one sentence."
                )

            content: list[dict] = []
            if main_images is not None:
                pil_img = self._tensor_to_pil(main_images[i])
                content.append({"type": "image", "image": pil_img})
            content.append({"type": "text", "text": prompt_text})

            messages_batch.append([{"role": "user", "content": content}])

        # Two-step pattern (matches official Qwen3-VL batch inference docs):
        # 1. apply_chat_template per sample to get text with <think> prompt.
        # 2. process_vision_info to extract images, then processor to tokenize.
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, **generate_kwargs)

        # Trim input tokens per sample and strip thinking content.
        outputs = []
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids):
            output_ids = out_ids[len(in_ids):].tolist()
            content_text = self._strip_thinking(output_ids)
            outputs.append(content_text)

        return outputs

    # ------------------------------------------------------------------
    # Thinking token handling
    # ------------------------------------------------------------------

    def _strip_thinking(self, output_ids: list[int]) -> str:
        """Strip the ``<think>...</think>`` block from generated token IDs.

        Uses token-ID-level splitting: finds the last ``</think>`` token and
        only decodes tokens after it.  Falls back to regex if the token is not
        found (e.g. if generation was truncated before ``</think>``).
        """
        try:
            # Find the last </think> token.
            idx = len(output_ids) - 1 - output_ids[::-1].index(_THINK_END_TOKEN_ID)
            content_ids = output_ids[idx + 1:]
        except ValueError:
            # </think> not found — decode everything and try regex stripping.
            content_ids = output_ids

        text = self.processor.decode(
            content_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ).strip()

        # Fallback regex: remove any remaining <think>...</think> in text.
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    # ------------------------------------------------------------------
    # BasePolicy abstract method stubs
    # ------------------------------------------------------------------

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "Qwen3_VLPolicy is a text-generation VLM and does not support "
            "default_forward. Use generate_subtask() instead."
        )

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError(
            "Qwen3_VLPolicy is a text-generation VLM and does not produce "
            "action tensors. Use generate_subtask() instead."
        )

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
    def _tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
        """Convert a [H, W, C] tensor (uint8 or float in [0,1]) to PIL Image."""
        if img_tensor.dtype == torch.uint8:
            arr = img_tensor.cpu().numpy()
        else:
            arr = img_tensor.cpu().float().numpy()
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)
