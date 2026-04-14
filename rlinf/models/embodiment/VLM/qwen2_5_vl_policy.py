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

"""Qwen2.5-VL policy wrapper for use as a VLM in the dual-system embodied agentloop.

Uses ``qwen_vl_utils.process_vision_info`` for proper image handling following
the official Qwen2.5-VL inference pattern.

Primary interface: ``generate_subtask(obs, prompt, **kwargs) -> list[str]``.
"""

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from rlinf.models.embodiment.base_policy import BasePolicy


class Qwen2_5_VLPolicy(nn.Module, BasePolicy):
    """Qwen2.5-VL wrapper that generates language subtasks from visual observations.

    The prompt is **not** stored in the model config — it is passed at call time
    via the ``prompt`` parameter of :meth:`generate_subtask`, allowing the
    agentloop to swap prompts (e.g. memory-aware vs. memoryless) dynamically.
    """

    def __init__(self, cfg: DictConfig, torch_dtype=None):
        nn.Module.__init__(self)

        self.model_path = cfg.model_path
        self.min_pixels = int(cfg.get("min_pixels", 256 * 28 * 28))
        self.max_pixels = int(cfg.get("max_pixels", 1280 * 28 * 28))

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )
        self.processor.tokenizer.padding_side = "left"
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

        Args:
            obs: Environment observation dict.  Expected keys:
                ``"main_images"`` — ``torch.Tensor [B, H, W, C]`` uint8 or float,
                ``"task_descriptions"`` — ``list[str]`` of length B.
            prompt: A pre-formatted prompt string shared across the batch,
                or a ``list[str]`` of length B with per-sample prompts.
                If empty (or an empty string at index *i*), a minimal
                default is used for that sample.
            **generate_kwargs: Forwarded to ``model.generate()``
                (``temperature``, ``top_p``, ``top_k``, ``max_new_tokens``,
                ``repetition_penalty``, ``do_sample``, etc.).

        Returns:
            A list of generated strings, one per environment (length B).
        """
        main_images: Optional[torch.Tensor] = obs.get("main_images")
        task_descriptions: Optional[list] = obs.get("task_descriptions")
        batch_size = self._infer_batch_size(obs)

        # Build per-sample messages in the Qwen2.5-VL chat format.
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

        # Apply chat template per sample.
        texts = [
            self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            for msgs in messages_batch
        ]

        # Use qwen_vl_utils to extract vision inputs from the messages.
        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Generate with caller-supplied sampling params.
        generated_ids = self.model.generate(**inputs, **generate_kwargs)

        # Trim input tokens per sample (handles variable-length padding).
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return outputs

    # ------------------------------------------------------------------
    # BasePolicy abstract method stubs
    # ------------------------------------------------------------------

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "Qwen2_5_VLPolicy is a text-generation VLM and does not support "
            "default_forward. Use generate_subtask() instead."
        )

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError(
            "Qwen2_5_VLPolicy is a text-generation VLM and does not produce "
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
