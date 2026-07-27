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

"""PaliGemma tokenization for OpenPI and π₀.₅ subtask supervision."""

from __future__ import annotations

import dataclasses
import logging
import string
from typing import Any

import numpy as np
import openpi.shared.download as download
import sentencepiece


class PaligemmaTokenizer:
    """OpenPI's PaliGemma tokenizer with optional subtask supervision."""

    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with path.open("rb") as file:
            self._tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=file.read()
            )

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize an OpenPI prompt using the official PaliGemma format."""
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # This is the Pi05 format, where the state is part of the discrete
            # language input.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous
            # action expert input. Tokenize "\n" separately as the
            # "start of answer" token.
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating. "
                    "Consider increasing the `max_token_len` in your model config "
                    "if this happens frequently.",
                    len(tokens),
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    @property
    def eos_token_id(self) -> int:
        """Return the EOS id used to terminate generated subtasks."""
        eos_id = int(self._tokenizer.eos_id())
        if eos_id < 0:
            raise ValueError(
                "The PaliGemma SentencePiece model has no EOS token, but "
                "vlm_vla supervision and generation require one."
            )
        return eos_id

    @property
    def max_len(self) -> int:
        """Return the fixed prompt/response token budget."""
        return self._max_len

    def decode(self, token_ids: Any) -> str:
        """Decode generated token IDs."""
        return self._tokenizer.decode([int(token) for token in token_ids])

    @staticmethod
    def _clean_subtask_text(text: str) -> str:
        cleaned = text.lower().strip().replace("_", " ").replace("\n", " ")
        if cleaned and cleaned[-1] in string.punctuation:
            cleaned = cleaned[:-1]
        return cleaned

    def tokenize_with_subtask(
        self,
        prompt: str,
        state: np.ndarray | None,
        response: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a main task and optional autoregressive subtask target.

        Prefix tokens use bidirectional attention and are visible to the action
        expert. Response tokens are causal and supervised. The supervised EOS
        token terminates generation but is excluded from the action expert's
        KV cache.
        """
        task_text = self._clean_subtask_text(prompt)
        if state is None:
            prefix = f"Task: {task_text}. Subtask: "
        else:
            bins = np.linspace(-1, 1, 256 + 1)[:-1]
            discrete_state = np.digitize(np.asarray(state), bins=bins) - 1
            state_text = " ".join(map(str, discrete_state))
            prefix = f"Task: {task_text}. State: {state_text};\nSubtask: "

        tokens = list(self._tokenizer.encode(prefix, add_bos=True))
        ar_mask = [False] * len(tokens)
        loss_mask = [False] * len(tokens)
        kv_cache_mask = [True] * len(tokens)

        if response is not None:
            response_text = self._clean_subtask_text(response)
            response_tokens = list(self._tokenizer.encode(f"{response_text}."))
            tokens.extend(response_tokens)
            ar_mask.extend([True] * len(response_tokens))
            loss_mask.extend([True] * len(response_tokens))
            kv_cache_mask.extend([True] * len(response_tokens))

            tokens.append(self.eos_token_id)
            ar_mask.append(True)
            loss_mask.append(True)
            kv_cache_mask.append(False)

        if len(tokens) > self._max_len:
            raise ValueError(
                f"Subtask sequence uses {len(tokens)} tokens, exceeding "
                f"max_token_len={self._max_len}. Increase max_token_len; "
                "truncation would silently remove response/EOS supervision."
            )

        pad_len = self._max_len - len(tokens)
        input_mask = [True] * len(tokens) + [False] * pad_len
        tokens.extend([0] * pad_len)
        ar_mask.extend([False] * pad_len)
        loss_mask.extend([False] * pad_len)
        kv_cache_mask.extend([False] * pad_len)

        return (
            np.asarray(tokens),
            np.asarray(input_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
            np.asarray(kv_cache_mask),
        )


@dataclasses.dataclass(frozen=True)
class TokenizeSubtaskPrompt:
    """OpenPI transform that consumes ``prompt`` and optional ``response``."""

    tokenizer: PaligemmaTokenizer
    inject_state: bool

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data = dict(data)
        prompt = data.pop("prompt", None)
        if prompt is None:
            raise ValueError("Prompt is required for vlm_vla tokenization.")
        if not isinstance(prompt, str):
            prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)

        response = data.pop("response", None)
        if response is not None and not isinstance(response, str):
            response = response.item() if hasattr(response, "item") else str(response)

        state = data.get("state") if self.inject_state else None
        tokens, input_mask, ar_mask, loss_mask, kv_cache_mask = (
            self.tokenizer.tokenize_with_subtask(
                prompt,
                np.asarray(state) if state is not None else None,
                response,
            )
        )
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": input_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "token_kv_cache_mask": kv_cache_mask,
        }
