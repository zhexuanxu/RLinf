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

"""Self-contained PaliGemma tokenizer for the BEHAVIOR pi05 eval path.

Vendored re-implementation of ``openpi.models.tokenizer.PaligemmaTokenizer``,
loading a bundled copy of the SentencePiece model so the package does not depend
on the installed ``openpi`` distribution or its network download machinery. The
``tokenize`` logic is byte-identical to upstream (verified by a cross-check test
against the installed ``openpi`` tokenizer).
"""

from __future__ import annotations

import logging
import pathlib
import string

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)


class PaligemmaTokenizer:
    """PaliGemma SentencePiece tokenizer with the pi05 discrete-state prompt.

    The SentencePiece model lives OUTSIDE the code repository (model files do not
    belong in source control), so ``path`` is required and supplied from YAML
    (``actor.model.openpi.paligemma_tokenizer``); there is no hard-coded fallback.
    """

    def __init__(self, path: pathlib.Path | str, max_len: int = 48):
        self._max_len = max_len
        tokenizer_path = pathlib.Path(path)
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"PaliGemma tokenizer model not found at: {tokenizer_path}"
            )
        with tokenizer_path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize ``prompt`` (+ discretized ``state`` for pi05) to ids + mask."""
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # Pi05 format: the state is part of the discrete language input.
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format: state goes to the continuous action expert input.
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logger.warning(
                    "Token length (%d) exceeds max length (%d), truncating.",
                    tokens_len,
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)

    @property
    def eos_token_id(self) -> int:
        """The SentencePiece EOS id (terminates generated subtask text)."""
        eos_id = self._tokenizer.eos_id()
        if eos_id < 0:
            raise ValueError(
                "The PaliGemma SentencePiece model defines no EOS id; subtask "
                "supervision and generation require one."
            )
        return eos_id

    def decode(self, token_ids) -> str:
        """Decode token ids back to text (for inspection of generated subtasks)."""
        return self._tokenizer.decode([int(t) for t in token_ids])

    @staticmethod
    def _clean_subtask_text(text: str) -> str:
        """Reference-style cleanup: lowercase, normalize, strip end punctuation."""
        cleaned = text.lower().strip().replace("_", " ").replace("\n", " ")
        if cleaned and cleaned[-1] in string.punctuation:
            cleaned = cleaned[:-1]
        return cleaned

    def tokenize_with_subtask(
        self,
        prompt: str,
        state: np.ndarray,
        response: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize the subtask-supervision template with per-token masks.

        Template: ``Task: {prompt}. State: {state}. Subtask: [{response}.][EOS]``
        (the reference pi05 VLM-VLA format). The three returned per-token masks
        encode the attention/loss/KV-cache contract:

        * prefix (everything through ``Subtask: ``): bidirectional
          (``ar=False``), no CE loss, visible to the action expert (``kv=True``);
        * response (``{response}.``, training only): causal, CE loss, ``kv=True``;
        * EOS (training only): causal, CE loss, ``kv=False`` — the action expert
          never attends to it;
        * right padding: invalid, ``ar=False``, no loss, ``kv=False``.

        Args:
            prompt: The main-task text (the model input at every level).
            state: The normalized (pre-padding) state vector, discretized into
                the prompt exactly like the action-only format.
            response: The subtask label to supervise (SFT); ``None`` emits the
                generation prefix only (eval).

        Returns:
            ``(tokens, input_mask, ar_mask, loss_mask, kv_cache_mask)`` numpy
            arrays of length ``max_len`` (tokens ``int``, masks ``bool``).

        Raises:
            ValueError: If the assembled sequence exceeds ``max_len`` —
                truncation would silently drop response/EOS supervision.
        """
        task_text = self._clean_subtask_text(prompt)
        discretized_state = (
            np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        )
        state_str = " ".join(map(str, discretized_state))
        prefix_text = f"Task: {task_text}. State: {state_str}. Subtask: "

        tokens = self._tokenizer.encode(prefix_text, add_bos=True)
        ar_mask = [False] * len(tokens)
        loss_mask = [False] * len(tokens)
        kv_cache_mask = [True] * len(tokens)

        if response is not None:
            response_text = self._clean_subtask_text(response)
            response_tokens = self._tokenizer.encode(f"{response_text}.")
            tokens += response_tokens
            ar_mask += [True] * len(response_tokens)
            loss_mask += [True] * len(response_tokens)
            kv_cache_mask += [True] * len(response_tokens)

            tokens += [self.eos_token_id]
            ar_mask += [True]
            loss_mask += [True]
            kv_cache_mask += [False]

        tokens_len = len(tokens)
        if tokens_len > self._max_len:
            raise ValueError(
                f"Subtask-supervision sequence is {tokens_len} tokens but "
                f"max_token_len is {self._max_len}; truncating would drop "
                "response/EOS supervision. Increase max_token_len."
            )
        pad_len = self._max_len - tokens_len
        input_mask = [True] * tokens_len + [False] * pad_len
        tokens = tokens + [0] * pad_len
        ar_mask += [False] * pad_len
        loss_mask += [False] * pad_len
        kv_cache_mask += [False] * pad_len

        return (
            np.asarray(tokens),
            np.asarray(input_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
            np.asarray(kv_cache_mask),
        )
