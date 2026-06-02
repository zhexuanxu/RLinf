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

import numpy as np
import sentencepiece

logger = logging.getLogger(__name__)

# The SentencePiece model bundled with this package (md5-identical to the
# upstream gs://big_vision/paligemma_tokenizer.model asset openpi downloads).
_DEFAULT_TOKENIZER_PATH = (
    pathlib.Path(__file__).resolve().parent / "assets" / "paligemma_tokenizer.model"
)


class PaligemmaTokenizer:
    """PaliGemma SentencePiece tokenizer with the pi05 discrete-state prompt."""

    def __init__(self, max_len: int = 48, model_path: pathlib.Path | str | None = None):
        self._max_len = max_len
        path = pathlib.Path(model_path) if model_path is not None else _DEFAULT_TOKENIZER_PATH
        if not path.exists():
            raise FileNotFoundError(f"PaliGemma tokenizer model not found at: {path}")
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Tokenize ``prompt`` (+ discretized ``state`` for pi05) to ids + mask."""
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            # Pi05 format: the state is part of the discrete language input.
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # Pi0 format: state goes to the continuous action expert input.
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode(
                "\n"
            )
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
