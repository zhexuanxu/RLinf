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

"""Pi0.5 VLM-only SFT dataset for Robo2VLM data.

Loads Robo2VLM parquet files and converts them to openpi Observation format
for VLM-only SFT training (CE loss on language tokens, no actions).

The dataset produces (observation_dict, None, meta_dict) tuples where:
- observation_dict follows openpi's Observation format
- actions is None (VLM-only mode)
- meta_dict carries raw text fields (only populated in eval_mode for
  per-sample accuracy and stdout debugging)
"""

import bisect
import glob
import io
import logging

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

ANSWER_LABELS = ["A", "B", "C", "D", "E", "F"]


class Pi05VLMDataset(Dataset):
    """Dataset for VLM-only SFT on pi0.5 with Robo2VLM-style MCQ data.

    Train mode: tokens = [prompt_tokens, answer_token, EOS, pad...] with
    token_loss_mask True for the answer+EOS region (CE-loss target).

    Eval mode:  tokens = [prompt_tokens, pad...] with token_loss_mask all
    False — used as a generation prompt by ``generate_language``. Raw text
    fields are returned in the third tuple element so the worker can
    compute string-match accuracy and pretty-print samples.

    Uses lazy loading: only stores file paths and row counts at init time,
    reads individual parquet files on demand in __getitem__ with a 1-file
    LRU cache.
    """

    def __init__(
        self,
        data_dir: str,
        max_token_len: int = 200,
        image_size: int = 224,
        num_images: int = 1,
        prompt_key: str = "question",
        choice_key: str = "choices",
        answer_key: str = "correct_answer",
        image_key: str = "image",
        eval_mode: bool = False,
        tokenizer_path: str = "gs://big_vision/paligemma_tokenizer.model",
    ):
        import openpi.shared.download as download
        import sentencepiece

        path = download.maybe_download(tokenizer_path, gs={"token": "anon"})
        with path.open("rb") as f:
            self.tokenizer = sentencepiece.SentencePieceProcessor(
                model_proto=f.read()
            )

        self.max_token_len = max_token_len
        self.image_size = image_size
        self.num_images = num_images
        self.prompt_key = prompt_key
        self.choice_key = choice_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.eval_mode = eval_mode
        self._eos_token_id = 1  # PaliGemma default

        self._parquet_files = sorted(glob.glob(f"{data_dir}/*.parquet"))
        if not self._parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        self._cumulative_lengths = []
        total = 0
        for f in self._parquet_files:
            metadata = pq.read_metadata(f)
            total += metadata.num_rows
            self._cumulative_lengths.append(total)
        self._total_len = total

        self._cached_file_idx = -1
        self._cached_df = None

        logger.info(
            "Indexed %d samples from %d parquet files in %s (eval_mode=%s)",
            self._total_len,
            len(self._parquet_files),
            data_dir,
            self.eval_mode,
        )

    def _get_row(self, idx):
        file_idx = bisect.bisect_right(self._cumulative_lengths, idx)
        row_idx = idx if file_idx == 0 else idx - self._cumulative_lengths[file_idx - 1]

        if self._cached_file_idx != file_idx:
            self._cached_df = pq.read_table(self._parquet_files[file_idx]).to_pandas()
            self._cached_file_idx = file_idx

        return self._cached_df.iloc[row_idx]

    def __len__(self):
        return self._total_len

    def _load_image(self, image_data):
        if isinstance(image_data, dict) and "bytes" in image_data:
            img = Image.open(io.BytesIO(image_data["bytes"])).convert("RGB")
        elif isinstance(image_data, bytes):
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else:
            img = Image.new("RGB", (self.image_size, self.image_size), (128, 128, 128))
        img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32) / 255.0 * 2.0 - 1.0

    def _build_observation(self, img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask):
        image_dict = {"image_0": img_array}
        image_mask_dict = {"image_0": True}
        for i in range(1, self.num_images):
            dummy = np.zeros_like(img_array)
            image_dict[f"image_{i}"] = dummy
            image_mask_dict[f"image_{i}"] = False

        return {
            "image": image_dict,
            "image_mask": image_mask_dict,
            "state": np.zeros(1, dtype=np.float32),
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "token_kv_cache_mask": kv_cache_mask,
        }

    def __getitem__(self, idx):
        row = self._get_row(idx)

        question = str(row[self.prompt_key])
        choices = str(row.get(self.choice_key, "")) if self.choice_key else ""
        correct_answer = int(row[self.answer_key])

        if choices:
            prompt_text = f"Task: {question} Choices: {choices}\nAnswer: "
        else:
            prompt_text = f"Task: {question}\nAnswer: "

        gold_letter = (
            ANSWER_LABELS[correct_answer]
            if 0 <= correct_answer < len(ANSWER_LABELS)
            else str(correct_answer)
        )

        img_array = self._load_image(row[self.image_key])

        prompt_tokens = self.tokenizer.encode(prompt_text, add_bos=True)
        max_len = self.max_token_len

        if self.eval_mode:
            # Prompt-only tokens; nothing to learn / predict here. The worker
            # calls model.generate_language() on this prompt at eval time.
            if len(prompt_tokens) > max_len:
                prompt_tokens = prompt_tokens[:max_len]
            prompt_len = len(prompt_tokens)

            tokens = np.zeros(max_len, dtype=np.int32)
            tokens[:prompt_len] = prompt_tokens

            token_mask = np.zeros(max_len, dtype=bool)
            token_mask[:prompt_len] = True

            ar_mask = np.zeros(max_len, dtype=np.int32)  # bidirectional prefix
            loss_mask = np.zeros(max_len, dtype=bool)    # no loss in eval
            kv_cache_mask = token_mask.copy()

            observation = self._build_observation(
                img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask
            )
            meta = {
                "question": question,
                "choices": choices,
                "correct_answer_letter": gold_letter,
                "prompt_text": prompt_text,
            }
            return observation, None, meta

        # Train mode (existing behavior)
        answer_tokens = self.tokenizer.encode(gold_letter, add_bos=False)

        prompt_len = len(prompt_tokens)
        answer_len = len(answer_tokens)

        if prompt_len > max_len - 2:
            prompt_tokens = prompt_tokens[:max_len - 2]
            prompt_len = len(prompt_tokens)

        available = max_len - prompt_len - 1  # -1 for EOS
        if available < answer_len:
            answer_tokens = answer_tokens[:available]
            answer_len = len(answer_tokens)

        total_len = prompt_len + answer_len + 1  # +1 for EOS

        tokens = np.zeros(max_len, dtype=np.int32)
        tokens[:prompt_len] = prompt_tokens
        tokens[prompt_len : prompt_len + answer_len] = answer_tokens
        tokens[prompt_len + answer_len] = self._eos_token_id

        token_mask = np.zeros(max_len, dtype=bool)
        token_mask[:total_len] = True

        ar_mask = np.zeros(max_len, dtype=np.int32)
        ar_mask[prompt_len:total_len] = 1

        loss_mask = np.zeros(max_len, dtype=bool)
        loss_mask[prompt_len:total_len] = True

        kv_cache_mask = token_mask.copy()
        kv_cache_mask[prompt_len + answer_len] = False  # EOS excluded

        observation = self._build_observation(
            img_array, tokens, token_mask, ar_mask, loss_mask, kv_cache_mask
        )
        return observation, None, {}


def pi05_vlm_collate_fn(batch):
    """Collate function for Pi05VLMDataset.

    Returns a 3-tuple ``(batched_obs, None, meta_list)``. ``meta_list`` is a
    Python list of per-sample dicts (empty in train mode, populated with raw
    text in eval mode) — not stacked, since fields are strings.
    """
    observations, _, metas = zip(*batch)

    batched_obs = {"image": {}, "image_mask": {}}
    first = observations[0]

    for key in first["image"]:
        batched_obs["image"][key] = np.stack(
            [obs["image"][key] for obs in observations]
        )
        batched_obs["image_mask"][key] = np.array(
            [obs["image_mask"][key] for obs in observations]
        )

    for key in (
        "state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "token_ar_mask",
        "token_loss_mask",
        "token_kv_cache_mask",
    ):
        batched_obs[key] = np.stack([obs[key] for obs in observations])

    return batched_obs, None, list(metas)
