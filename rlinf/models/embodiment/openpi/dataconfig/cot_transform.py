"""CoT (Chain-of-Thought) data transform for full pi0.5 SFT training.

Generates the three masks needed for full pi0.5:
- token_ar_mask: 0=bidirectional (prompt), 1=causal (CoT response)
- token_loss_mask: True for CoT response tokens (CE loss target)
- token_kv_cache_mask: True for tokens in action expert KV cache (False for EOS)

Usage:
    Insert this transform AFTER TokenizePrompt in the data pipeline.
    It reads `cot_text` from the data dict, tokenizes it, appends it to
    `tokenized_prompt`, and generates the three masks.

Modes:
    - VLA-only: no `cot_text` in data → masks default to all-zero/all-True
    - VLM-only: `cot_text` present, no `actions` → CE loss only
    - VLM+VLA: `cot_text` present + `actions` → CE + flow matching
"""

import dataclasses

import numpy as np
import openpi.shared.download as download
import sentencepiece


@dataclasses.dataclass(frozen=True)
class AppendCoTTokens:
    """Append CoT response tokens to tokenized_prompt and generate masks.

    This transform must be applied AFTER TokenizePrompt, which produces
    `tokenized_prompt` and `tokenized_prompt_mask` from the prompt text.

    Fields in data dict:
        Input:
            tokenized_prompt: [max_len] int array (from TokenizePrompt)
            tokenized_prompt_mask: [max_len] bool array
            cot_text: Optional[str] — CoT response text to append

        Output (added/modified):
            tokenized_prompt: [max_len] with CoT tokens appended
            tokenized_prompt_mask: [max_len] updated
            token_ar_mask: [max_len] int (0=bidirectional, 1=causal)
            token_loss_mask: [max_len] bool (True for CE loss tokens)
            token_kv_cache_mask: [max_len] bool (True if in action expert KV cache)
    """

    max_token_len: int = 200
    eos_token_id: int = 1

    def __post_init__(self):
        # Load PaliGemma tokenizer for encoding CoT text
        path = download.maybe_download(
            "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            object.__setattr__(
                self,
                "_tokenizer",
                sentencepiece.SentencePieceProcessor(model_proto=f.read()),
            )

    def __call__(self, data: dict) -> dict:
        tokenized_prompt = data["tokenized_prompt"]
        tokenized_prompt_mask = data["tokenized_prompt_mask"]
        max_len = len(tokenized_prompt)

        cot_text = data.pop("cot_text", None)

        if cot_text is None or cot_text == "":
            # No CoT text: VLA-only mode
            # All masks default to safe values (bidirectional, no CE loss, all in cache)
            data["token_ar_mask"] = np.zeros(max_len, dtype=np.int32)
            data["token_loss_mask"] = np.zeros(max_len, dtype=bool)
            data["token_kv_cache_mask"] = tokenized_prompt_mask.copy()
            return data

        # Tokenize CoT text (without BOS — it's already in the prompt)
        cot_tokens = self._tokenizer.encode(cot_text, add_bos=False)

        # Find where prompt tokens end
        prompt_len = int(tokenized_prompt_mask.sum())

        # Calculate how many CoT tokens we can fit (leaving room for EOS)
        available = max_len - prompt_len - 1  # -1 for EOS
        if available <= 0:
            # No room for CoT, fall back to VLA-only
            data["token_ar_mask"] = np.zeros(max_len, dtype=np.int32)
            data["token_loss_mask"] = np.zeros(max_len, dtype=bool)
            data["token_kv_cache_mask"] = tokenized_prompt_mask.copy()
            return data

        cot_tokens = cot_tokens[:available]
        cot_len = len(cot_tokens)

        # Build new tokenized_prompt: [prompt_tokens..., cot_tokens..., EOS, pad...]
        new_tokens = tokenized_prompt.copy()
        new_mask = tokenized_prompt_mask.copy()

        # Append CoT tokens
        new_tokens[prompt_len : prompt_len + cot_len] = np.array(
            cot_tokens, dtype=new_tokens.dtype
        )
        new_mask[prompt_len : prompt_len + cot_len] = True

        # Append EOS
        eos_pos = prompt_len + cot_len
        new_tokens[eos_pos] = self.eos_token_id
        new_mask[eos_pos] = True

        # Build ar_mask: 0 for prompt, 1 for CoT + EOS
        ar_mask = np.zeros(max_len, dtype=np.int32)
        ar_mask[prompt_len : eos_pos + 1] = 1

        # Build loss_mask: True for CoT + EOS (predict these tokens)
        loss_mask = np.zeros(max_len, dtype=bool)
        loss_mask[prompt_len : eos_pos + 1] = True

        # Build kv_cache_mask: True for all valid tokens EXCEPT EOS
        kv_cache_mask = new_mask.copy()
        kv_cache_mask[eos_pos] = False

        data["tokenized_prompt"] = new_tokens
        data["tokenized_prompt_mask"] = new_mask
        data["token_ar_mask"] = ar_mask
        data["token_loss_mask"] = loss_mask
        data["token_kv_cache_mask"] = kv_cache_mask

        return data
