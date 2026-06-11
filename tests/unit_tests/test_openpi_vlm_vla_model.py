# Copyright 2026 The RLinf Authors.
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

"""Unit tests for the pi0.5 VLM token-output building blocks.

Covers, on CPU with the dummy Gemma variant:

* the subtask-supervision tokenizer mask table (prefix/response/EOS/padding);
* attention-mask construction (block-causal cumsum semantics, per-batch
  autoregressive masks, KV-exclusion of EOS from action-expert queries);
* ``left_to_right_align`` and the preallocated static KV cache (write cursor,
  frozen concatenation, overflow);
* the unified-vs-split attention equivalence and the knowledge-insulation
  gradient routing (flow gradients reach only the action expert; language CE
  gradients reach the VLM);
* the next-token CE mechanics (shift, per-sample normalization, padding
  isolation).
"""

import types
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import (
    Pi0,
    block_suffix_from_excluded_prefix,
    make_attn_mask,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.static_kv_cache import (
    StaticKVCache,
    left_to_right_align,
)

TOKENIZER_PATH = Path(
    "/mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model"
)

requires_tokenizer = pytest.mark.skipif(
    not TOKENIZER_PATH.exists(), reason="PaliGemma tokenizer model not available"
)


@requires_tokenizer
class TestSubtaskTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
            PaligemmaTokenizer,
        )

        return PaligemmaTokenizer(TOKENIZER_PATH, max_len=64)

    @pytest.fixture(scope="class")
    def state(self):
        return np.linspace(-0.9, 0.9, 5)

    def test_training_mask_table(self, tokenizer, state):
        tokens, valid, ar, loss, kv = tokenizer.tokenize_with_subtask(
            "Turn on the radio.", state, "press radio"
        )
        n_valid = int(valid.sum())
        supervised = np.flatnonzero(loss)
        prefix_len = int(supervised[0])
        eos_pos = int(supervised[-1])

        assert tokens[eos_pos] == tokenizer.eos_token_id
        assert eos_pos == n_valid - 1, "EOS is the last valid token"
        # Prefix: bidirectional, no loss, in the KV contract.
        assert not ar[:prefix_len].any()
        assert not loss[:prefix_len].any()
        assert kv[:prefix_len].all()
        # Response: causal, loss, in the KV contract.
        assert ar[prefix_len:eos_pos].all()
        assert loss[prefix_len:eos_pos].all()
        assert kv[prefix_len:eos_pos].all()
        # EOS: causal, loss, NOT visible to the action expert.
        assert ar[eos_pos] and loss[eos_pos] and not kv[eos_pos]
        # Padding: fully inert.
        assert not valid[n_valid:].any()
        assert not ar[n_valid:].any()
        assert not loss[n_valid:].any()
        assert not kv[n_valid:].any()

        decoded_prefix = tokenizer.decode(tokens[:prefix_len])
        assert decoded_prefix.startswith("Task: turn on the radio. State:")
        assert decoded_prefix.rstrip().endswith("Subtask:")
        assert tokenizer.decode(tokens[prefix_len:eos_pos]) == "press radio."

    def test_generation_prefix_has_no_supervised_positions(self, tokenizer, state):
        tokens, valid, ar, loss, kv = tokenizer.tokenize_with_subtask(
            "Turn on the radio.", state, response=None
        )
        n_valid = int(valid.sum())
        assert not loss.any()
        assert not ar.any()
        assert kv[:n_valid].all() and not kv[n_valid:].any()
        assert tokenizer.eos_token_id not in tokens[:n_valid]

    def test_overlong_sequence_rejected_not_truncated(self, tokenizer, state):
        with pytest.raises(ValueError, match="max_token_len"):
            tokenizer.tokenize_with_subtask("word " * 200, state, "press radio")

    def test_action_only_format_unchanged(self, tokenizer, state):
        tokens, mask = tokenizer.tokenize("Turn on the radio", state)
        n_valid = int(np.asarray(mask).sum())
        decoded = tokenizer.decode(np.asarray(tokens)[:n_valid])
        assert decoded.startswith("Task: Turn on the radio, State:")
        assert decoded.endswith(";\nAction: ") or decoded.rstrip().endswith(
            ";\nAction:"
        )


class TestAttentionMasks:
    def test_make_attn_mask_accepts_per_batch_ar(self):
        valid = torch.ones(2, 5, dtype=torch.bool)
        ar = torch.tensor(
            [
                [False, False, True, True, False],
                [False, False, False, True, True],
            ]
        )
        mask = make_attn_mask(valid, ar)
        # Row 0: blocks are {0,1}, {2}, {3,4}: position 2 sees 0..2 only.
        assert mask[0, 2].tolist() == [True, True, True, False, False]
        assert mask[0, 1].tolist() == [True, True, False, False, False]
        assert mask[0, 4].tolist() == [True] * 5
        # Row 1 has a different block structure: position 2 is bidirectional
        # with 0,1 and 2 (block 0 spans 0..2).
        assert mask[1, 2].tolist() == [True, True, True, False, False]
        assert mask[1, 0].tolist() == [True, True, True, False, False]

    def test_block_suffix_from_excluded_prefix(self):
        batch, text_len, num_images, suffix_len = 2, 4, 3, 2
        prefix_len = num_images + text_len
        total = prefix_len + suffix_len
        attn = torch.ones(batch, total, total, dtype=torch.bool)
        kv_mask = torch.ones(batch, text_len, dtype=torch.bool)
        kv_mask[0, -1] = False  # row 0's EOS at the last text position
        kv_mask[1, -2] = False  # row 1's EOS one position earlier

        blocked = block_suffix_from_excluded_prefix(attn, kv_mask, prefix_len)

        eos_col_row0 = num_images + text_len - 1
        eos_col_row1 = num_images + text_len - 2
        for suffix_row in range(prefix_len, total):
            assert not blocked[0, suffix_row, eos_col_row0]
            assert not blocked[1, suffix_row, eos_col_row1]
            assert blocked[0, suffix_row, eos_col_row1]
        # Prefix self-attention is untouched, including onto the EOS column.
        assert blocked[0, :prefix_len, eos_col_row0].all()
        assert blocked[0, :, :eos_col_row0].all()


class TestLeftToRightAlign:
    def test_rolls_valid_tokens_to_the_right(self):
        batch, seq, dim = 2, 6, 3
        x = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(
            batch, seq, dim
        )
        valid = torch.tensor(
            [
                [True, True, True, False, False, False],
                [True, True, True, True, True, False],
            ]
        )
        attn = make_attn_mask(valid, torch.zeros(seq, dtype=torch.bool))
        x_r, valid_r, attn_r = left_to_right_align(x, valid, attn)

        assert valid_r[0].tolist() == [False] * 3 + [True] * 3
        assert valid_r[1].tolist() == [False] * 1 + [True] * 5
        assert torch.equal(x_r[0, 3:], x[0, :3])
        assert torch.equal(x_r[1, 1:], x[1, :5])
        # The rolled attention agrees with a mask rebuilt on rolled validity.
        rebuilt = make_attn_mask(valid_r, torch.zeros(seq, dtype=torch.bool))
        assert torch.equal(attn_r, rebuilt)


class TestStaticKVCache:
    def _cache(self, max_len=8, layers=2, batch=2):
        return StaticKVCache(
            batch_size=batch,
            max_len=max_len,
            num_layers=layers,
            num_kv_heads=1,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )

    def test_write_cursor_and_freeze(self):
        cache = self._cache()
        layer = cache.layers()[0]
        prefill = torch.randn(2, 6, 1, 4)
        cache.set_write_col(0)
        layer.write(prefill, prefill + 1)
        assert torch.equal(layer.k[:, :6], prefill)

        step = torch.randn(2, 1, 1, 4)
        cache.set_write_col(6)
        layer.write(step, step)
        assert torch.equal(layer.k[:, 6:7], step)

        cache.freeze()
        assert cache.frozen

    def test_overflow_raises(self):
        cache = self._cache(max_len=4)
        layer = cache.layers()[0]
        cache.set_write_col(3)
        with pytest.raises(RuntimeError, match="overflow"):
            layer.write(torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4))


def _dummy_attention():
    config = gemma.get_config("dummy")
    return gemma.Attention([config, config]), config


class TestAttentionEquivalence:
    """The refactored attention core must match a hand-rolled reference, and
    the split (knowledge-insulation) path must match the unified path in value.
    """

    def _run(self, attention, config, prefix_len, suffix_len, detach):
        torch.manual_seed(0)
        batch = 2
        prefix = torch.randn(batch, prefix_len, config.width)
        suffix = torch.randn(batch, suffix_len, config.width)
        total = prefix_len + suffix_len
        positions = torch.arange(total).expand(batch, total)
        ar = torch.tensor([False] * prefix_len + [True] + [False] * (suffix_len - 1))
        mask = make_attn_mask(torch.ones(batch, total, dtype=torch.bool), ar)
        attention.detach_prefix_kv = detach
        outputs, _ = attention([prefix, suffix], positions, mask.unsqueeze(1))
        return outputs

    def test_split_matches_unified_values(self):
        attention, config = _dummy_attention()
        unified = self._run(attention, config, 6, 4, detach=False)
        split = self._run(attention, config, 6, 4, detach=True)
        torch.testing.assert_close(unified[0], split[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(unified[1], split[1], rtol=1e-5, atol=1e-5)

    def test_static_cache_matches_tuple_cache(self):
        attention, config = _dummy_attention()
        torch.manual_seed(1)
        batch, prefix_len, suffix_len = 2, 5, 3
        prefix = torch.randn(batch, prefix_len, config.width)
        suffix = torch.randn(batch, suffix_len, config.width)

        prefix_positions = torch.arange(prefix_len).expand(batch, prefix_len)
        prefix_mask = torch.ones(batch, 1, prefix_len, prefix_len, dtype=torch.bool)

        # Tuple-cache path (the action-only behavior).
        _, tuple_cache = attention([prefix, None], prefix_positions, prefix_mask)
        suffix_positions = prefix_len + torch.arange(suffix_len).expand(
            batch, suffix_len
        )
        suffix_mask = torch.ones(
            batch, 1, suffix_len, prefix_len + suffix_len, dtype=torch.bool
        )
        tuple_out, _ = attention(
            [None, suffix], suffix_positions, suffix_mask, kv_cache=tuple_cache
        )

        # Static-cache path: prefill writes the buffer, freeze, then denoise.
        max_new = 2
        cache = StaticKVCache(
            batch_size=batch,
            max_len=prefix_len + max_new,
            num_layers=1,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        layer = cache.layers()[0]
        prefill_mask = F.pad(prefix_mask, (0, max_new))
        cache.set_write_col(0)
        static_prefill_out, _ = attention(
            [prefix, None], prefix_positions, prefill_mask, kv_cache=layer
        )
        unified_prefill_out, _ = attention(
            [prefix, None], prefix_positions, prefix_mask
        )
        torch.testing.assert_close(
            static_prefill_out[0], unified_prefill_out[0], rtol=1e-5, atol=1e-5
        )

        cache.freeze()
        # Suffix attends [buffer ++ its own K/V]; generation columns invalid.
        keys_valid = torch.cat(
            [
                torch.ones(batch, prefix_len, dtype=torch.bool),
                torch.zeros(batch, max_new, dtype=torch.bool),
            ],
            dim=1,
        )
        denoise_mask = torch.cat(
            [
                keys_valid[:, None, None, :].expand(batch, 1, suffix_len, -1),
                torch.ones(batch, 1, suffix_len, suffix_len, dtype=torch.bool),
            ],
            dim=-1,
        )
        static_out, _ = attention(
            [None, suffix], suffix_positions, denoise_mask, kv_cache=layer
        )
        torch.testing.assert_close(tuple_out[1], static_out[1], rtol=1e-5, atol=1e-5)


class TestStopGradientRouting:
    """Flow-loss gradients must reach only the action expert; language-loss
    gradients must reach the VLM (prefix expert)."""

    def _module(self, detach):
        torch.manual_seed(2)
        config = gemma.get_config("dummy")
        module = gemma.Module(
            configs=[config, config],
            embed_dtype="float32",
            adarms=[False, False],
            use_gradient_checkpointing=False,
        )
        if detach:
            for layer in module.layers:
                layer.attn.detach_prefix_kv = True
        return module, config

    def _forward(self, module, config):
        batch, prefix_len, suffix_len = 2, 6, 4
        prefix = torch.randn(batch, prefix_len, config.width)
        suffix = torch.randn(batch, suffix_len, config.width)
        total = prefix_len + suffix_len
        positions = torch.arange(total).expand(batch, total)
        ar = torch.tensor([False] * prefix_len + [True] + [False] * (suffix_len - 1))
        mask = make_attn_mask(torch.ones(batch, total, dtype=torch.bool), ar)
        outputs, _ = module([prefix, suffix], positions, mask)
        return outputs

    @staticmethod
    def _expert_params(module, expert):
        import re

        # Per-expert modules are indexed ModuleLists: the expert id is the
        # index right after one of these names (e.g. layers.3.attn.q_proj.0).
        pattern = re.compile(
            r"\.(?:q_proj|k_proj|v_proj|o_proj|mlps|pre_attention_norms"
            r"|pre_ffw_norms)\.(\d+)\."
        )
        for name, param in module.named_parameters():
            match = pattern.search(name)
            if match and int(match.group(1)) == expert:
                yield name, param

    @staticmethod
    def _has_nonzero_grad(params):
        return any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in params)

    def test_flow_gradients_blocked_from_vlm_when_detached(self):
        module, config = self._module(detach=True)
        outputs = self._forward(module, config)
        outputs[1].square().mean().backward()  # flow-style loss on the suffix
        assert not self._has_nonzero_grad(self._expert_params(module, 0)), (
            "flow gradients leaked into the VLM expert"
        )
        assert self._has_nonzero_grad(self._expert_params(module, 1))

    def test_language_gradients_reach_vlm_when_detached(self):
        module, config = self._module(detach=True)
        outputs = self._forward(module, config)
        outputs[0].square().mean().backward()  # CE-style loss on the prefix
        assert self._has_nonzero_grad(self._expert_params(module, 0))

    def test_flow_gradients_reach_vlm_without_detach(self):
        module, config = self._module(detach=False)
        outputs = self._forward(module, config)
        outputs[1].square().mean().backward()
        assert self._has_nonzero_grad(self._expert_params(module, 0))


class TestSuffixPositionParity:
    """The action expert's RoPE position base must be identical between the
    SFT forward (where the EOS is an input token) and the eval denoise (where
    the EOS never enters the cache): KV-invisible tokens occupy no position
    slot."""

    def test_train_suffix_base_matches_eval_cache_base(self):
        batch, num_images = 2, 5
        prefix_text, response, eos, pad = 4, 3, 1, 2
        text_len = prefix_text + response + eos + pad
        prefix_len = num_images + text_len
        suffix_len = 4

        text_valid = torch.tensor(
            [[True] * (prefix_text + response + eos) + [False] * pad] * batch
        )
        kv_mask = torch.tensor(
            [[True] * (prefix_text + response) + [False] * (eos + pad)] * batch
        )
        input_mask = torch.cat(
            [
                torch.ones(batch, num_images, dtype=torch.bool),
                text_valid,
                torch.ones(batch, suffix_len, dtype=torch.bool),
            ],
            dim=1,
        )

        # The SFT formula: cumsum over the validity with KV-invisible text
        # positions removed (mirrors Pi0.compute_loss in vlm_vla mode).
        position_mask = input_mask.clone()
        position_mask[:, prefix_len - text_len : prefix_len] &= kv_mask
        train_positions = torch.cumsum(position_mask.int(), dim=1) - 1
        train_suffix_base = train_positions[:, prefix_len]

        # The eval formula: the denoise suffix continues from the number of
        # KV-visible cache columns (mirrors Pi0._denoise_actions after
        # generation, where the generated response has `response` tokens and
        # no EOS column).
        cache_valid = torch.cat(
            [
                torch.ones(batch, num_images, dtype=torch.bool),
                torch.ones(batch, prefix_text, dtype=torch.bool),
                torch.ones(batch, response, dtype=torch.bool),
            ],
            dim=1,
        )
        eval_suffix_base = (
            cache_valid.sum(dim=-1)
            + torch.cumsum(torch.ones(batch, suffix_len, dtype=torch.int), dim=-1)[:, 0]
            - 1
        )

        assert torch.equal(train_suffix_base, eval_suffix_base)


class TestLanguageLoss:
    def _stub(self, vocab=32, width=16):
        embedder = gemma.Embedder(vocab_size=vocab, embed_dim=width)
        return types.SimpleNamespace(llm=types.SimpleNamespace(embedder=embedder))

    def test_shift_and_mask_isolation(self):
        torch.manual_seed(3)
        stub = self._stub()
        batch, num_images, text_len, width = 2, 3, 6, 16
        prefix_out = torch.randn(batch, num_images + text_len, width)
        tokens = torch.randint(0, 32, (batch, text_len))
        loss_mask = torch.zeros(batch, text_len, dtype=torch.bool)
        loss_mask[:, 3:] = True  # supervise the last three positions

        ce, acc = Pi0._compute_language_loss(stub, prefix_out, tokens, loss_mask)
        assert ce.ndim == 0 and 0.0 <= acc <= 1.0

        # Changing an UNsupervised target leaves the loss unchanged...
        tokens_perturbed = tokens.clone()
        tokens_perturbed[:, 1] = (tokens_perturbed[:, 1] + 7) % 32
        ce_same, _ = Pi0._compute_language_loss(
            stub, prefix_out, tokens_perturbed, loss_mask
        )
        torch.testing.assert_close(ce, ce_same)

        # ...changing a supervised target does not.
        tokens_supervised = tokens.clone()
        tokens_supervised[:, 4] = (tokens_supervised[:, 4] + 7) % 32
        ce_diff, _ = Pi0._compute_language_loss(
            stub, prefix_out, tokens_supervised, loss_mask
        )
        assert not torch.isclose(ce, ce_diff)

    def test_perfect_prediction_gives_full_accuracy(self):
        stub = self._stub()
        batch, num_images, text_len = 1, 2, 5
        width = stub.llm.embedder.embed_dim
        tokens = torch.arange(text_len).unsqueeze(0)
        # Build hidden states whose decode() argmax equals the next token.
        weight = stub.llm.embedder.embedding.weight
        prefix_out = torch.zeros(batch, num_images + text_len, width)
        for position in range(text_len - 1):
            prefix_out[0, num_images + position] = weight[tokens[0, position + 1]]
        loss_mask = torch.ones(batch, text_len, dtype=torch.bool)
        _, acc = Pi0._compute_language_loss(stub, prefix_out, tokens, loss_mask)
        assert acc == pytest.approx(1.0)
