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

"""Focused CPU tests for the π0.5 VLM→VLA model building blocks."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_pytorch.pi0_model import gemma
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import (
    Pi0,
    block_suffix_from_excluded_prefix,
    make_attn_mask,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_pytorch.pi0_model.static_kv_cache import (
    StaticKVCache,
    left_to_right_align,
)


def test_vlm_vla_config_requires_pi05():
    with pytest.raises(ValueError, match="requires the pi05"):
        Pi0Config(mode="vlm_vla")

    config = Pi0Config(pi05=True, mode="vlm_vla")
    observation = config.fake_obs(batch_size=2)
    assert observation.token_ar_mask.shape == (2, config.max_token_len)
    assert observation.token_loss_mask[:, -1].all()
    assert not observation.token_kv_cache_mask[:, -1].any()


def test_make_attn_mask_accepts_per_batch_ar_masks():
    valid = torch.ones(2, 5, dtype=torch.bool)
    ar = torch.tensor(
        [
            [False, False, True, True, False],
            [False, False, False, True, True],
        ]
    )
    mask = make_attn_mask(valid, ar)

    assert mask[0, 1].tolist() == [True, True, False, False, False]
    assert mask[0, 4].tolist() == [True] * 5
    assert mask[1, 0].tolist() == [True, True, True, False, False]


def test_suffix_cannot_attend_kv_excluded_eos():
    batch_size, image_len, text_len, suffix_len = 2, 3, 4, 2
    prefix_len = image_len + text_len
    attn = torch.ones(
        batch_size,
        prefix_len + suffix_len,
        prefix_len + suffix_len,
        dtype=torch.bool,
    )
    kv_mask = torch.ones(batch_size, text_len, dtype=torch.bool)
    kv_mask[0, -1] = False
    kv_mask[1, -2] = False

    result = block_suffix_from_excluded_prefix(attn, kv_mask, prefix_len)

    for query in range(prefix_len, prefix_len + suffix_len):
        assert not result[0, query, image_len + text_len - 1]
        assert not result[1, query, image_len + text_len - 2]
    assert result[:, :prefix_len].all()


def test_left_to_right_align_preserves_attention():
    x = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    valid = torch.tensor(
        [
            [True, True, True, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    attn = make_attn_mask(valid, torch.zeros(6, dtype=torch.bool))

    aligned_x, aligned_valid, aligned_attn = left_to_right_align(x, valid, attn)

    assert aligned_valid[0].tolist() == [False] * 3 + [True] * 3
    assert aligned_valid[1].tolist() == [False] + [True] * 5
    assert torch.equal(aligned_x[0, 3:], x[0, :3])
    expected = make_attn_mask(aligned_valid, torch.zeros(6, dtype=torch.bool))
    assert torch.equal(aligned_attn, expected)


def _static_cache(*, max_len=8, num_layers=2, batch_size=2):
    return StaticKVCache(
        batch_size=batch_size,
        max_len=max_len,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float32,
        device="cpu",
    )


def test_static_cache_write_freeze_and_overflow():
    cache = _static_cache(max_len=6)
    layer = cache.layers()[0]
    prefix = torch.randn(2, 5, 1, 4)
    layer.write(prefix, prefix + 1)
    assert torch.equal(layer.k[:, :5], prefix)

    cache.set_write_col(5)
    step = torch.randn(2, 1, 1, 4)
    layer.write(step, step)
    cache.freeze()
    assert cache.frozen

    cache.set_write_col(5)
    with pytest.raises(RuntimeError, match="overflow"):
        layer.write(torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4))


def _dummy_attention():
    config = gemma.get_config("dummy")
    return gemma.Attention([config, config]), config


def _attention_outputs(attention, config, *, detach):
    torch.manual_seed(0)
    batch_size, prefix_len, suffix_len = 2, 6, 4
    prefix = torch.randn(batch_size, prefix_len, config.width)
    suffix = torch.randn(batch_size, suffix_len, config.width)
    total = prefix_len + suffix_len
    positions = torch.arange(total).expand(batch_size, total)
    ar = torch.tensor([False] * prefix_len + [True] + [False] * (suffix_len - 1))
    mask = make_attn_mask(torch.ones(batch_size, total, dtype=torch.bool), ar)
    attention.detach_prefix_kv = detach
    return attention([prefix, suffix], positions, mask.unsqueeze(1))[0]


def test_split_attention_matches_unified_values():
    attention, config = _dummy_attention()
    unified = _attention_outputs(attention, config, detach=False)
    split = _attention_outputs(attention, config, detach=True)
    torch.testing.assert_close(unified[0], split[0], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(unified[1], split[1], rtol=1e-5, atol=1e-5)


def test_static_attention_cache_matches_tuple_cache():
    attention, config = _dummy_attention()
    torch.manual_seed(1)
    batch_size, prefix_len, suffix_len = 2, 5, 3
    prefix = torch.randn(batch_size, prefix_len, config.width)
    suffix = torch.randn(batch_size, suffix_len, config.width)
    prefix_positions = torch.arange(prefix_len).expand(batch_size, prefix_len)
    prefix_mask = torch.ones(batch_size, 1, prefix_len, prefix_len, dtype=torch.bool)

    _, tuple_cache = attention([prefix, None], prefix_positions, prefix_mask)
    suffix_positions = prefix_len + torch.arange(suffix_len).expand(
        batch_size, suffix_len
    )
    suffix_mask = torch.ones(
        batch_size, 1, suffix_len, prefix_len + suffix_len, dtype=torch.bool
    )
    tuple_output, _ = attention(
        [None, suffix],
        suffix_positions,
        suffix_mask,
        kv_cache=tuple_cache,
    )

    max_new = 2
    cache = StaticKVCache(
        batch_size=batch_size,
        max_len=prefix_len + max_new,
        num_layers=1,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    layer = cache.layers()[0]
    static_prefill, _ = attention(
        [prefix, None],
        prefix_positions,
        F.pad(prefix_mask, (0, max_new)),
        kv_cache=layer,
    )
    tuple_prefill, _ = attention([prefix, None], prefix_positions, prefix_mask)
    torch.testing.assert_close(static_prefill[0], tuple_prefill[0])

    cache.freeze()
    cache_valid = torch.cat(
        [
            torch.ones(batch_size, prefix_len, dtype=torch.bool),
            torch.zeros(batch_size, max_new, dtype=torch.bool),
        ],
        dim=1,
    )
    static_mask = torch.cat(
        [
            cache_valid[:, None, None].expand(batch_size, 1, suffix_len, -1),
            torch.ones(batch_size, 1, suffix_len, suffix_len, dtype=torch.bool),
        ],
        dim=-1,
    )
    static_output, _ = attention(
        [None, suffix],
        suffix_positions,
        static_mask,
        kv_cache=layer,
    )
    torch.testing.assert_close(tuple_output[1], static_output[1])


def _run_gradient_routing(*, detach):
    attention, config = _dummy_attention()
    attention.detach_prefix_kv = detach
    batch_size, prefix_len, suffix_len = 2, 5, 3
    prefix = torch.randn(batch_size, prefix_len, config.width, requires_grad=True)
    suffix = torch.randn(batch_size, suffix_len, config.width, requires_grad=True)
    positions = torch.arange(prefix_len + suffix_len).expand(batch_size, -1)
    ar = torch.tensor([False] * prefix_len + [True] + [False] * (suffix_len - 1))
    mask = make_attn_mask(
        torch.ones(batch_size, prefix_len + suffix_len, dtype=torch.bool), ar
    )
    outputs, _ = attention([prefix, suffix], positions, mask.unsqueeze(1))
    outputs[1].square().mean().backward()
    return prefix.grad, suffix.grad


def test_stop_gradient_detaches_prefix_from_suffix_loss():
    prefix_grad, suffix_grad = _run_gradient_routing(detach=True)
    assert prefix_grad is None or not prefix_grad.any()
    assert suffix_grad is not None and suffix_grad.abs().sum() > 0

    prefix_grad, _ = _run_gradient_routing(detach=False)
    assert prefix_grad is not None and prefix_grad.abs().sum() > 0


def test_eos_exclusion_keeps_train_and_eval_suffix_positions_equal():
    image_len, text_len, suffix_len = 5, 8, 4
    prefix_len = image_len + text_len
    text_valid = torch.tensor([[True, True, True, True, True, True, False, False]])
    # The sixth valid text token is EOS and is deliberately cache-invisible.
    text_kv = torch.tensor([[True, True, True, True, True, False, False, False]])
    input_mask = torch.cat(
        [
            torch.ones(1, image_len, dtype=torch.bool),
            text_valid,
            torch.ones(1, suffix_len, dtype=torch.bool),
        ],
        dim=1,
    )
    position_mask = input_mask.clone()
    position_mask[:, prefix_len - text_len : prefix_len] &= text_kv
    train_suffix_position = (torch.cumsum(position_mask.int(), dim=1) - 1)[
        :, prefix_len
    ]

    eval_cache_valid = torch.cat(
        [
            torch.ones(1, image_len, dtype=torch.bool),
            text_kv[:, :5],
        ],
        dim=1,
    )
    eval_suffix_position = eval_cache_valid.sum(dim=-1)
    assert torch.equal(train_suffix_position, eval_suffix_position)


def _observation(*, with_masks):
    token_masks = {}
    if with_masks:
        ar = torch.zeros(2, 6, dtype=torch.bool)
        ar[:, 3:] = True
        token_masks = {
            "token_ar_mask": ar,
            "token_loss_mask": ar.clone(),
            "token_kv_cache_mask": ~ar,
        }
    return Observation(
        images={"camera": torch.zeros(2, 4, 4, 3)},
        image_masks={"camera": torch.ones(2, dtype=torch.bool)},
        state=torch.zeros(2, 8),
        tokenized_prompt=torch.zeros(2, 6, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 6, dtype=torch.bool),
        **token_masks,
    )


def _pi0_prefix_stub(*, vlm_vla):
    def image_encoder(images):
        return torch.zeros(images.shape[0], 4, 8), None

    return types.SimpleNamespace(
        vlm_vla=vlm_vla,
        pcd=False,
        img=image_encoder,
        llm=types.SimpleNamespace(embed=lambda tokens: torch.zeros(*tokens.shape, 8)),
    )


def test_action_only_prefix_ignores_vlm_masks():
    stub = _pi0_prefix_stub(vlm_vla=False)
    _, _, plain = Pi0.embed_prefix(stub, _observation(with_masks=False))
    _, _, hostile = Pi0.embed_prefix(stub, _observation(with_masks=True))
    assert plain.dim() == 1
    assert torch.equal(plain, hostile)
    assert not hostile.any()

    observation = _observation(with_masks=True)
    _, _, vlm_ar = Pi0.embed_prefix(_pi0_prefix_stub(vlm_vla=True), observation)
    assert vlm_ar.dim() == 2
    assert torch.equal(vlm_ar[:, -6:], observation.token_ar_mask)


def test_action_only_sampling_keeps_tuple_cache_api(monkeypatch):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model
    from rlinf.models.embodiment.openpi_pytorch.pi0_model import pi0 as pi0_module

    monkeypatch.setattr(
        model, "preprocess_observation", lambda observation, train=False: observation
    )

    def forbidden_static_cache(*args, **kwargs):
        raise AssertionError("action-only sampling must not use StaticKVCache")

    monkeypatch.setattr(pi0_module, "StaticKVCache", forbidden_static_cache)
    tuple_cache = ((torch.zeros(1), torch.zeros(1)),)
    stub = types.SimpleNamespace(
        action_horizon=3,
        action_dim=2,
        build_prefix_cache=lambda observation: (
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            tuple_cache,
        ),
        run_suffix=lambda observation, x_t, time, cache, mask: torch.zeros(1, 3, 4),
        velocity_from_suffix=lambda suffix: torch.zeros(1, 3, 2),
    )
    stub._denoise_actions = types.MethodType(Pi0._denoise_actions, stub)
    observation = Observation(images={}, image_masks={}, state=torch.zeros(1, 2))
    noise = torch.randn(1, 3, 2)

    actions = Pi0.sample_actions(stub, observation, num_steps=2, noise=noise)
    torch.testing.assert_close(actions, noise)


def _language_loss_stub(*, vocab=32, width=16):
    embedder = gemma.Embedder(vocab_size=vocab, embed_dim=width)
    return types.SimpleNamespace(llm=types.SimpleNamespace(embedder=embedder))


def test_language_loss_is_shifted_and_masked():
    torch.manual_seed(2)
    stub = _language_loss_stub()
    prefix_out = torch.randn(2, 3 + 6, 16)
    tokens = torch.randint(0, 32, (2, 6))
    loss_mask = torch.zeros(2, 6, dtype=torch.bool)
    loss_mask[:, 3:] = True

    loss, accuracy = Pi0._compute_language_loss(stub, prefix_out, tokens, loss_mask)
    assert loss.ndim == 0
    assert 0 <= accuracy <= 1

    unsupervised_change = tokens.clone()
    unsupervised_change[:, 1] = (unsupervised_change[:, 1] + 7) % 32
    same_loss, _ = Pi0._compute_language_loss(
        stub, prefix_out, unsupervised_change, loss_mask
    )
    torch.testing.assert_close(loss, same_loss)

    supervised_change = tokens.clone()
    supervised_change[:, 4] = (supervised_change[:, 4] + 7) % 32
    changed_loss, _ = Pi0._compute_language_loss(
        stub, prefix_out, supervised_change, loss_mask
    )
    assert not torch.isclose(loss, changed_loss)


class _ScheduledEmbedder:
    @staticmethod
    def decode(hidden):
        return hidden


class _ScheduledLLM:
    """Return fixed logits: row 0 ends immediately, row 1 after one token."""

    def __init__(self):
        config = types.SimpleNamespace(depth=1, num_kv_heads=1, head_dim=2)
        self.configs = [config]
        self.embedder = _ScheduledEmbedder()
        self.calls = 0

    @staticmethod
    def embed(tokens):
        return torch.zeros(*tokens.shape, 4)

    def __call__(self, embedded, positions, mask, adarms_cond=None, *, kv_cache=None):
        batch_size = positions.shape[0]
        sequence_len = positions.shape[1]
        hidden = torch.zeros(batch_size, sequence_len, 4)
        if self.calls == 0:
            hidden[0, -1, 1] = 10  # EOS
            hidden[1, -1, 2] = 10  # one generated response token
        else:
            hidden[:, -1, 1] = 10  # EOS
        self.calls += 1
        return [hidden, None], kv_cache


def test_generation_terminates_rows_independently(monkeypatch):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model import model

    monkeypatch.setattr(
        model, "preprocess_observation", lambda observation, train=False: observation
    )
    llm = _ScheduledLLM()
    stub = types.SimpleNamespace(
        vlm_vla=True,
        pcd=False,
        max_new_tokens=3,
        language_temperature=0.0,
        embed_dtype=torch.float32,
        llm=llm,
    )

    def embed_prefix(observation):
        return (
            torch.zeros(2, 4, 4),
            torch.tensor([[True, True, False, False], [True, True, True, False]]),
            torch.zeros(2, 4, dtype=torch.bool),
        )

    stub.embed_prefix = embed_prefix
    observation = Observation(
        images={},
        image_masks={},
        state=torch.zeros(2, 2),
        tokenized_prompt=torch.zeros(2, 2, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(2, 2, dtype=torch.bool),
        token_ar_mask=torch.zeros(2, 2, dtype=torch.bool),
    )

    generation = Pi0.generate_language(stub, observation, eos_token_id=1)

    assert generation["tokens"].tolist() == [[1, 1], [2, 1]]
    assert generation["eos_steps"].tolist() == [0, 1]
    assert generation["terminated"].tolist() == [True, True]
    cache_valid = generation["cache_valid_mask"]
    assert cache_valid[:, -3:].tolist() == [
        [False, False, False],
        [True, False, False],
    ]
