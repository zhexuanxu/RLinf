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

"""Byte-exact parity of the vendored BEHAVIOR pi05 preprocessing primitives.

These tests cross-check the self-contained ``tokenizer`` and ``normalize``
modules against the installed ``openpi`` implementations they were ported from,
proving the vendored copies reproduce upstream exactly. They are skipped if the
installed ``openpi`` (or the cached norm stats) is unavailable.

Note: importing ``openpi`` here is fine — these are *tests*, not the package; the
package itself stays free of installed-``openpi`` imports (see
``test_openpi_pytorch_isolation.py``).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from rlinf.models.embodiment.openpi_pytorch.normalize import (
    load_norm_stats,
    normalize_quantile,
    unnormalize_quantile,
)
from rlinf.models.embodiment.openpi_pytorch.tokenizer import PaligemmaTokenizer

_NORM_STATS_DIR = pathlib.Path(
    "/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999/physical-intelligence/behavior"
)


def test_tokenizer_parity_against_openpi():
    openpi_tok_mod = pytest.importorskip("openpi.models.tokenizer")
    vendored = PaligemmaTokenizer(max_len=48)
    upstream = openpi_tok_mod.PaligemmaTokenizer(max_len=48)

    rng = np.random.default_rng(0)
    state = rng.uniform(-1.0, 1.0, size=(23,))
    for prompt in ["turn on radio", "pick up the cup", "  weird_prompt\nwith newline "]:
        # pi05 format: state is part of the discrete language input.
        v_tok, v_mask = vendored.tokenize(prompt, state)
        u_tok, u_mask = upstream.tokenize(prompt, state)
        assert np.array_equal(v_tok, u_tok), f"token mismatch for prompt={prompt!r}"
        assert np.array_equal(v_mask, u_mask), f"mask mismatch for prompt={prompt!r}"

        # pi0 format: no state.
        v_tok0, v_mask0 = vendored.tokenize(prompt, None)
        u_tok0, u_mask0 = upstream.tokenize(prompt, None)
        assert np.array_equal(v_tok0, u_tok0)
        assert np.array_equal(v_mask0, u_mask0)


def test_normalize_parity_against_openpi():
    if not (_NORM_STATS_DIR / "norm_stats.json").exists():
        pytest.skip("BEHAVIOR norm_stats.json not available")
    openpi_transforms = pytest.importorskip("openpi.transforms")
    openpi_normalize = pytest.importorskip("openpi.shared.normalize")

    vendored_stats = load_norm_stats(_NORM_STATS_DIR)
    upstream_stats = openpi_normalize.load(_NORM_STATS_DIR)

    rng = np.random.default_rng(1)

    # Single-key transforms isolate each key (openpi's Unnormalize is strict and
    # would otherwise require every norm_stats key to be present in the data).
    state_norm = openpi_transforms.Normalize(
        {"state": upstream_stats["state"]}, use_quantiles=True
    )
    actions_unnorm = openpi_transforms.Unnormalize(
        {"actions": upstream_stats["actions"]}, use_quantiles=True
    )

    # Quantile normalize on a 23-dim state (stats are 32-dim; upstream slices).
    state = rng.uniform(-2.0, 2.0, size=(23,))
    v_norm = normalize_quantile(state, vendored_stats["state"])
    u_norm = state_norm({"state": state.copy()})["state"]
    np.testing.assert_allclose(v_norm, u_norm, rtol=0, atol=0)

    # Quantile unnormalize on a (B, 32) action tensor.
    actions = rng.uniform(-1.0, 1.0, size=(2, 32))
    v_un = unnormalize_quantile(actions, vendored_stats["actions"])
    u_un = actions_unnorm({"actions": actions.copy()})["actions"]
    np.testing.assert_allclose(v_un, u_un, rtol=0, atol=0)
