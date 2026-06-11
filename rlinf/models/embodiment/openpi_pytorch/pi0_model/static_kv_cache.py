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

"""Preallocated per-layer KV cache for autoregressive subtask generation.

Ported from the reference pi05 implementation's ``StaticKVCache`` /
``left_to_right_align`` design, adapted to this package's attention layout
(``[batch, sequence, kv_heads, head_dim]`` per layer, K/V written after RoPE).

The combination works as follows: each row's valid prefix tokens are
right-aligned (:func:`left_to_right_align`), so every row's next free slot is
the SAME buffer column. Generation then writes one column per step at a shared
write cursor — including masked dummy writes for rows that already emitted EOS;
a per-row column-validity mask (maintained by the caller) keeps those columns,
and every row's EOS, invisible to all subsequent attention. After generation
the cache is frozen: the action expert's K/V are concatenated after the buffer
per denoise step instead of being written into it.
"""

from __future__ import annotations

import torch

__all__ = ["StaticKVCache", "StaticLayerKV", "left_to_right_align"]


class StaticLayerKV:
    """One layer's view of the preallocated cache (consumed by attention)."""

    __slots__ = ("cache", "k", "v")

    def __init__(self, cache: "StaticKVCache", k: torch.Tensor, v: torch.Tensor):
        self.cache = cache
        self.k = k
        self.v = v

    def write(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Write new (already-RoPE'd) K/V at the shared write cursor."""
        col = self.cache.write_col
        length = k_new.shape[1]
        if col + length > self.cache.max_len:
            raise RuntimeError(
                f"static KV cache overflow: writing {length} tokens at column "
                f"{col} exceeds the preallocated length {self.cache.max_len}."
            )
        self.k[:, col : col + length] = k_new
        self.v[:, col : col + length] = v_new


class StaticKVCache:
    """Preallocated per-layer K/V buffers with a shared write cursor.

    Args:
        batch_size: Batch dimension of the buffers.
        max_len: Total columns (prefill size + generation budget).
        num_layers: Transformer depth.
        num_kv_heads: KV heads per layer.
        head_dim: Head dimension.
        dtype: Buffer dtype (the model's embed dtype).
        device: Buffer device.
    """

    def __init__(
        self,
        *,
        batch_size: int,
        max_len: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ):
        self.max_len = max_len
        self.write_col = 0
        self.frozen = False
        shape = (batch_size, max_len, num_kv_heads, head_dim)
        self._layers = [
            StaticLayerKV(
                self,
                torch.zeros(shape, dtype=dtype, device=device),
                torch.zeros(shape, dtype=dtype, device=device),
            )
            for _ in range(num_layers)
        ]

    def layers(self) -> list[StaticLayerKV]:
        """Per-layer handles, threaded through the transformer like tuple caches."""
        return list(self._layers)

    def set_write_col(self, col: int) -> None:
        """Position the shared write cursor (prefill at 0, step ``t`` at P+t)."""
        self.write_col = col
        self.frozen = False

    def freeze(self) -> None:
        """Stop writes: subsequent K/V (the action expert's) are concatenated
        after the buffer by attention instead of being stored."""
        self.frozen = True


def left_to_right_align(
    x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Roll each row so its valid tokens end at the last column.

    Right alignment gives every row the same next-token slot, which is what
    lets the static cache use one shared write column per generation step.

    Args:
        x: ``(B, S, D)`` embedded tokens whose valid entries are contiguous
            from the left (the right-padded convention).
        input_mask: ``(B, S)`` bool validity.
        attn_mask: ``(B, S, S)`` attention mask over the same positions.

    Returns:
        The rolled ``(x, input_mask, attn_mask)``.
    """
    batch_size, seq_len = input_mask.shape
    device = x.device

    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    seqlens = (input_mask.float() * arange).max(dim=1).values.long() + 1

    rolled = (arange + seqlens.unsqueeze(1)) % seq_len
    batch_index = torch.arange(batch_size, device=device).unsqueeze(1)

    x_rolled = x[batch_index, rolled]
    mask_rolled = input_mask[batch_index, rolled]
    attn_rolled = attn_mask[
        batch_index.unsqueeze(2), rolled.unsqueeze(2), rolled.unsqueeze(1)
    ]
    return x_rolled, mask_rolled, attn_rolled
