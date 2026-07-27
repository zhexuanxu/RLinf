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

"""Preallocated per-layer KV cache for autoregressive subtask generation."""

from __future__ import annotations

import torch

__all__ = ["StaticKVCache", "StaticLayerKV", "left_to_right_align"]


class StaticLayerKV:
    """One transformer's layer view of a shared preallocated KV cache."""

    __slots__ = ("cache", "k", "v")

    def __init__(self, cache: StaticKVCache, k: torch.Tensor, v: torch.Tensor):
        self.cache = cache
        self.k = k
        self.v = v

    def write(self, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        """Write already-RoPE-transformed K/V at the shared cursor."""
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
    """Preallocate one K/V buffer per transformer layer."""

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
        """Return per-layer handles consumed by the transformer."""
        return list(self._layers)

    def set_write_col(self, col: int) -> None:
        """Set the shared write column and enable in-place writes."""
        if col < 0 or col > self.max_len:
            raise ValueError(f"write column must be in [0, {self.max_len}], got {col}.")
        self.write_col = col
        self.frozen = False

    def freeze(self) -> None:
        """Freeze the prefix cache before appending action-expert K/V."""
        self.frozen = True


def left_to_right_align(
    x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-align each row so its last valid token occupies the last column."""
    batch_size, seq_len = input_mask.shape
    device = x.device

    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    # The transform pipeline produces at least one valid prefix token. Explicit
    # validation avoids silently rolling an empty row by one position.
    if not bool(input_mask.any(dim=1).all()):
        raise ValueError("each generation prefix must contain at least one valid token")
    last_valid = (input_mask.long() * arange).max(dim=1).values
    rolled = (arange + last_valid.unsqueeze(1) + 1) % seq_len
    batch_index = torch.arange(batch_size, device=device).unsqueeze(1)

    x_rolled = x[batch_index, rolled]
    mask_rolled = input_mask[batch_index, rolled]
    attn_rolled = attn_mask[
        batch_index.unsqueeze(2), rolled.unsqueeze(2), rolled.unsqueeze(1)
    ]
    return x_rolled, mask_rolled, attn_rolled
