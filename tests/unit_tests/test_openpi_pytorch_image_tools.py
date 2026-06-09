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

"""Boundary-guard tests for the retained ``image_tools.resize_with_pad`` contract.

``resize_with_pad`` is a retained external-input boundary guard (per the package
cleanup): it must validate its inputs and reject out-of-contract data with a
clear ``TypeError``/``ValueError`` rather than failing deep inside PIL.
"""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.models.embodiment.openpi_pytorch.utils.image_tools import resize_with_pad


def test_resize_with_pad_valid_input_resizes_to_target():
    img = np.zeros((2, 64, 48, 3), dtype=np.uint8)  # (batch, H, W, C)
    out = resize_with_pad(img, 224, 224)
    assert out.shape == (2, 224, 224, 3)


def test_resize_with_pad_rejects_non_array():
    with pytest.raises(TypeError, match="numpy array"):
        resize_with_pad([[0, 0, 0]], 224, 224)


def test_resize_with_pad_rejects_low_rank():
    with pytest.raises(ValueError, match="rank >= 3"):
        resize_with_pad(np.zeros((64, 48), dtype=np.uint8), 224, 224)


def test_resize_with_pad_rejects_nonpositive_target():
    with pytest.raises(ValueError, match="positive ints"):
        resize_with_pad(np.zeros((64, 48, 3), dtype=np.uint8), 0, 224)


def test_resize_with_pad_rejects_bad_channel_count():
    with pytest.raises(ValueError, match="1/3/4 image channels"):
        resize_with_pad(np.zeros((64, 48, 5), dtype=np.uint8), 224, 224)
