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

"""Lock the BEHAVIOR 23-dim state layout for both ``state_order`` values.

``extract_state_from_proprio`` supports two channel orderings, selected by
``actor.model.openpi.state_order``:

* ``"comet"`` (default) — reference / pretrained-checkpoint order, both grippers
  at the tail: ``[base, trunk, arm_left, arm_right, left_gripper, right_gripper]``
  (left gripper at index 21).
* ``"align"`` — action-aligned order, left gripper moved to index 14:
  ``[base, trunk, arm_left, left_gripper, arm_right, right_gripper]``.

The two produce different layouts, so ``norm_stats.json`` is not interchangeable
between them. This test pins both orderings (and the default) so neither can
silently change.
"""

import numpy as np
import pytest

from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    R1PRO_PROPRIO_INDICES,
    STATE_ORDERS,
    extract_state_from_proprio,
)

_PROPRIO_DIM = 256


def _sentinel_proprio() -> np.ndarray:
    """A 256-dim proprio whose value at index ``i`` is ``i`` (float)."""
    return np.arange(_PROPRIO_DIM, dtype=np.float64)


def _left_gripper_width(proprio: np.ndarray) -> float:
    return proprio[R1PRO_PROPRIO_INDICES["gripper_left_qpos"]].sum()


def _right_gripper_width(proprio: np.ndarray) -> float:
    return proprio[R1PRO_PROPRIO_INDICES["gripper_right_qpos"]].sum()


def test_supported_orders():
    assert STATE_ORDERS == ("comet", "align")


def test_output_is_23_dim():
    for order in STATE_ORDERS:
        state = extract_state_from_proprio(_sentinel_proprio(), order)
        assert state.shape == (23,), order


def test_default_is_comet():
    """Calling without state_order must reproduce the comet (reference) order."""
    proprio = _sentinel_proprio()
    np.testing.assert_array_equal(
        extract_state_from_proprio(proprio),
        extract_state_from_proprio(proprio, "comet"),
    )


def test_shared_head_is_order_independent():
    """base/trunk/arm_left (state[0:14]) are identical in both orders."""
    proprio = _sentinel_proprio()
    comet = extract_state_from_proprio(proprio, "comet")
    align = extract_state_from_proprio(proprio, "align")

    np.testing.assert_array_equal(
        comet[0:3], proprio[R1PRO_PROPRIO_INDICES["base_qvel"]]
    )
    np.testing.assert_array_equal(
        comet[3:7], proprio[R1PRO_PROPRIO_INDICES["trunk_qpos"]]
    )
    np.testing.assert_array_equal(
        comet[7:14], proprio[R1PRO_PROPRIO_INDICES["arm_left_qpos"]]
    )
    np.testing.assert_array_equal(comet[0:14], align[0:14])


def test_comet_order_grippers_at_tail():
    """comet: arm_right at 14:21, left gripper at 21, right gripper at 22."""
    proprio = _sentinel_proprio()
    state = extract_state_from_proprio(proprio, "comet")

    np.testing.assert_array_equal(
        state[14:21], proprio[R1PRO_PROPRIO_INDICES["arm_right_qpos"]]
    )
    assert state[21] == _left_gripper_width(proprio)
    assert state[22] == _right_gripper_width(proprio)


def test_align_order_left_gripper_at_14():
    """align: left gripper at 14, arm_right at 15:22, right gripper at 22."""
    proprio = _sentinel_proprio()
    state = extract_state_from_proprio(proprio, "align")

    assert state[14] == _left_gripper_width(proprio)
    np.testing.assert_array_equal(
        state[15:22], proprio[R1PRO_PROPRIO_INDICES["arm_right_qpos"]]
    )
    assert state[22] == _right_gripper_width(proprio)


def test_orders_differ_only_in_gripper_arm_block():
    """The two orders are a permutation of the same 23 values (idx 14:22)."""
    proprio = _sentinel_proprio()
    comet = extract_state_from_proprio(proprio, "comet")
    align = extract_state_from_proprio(proprio, "align")

    assert not np.array_equal(comet[14:22], align[14:22])
    np.testing.assert_array_equal(np.sort(comet), np.sort(align))


def test_invalid_order_raises():
    with pytest.raises(ValueError, match="state_order"):
        extract_state_from_proprio(_sentinel_proprio(), "bogus")


def test_batched_input_preserved():
    """The layout holds along the last axis for a batched (T, 256) input."""
    batch = np.stack([_sentinel_proprio(), _sentinel_proprio() + 1000.0])
    for order in STATE_ORDERS:
        state = extract_state_from_proprio(batch, order)
        assert state.shape == (2, 23), order
        np.testing.assert_array_equal(
            state[0], extract_state_from_proprio(batch[0], order)
        )
        np.testing.assert_array_equal(
            state[1], extract_state_from_proprio(batch[1], order)
        )
