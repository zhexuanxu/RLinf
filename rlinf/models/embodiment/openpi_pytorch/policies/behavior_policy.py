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

"""Self-contained BEHAVIOR input/output transforms (pi05 eval).

Vendored from ``rlinf/models/embodiment/openpi/policies/behavior_policy.py`` with
the installed-``openpi`` dependencies removed: the upstream ``transforms.DataTransformFn``
base and ``openpi.models.model.ModelType`` are replaced by a local lightweight
callable base, and the BEHAVIOR pi05 branch is kept (image key mapping and the
23-dim action slice).

``extract_state_from_proprio`` supports two channel orderings via its
``state_order`` argument (threaded from ``actor.model.openpi.state_order`` in
YAML): ``"comet"`` reproduces the reference repo (``openpi-comet``
``b1k_policy.py``) / pretrained-checkpoint order with both grippers at the tail,
while ``"align"`` reorders the left gripper to index 14 so the 23-dim state lines
up channel-for-channel with the R1Pro action space. The two orderings produce
different layouts, so norm stats are not interchangeable between them.
"""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

# Keep a local copy of the R1Pro proprio slices to avoid importing omnigibson
# in rollout worker init threads (omnigibson registers signal handlers at import time).
R1PRO_PROPRIO_INDICES = {
    "arm_left_qpos": np.s_[158:165],
    "gripper_left_qpos": np.s_[193:195],
    "arm_right_qpos": np.s_[197:204],
    "trunk_qpos": np.s_[236:240],
    "base_qvel": np.s_[253:256],
    "gripper_right_qpos": np.s_[232:234],
}


class DataTransformFn:
    """Minimal callable-transform base (replaces openpi.transforms.DataTransformFn)."""

    def __call__(self, data: dict) -> dict:  # pragma: no cover - interface only
        raise NotImplementedError


STATE_ORDERS = ("comet", "align")


def extract_state_from_proprio(
    proprio_data: np.ndarray, state_order: str = "comet"
) -> np.ndarray:
    """Extract the 23-dim policy state from the full R1Pro proprio vector.

    Two channel orderings are supported via ``state_order``:

    ``"comet"`` (reference / pretrained-checkpoint order; both grippers at the
    tail, matching ``openpi-comet`` ``b1k_policy.py`` and the legacy JAX twin)::

        state[0:3]   base_qvel
        state[3:7]   trunk_qpos
        state[7:14]  arm_left_qpos
        state[14:21] arm_right_qpos
        state[21]    left_gripper_width
        state[22]    right_gripper_width

    ``"align"`` (action-aligned order; left gripper moved to index 14 so the
    state lines up channel-for-channel with the R1Pro action space —
    OmniGibson ``_raw_controller_order``: base, trunk, arm_left, gripper_left,
    arm_right, gripper_right)::

        state[0:3]   base_qvel            == action base (vel x, y, yaw)
        state[3:7]   trunk_qpos           == action trunk
        state[7:14]  arm_left_qpos        == action arm_left
        state[14]    left_gripper_width   == action gripper_left
        state[15:22] arm_right_qpos       == action arm_right
        state[22]    right_gripper_width  == action gripper_right

    Each gripper's two finger joints are collapsed to a single width via
    ``.sum(axis=-1)`` (the action space uses a 1-dim smooth gripper command).

    The two orderings produce different 23-dim layouts, so ``norm_stats.json``
    must be regenerated with the matching ``state_order`` — stats are not
    interchangeable between ``"comet"`` and ``"align"``.
    """
    if state_order not in STATE_ORDERS:
        raise ValueError(
            f"state_order must be one of {STATE_ORDERS}, got {state_order!r}."
        )
    base_qvel = proprio_data[..., R1PRO_PROPRIO_INDICES["base_qvel"]]  # 3
    trunk_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["trunk_qpos"]]  # 4
    arm_left_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["arm_left_qpos"]]  # 7
    arm_right_qpos = proprio_data[..., R1PRO_PROPRIO_INDICES["arm_right_qpos"]]  # 7
    left_gripper_width = proprio_data[
        ..., R1PRO_PROPRIO_INDICES["gripper_left_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    right_gripper_width = proprio_data[
        ..., R1PRO_PROPRIO_INDICES["gripper_right_qpos"]
    ].sum(axis=-1, keepdims=True)  # 1
    if state_order == "comet":
        # Reference order: both grippers at the tail (left gripper at index 21).
        return np.concatenate(
            [
                base_qvel,
                trunk_qpos,
                arm_left_qpos,
                arm_right_qpos,
                left_gripper_width,  # gripper rearranged from 21 to 14 to match the action space
                right_gripper_width,
            ],
            axis=-1,
        )
    # "align": left gripper moved to index 14 to match the action space.
    return np.concatenate(
        [
            base_qvel,  # 3  -> state[0:3]   == action base
            trunk_qpos,  # 4  -> state[3:7]   == action trunk
            arm_left_qpos,  # 7  -> state[7:14]  == action arm_left
            left_gripper_width,  # 1  -> state[14] == action gripper_left (rearranged from 21 to 14 to match the action space)
            arm_right_qpos,  # 7  -> state[15:22] == action arm_right
            right_gripper_width,  # 1  -> state[22]    == action gripper_right
        ],
        axis=-1,
    )


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[0] == 2 and image.shape[1] == 3:
        image = einops.rearrange(image, "n c h w -> n h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BehaviorInputs(DataTransformFn):
    """Map a BEHAVIOR observation dict to the model input dict (pi05 branch)."""

    extract_state_from_proprio: bool = True
    use_all_wrist_images: bool = True
    state_order: str = "comet"

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])  # [h, w, c]

        # Handle both stacked wrist images and separate keys (BEHAVIOR v2.1).
        if "observation/wrist_image" in data:
            wrist_image = _parse_image(data["observation/wrist_image"])  # [2, h, w, c]
            left_wrist = wrist_image[0, ...]
            right_wrist = wrist_image[1, ...]
        else:
            left_wrist = _parse_image(data["observation/left_wrist_image"])
            right_wrist = _parse_image(data["observation/right_wrist_image"])

        state = (
            extract_state_from_proprio(data["observation/state"], self.state_order)
            if self.extract_state_from_proprio
            else data["observation/state"]
        )

        inputs = {
            "state": state[:32],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_
                if self.use_all_wrist_images
                else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(DataTransformFn):
    """Slice model actions back to the BEHAVIOR env action dimension."""

    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
