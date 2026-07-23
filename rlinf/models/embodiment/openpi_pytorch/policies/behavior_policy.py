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
    # Base-frame end-effector pose per arm (pos xyz + quat xyzw). Verified to
    # match the analytical URDF FK / get_relative_eef_pose (0.0mm/0.0deg), i.e.
    # the same base frame OmniGibson's InverseKinematicsController controls in.
    "eef_left_pos": np.s_[186:189],
    "eef_left_quat": np.s_[189:193],
    "eef_right_pos": np.s_[225:228],
    "eef_right_quat": np.s_[228:232],
}


class DataTransformFn:
    """Minimal callable-transform base (replaces openpi.transforms.DataTransformFn)."""

    def __call__(self, data: dict) -> dict:  # pragma: no cover - interface only
        raise NotImplementedError


# Prompt state channel layout (actor.model.openpi.state_token). Canonical values
# and their back-compat aliases (the old state_order key/values still resolve):
#   "abs_joint_old" (== legacy "comet"): both grippers at the tail (23-dim).
#   "abs_joint"     (== legacy "align"): action-aligned, left gripper at 14 (23-dim).
#   "abs_eef":       arms as base-frame EEF pose [x,y,z, ax,ay,az] (axis-angle),
#                    consistent with the eef action space (21-dim before padding).
STATE_TOKENS = ("abs_joint_old", "abs_joint", "abs_eef")
_STATE_TOKEN_ALIASES = {"comet": "abs_joint_old", "align": "abs_joint"}
# Legacy name kept for back-compat imports.
STATE_ORDERS = ("comet", "align")


def resolve_state_token(value: str) -> str:
    """Normalize a state_token / legacy state_order value to a canonical token."""
    v = _STATE_TOKEN_ALIASES.get(value, value)
    if v not in STATE_TOKENS:
        raise ValueError(
            f"state_token must be one of {STATE_TOKENS} (or legacy "
            f"{tuple(_STATE_TOKEN_ALIASES)}), got {value!r}."
        )
    return v


# Robot action space selection (actor.model.openpi.control_mode). All modes keep
# base(3, velocity) + trunk(4, abs joint) + grippers(1 each, smooth); they differ
# only in the two arm slices:
#   "joint_absolute": each arm is 7 absolute joint-position targets -> 23-dim action
#   "absolute_eef":   each arm is a 6-DoF base-frame ABSOLUTE EEF pose (pos+axisangle)
#                     -> 21-dim action (OmniGibson IK absolute_pose; re-anchors).
#   "delta_eef":      each arm is a 6-DoF base-frame EEF delta (state-to-state
#                     [dpos, relrot axisangle]) -> 21-dim (OmniGibson IK pose_delta_ori).
#   "eef_delta_pose": LEGACY FK-based delta (the original all-50 artifact); kept
#                     working -> 21-dim (IK pose_delta_ori). Superseded by delta_eef.
CONTROL_MODES = ("joint_absolute", "absolute_eef", "delta_eef", "eef_delta_pose")

# Semantic env action dimension per control mode (BEFORE padding to the model's
# model_action_dim). Fixed slices base(3)+trunk(4)+gripper_left(1)+gripper_right(1)
# = 9; each arm adds 7 (absolute joint) or 6 (any EEF mode).
CONTROL_MODE_ACTION_ENV_DIM = {
    "joint_absolute": 23,  # 9 + 7 + 7
    "absolute_eef": 21,  # 9 + 6 + 6
    "delta_eef": 21,  # 9 + 6 + 6
    "eef_delta_pose": 21,  # 9 + 6 + 6 (legacy)
}
# EEF-based control modes (arms are 6-DoF EEF, not joint).
EEF_CONTROL_MODES = ("absolute_eef", "delta_eef", "eef_delta_pose")



def extract_state_from_proprio(
    proprio_data: np.ndarray, state_token: str = "abs_joint_old"
) -> np.ndarray:
    """Extract the policy state from the full R1Pro proprio vector.

    ``state_token`` selects the channel layout (legacy ``state_order`` values
    ``comet``/``align`` still resolve, to ``abs_joint_old``/``abs_joint``):

    ``"abs_joint_old"`` (legacy ``comet``; reference / pretrained-checkpoint order,
    both grippers at the tail)::

        state[0:3]   base_qvel
        state[3:7]   trunk_qpos
        state[7:14]  arm_left_qpos
        state[14:21] arm_right_qpos
        state[21]    left_gripper_width
        state[22]    right_gripper_width

    ``"abs_joint"`` (legacy ``align``; action-aligned order, left gripper at 14 so
    the state lines up channel-for-channel with the R1Pro joint action)::

        state[0:3]   base_qvel            == action base (vel x, y, yaw)
        state[3:7]   trunk_qpos           == action trunk
        state[7:14]  arm_left_qpos        == action arm_left
        state[14]    left_gripper_width   == action gripper_left
        state[15:22] arm_right_qpos       == action arm_right
        state[22]    right_gripper_width  == action gripper_right

    ``"abs_eef"`` (EEF state, consistent with the eef action space): each arm is
    the base-frame absolute EEF pose ``[x,y,z, ax,ay,az]`` (pos + axis-angle)::

        state[0:3]   base_qvel
        state[3:7]   trunk_qpos
        state[7:13]  arm_left_eef  = [pos(3), quat2axisangle(quat)(3)]
        state[13]    left_gripper_width
        state[14:20] arm_right_eef = [pos(3), quat2axisangle(quat)(3)]
        state[20]    right_gripper_width

    Each gripper's two finger joints are collapsed to one width via
    ``.sum(axis=-1)``. ``norm_stats.json`` must be regenerated with the matching
    ``state_token`` — the layouts are not interchangeable.
    """
    token = resolve_state_token(state_token)
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
    if token == "abs_eef":
        # Arms as base-frame absolute EEF pose (pos + axis-angle), read straight
        # from the proprio state (no FK). Uses the SAME quat->axisangle as the eef
        # action conversion so state and action are consistent.
        from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef import (
            quat2axisangle,
        )

        def arm_eef(pos_key, quat_key):
            pos = proprio_data[..., R1PRO_PROPRIO_INDICES[pos_key]]
            ax = quat2axisangle(proprio_data[..., R1PRO_PROPRIO_INDICES[quat_key]])
            return np.concatenate([pos, ax], axis=-1)  # 6

        return np.concatenate(
            [
                base_qvel,  # 3
                trunk_qpos,  # 4
                arm_eef("eef_left_pos", "eef_left_quat"),  # 6 -> state[7:13]
                left_gripper_width,  # 1 -> state[13]
                arm_eef("eef_right_pos", "eef_right_quat"),  # 6 -> state[14:20]
                right_gripper_width,  # 1 -> state[20]
            ],
            axis=-1,
        )
    if token == "abs_joint_old":
        # Reference order: both grippers at the tail (left gripper at index 21).
        return np.concatenate(
            [
                base_qvel,
                trunk_qpos,
                arm_left_qpos,
                arm_right_qpos,
                left_gripper_width,
                right_gripper_width,
            ],
            axis=-1,
        )
    # "abs_joint": left gripper moved to index 14 to match the joint action space.
    return np.concatenate(
        [
            base_qvel,  # 3  -> state[0:3]   == action base
            trunk_qpos,  # 4  -> state[3:7]   == action trunk
            arm_left_qpos,  # 7  -> state[7:14]  == action arm_left
            left_gripper_width,  # 1  -> state[14] == action gripper_left
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
