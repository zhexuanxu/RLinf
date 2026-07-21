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

"""Offline conversion of the BEHAVIOR R1Pro SFT dataset from absolute-joint
actions to delta end-effector (delta-EEF) actions.

The original 23-dim action (base[3] velocity, trunk[4] abs joint, arm_left[7]
abs joint, gripper_left[1], arm_right[7] abs joint, gripper_right[1]) is
rewritten so each arm becomes a 6-DoF base-frame EEF delta command matching
OmniGibson's ``InverseKinematicsController`` in ``pose_delta_ori`` mode, giving
a 21-dim action: base[3], trunk[4], arm_left_eef[6], gripper_left[1],
arm_right_eef[6], gripper_right[1].

Per-arm delta (command-relative-to-achieved, DEC-3):

    eef_target  = FK_base(trunk_target,   arm_target)     # from the recorded action
    eef_current = FK_base(trunk_achieved, arm_achieved)   # from the recorded proprio state
    dpos = target_pos - current_pos                                   # metres, base frame
    dori = quat2axisangle(mat2quat(R_target @ R_current.T))           # exact relative rotation

``dori`` is the EXACT relative-rotation axis-angle the controller inverts
(``R_target = R_delta @ R_current``, a base-frame left-multiply), NOT the
first-order ``orientation_error`` vector. Command limits are ``null`` at eval,
so the stored deltas are raw metres/radians (no inverse scaling).

FK uses a Lula ``FKSolver`` over the R1Pro URDF. Because the URDF exposes
``{arm}_gripper_link`` but not the USD ``{arm}_eef_link``, we FK the gripper
link and compose the fixed gripper->eef offset. The FK is VALIDATED against a
live OmniGibson robot's ``get_relative_eef_pose`` on real recorded qpos before
any bulk conversion (guards joint-order / link-name / offset mistakes).

This module contains only pure-Python / numpy logic plus lazy OmniGibson (Isaac)
imports inside the functions that need them, so importing it never pulls in
Isaac. Run it inside the BEHAVIOR venv (``/mnt/public/xzxuan/.venv_pi``).
"""

from __future__ import annotations

import dataclasses

import numpy as np

# R1Pro proprio (256-dim observation.state) slices for achieved qpos. Mirrors
# R1PRO_PROPRIO_INDICES in behavior_policy but kept local so this converter does
# not import the model package.
_PROPRIO = {
    "arm_left_qpos": slice(158, 165),  # 7, joint1..7 order
    "arm_right_qpos": slice(197, 204),  # 7
    "trunk_qpos": slice(236, 240),  # 4, torso_joint1..4
}

# Recorded 23-dim action slices (native OmniGibson controller order).
_ACT = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "arm_left": slice(7, 14),
    "gripper_left": 14,
    "arm_right": slice(15, 22),
    "gripper_right": 22,
}

# Lula descriptor cspace order (the order joint values must be stacked before
# FKSolver.get_link_poses). Trunk + both arms; the steer/wheel/gripper-finger
# joints do not affect either arm's base-frame EEF pose and are held at the
# descriptor's default_q.
CSPACE_JOINTS = (
    ["torso_joint1", "torso_joint2", "torso_joint3", "torso_joint4"]
    + [f"left_arm_joint{i}" for i in range(1, 8)]
    + [f"right_arm_joint{i}" for i in range(1, 8)]
)

GRIPPER_LINK = {"left": "left_gripper_link", "right": "right_gripper_link"}

# Fixed gripper_link -> eef_link offset (from r1pro_source_cfg.yaml): a -0.06 m
# translation along gripper z and a 180-degree rotation about y (quat xyzw).
EEF_OFFSET_POS = np.array([0.0, 0.0, -0.06], dtype=np.float64)
EEF_OFFSET_QUAT_XYZW = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

NEW_ACTION_DIM = 21
OLD_ACTION_DIM = 23


# --------------------------------------------------------------------------- #
# Minimal quaternion math (xyzw), matching OmniGibson's transform_utils
# conventions, so the converter can run without importing OmniGibson for the
# pure-numpy path (the live-robot validation still imports OmniGibson).
# --------------------------------------------------------------------------- #
def quat2mat_xyzw(q: np.ndarray) -> np.ndarray:
    """(...,4) xyzw unit quaternion -> (...,3,3) rotation matrix."""
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (yy + zz)
    m[..., 0, 1] = 2 * (xy - wz)
    m[..., 0, 2] = 2 * (xz + wy)
    m[..., 1, 0] = 2 * (xy + wz)
    m[..., 1, 1] = 1 - 2 * (xx + zz)
    m[..., 1, 2] = 2 * (yz - wx)
    m[..., 2, 0] = 2 * (xz - wy)
    m[..., 2, 1] = 2 * (yz + wx)
    m[..., 2, 2] = 1 - 2 * (xx + yy)
    return m


def mat2axisangle(r: np.ndarray) -> np.ndarray:
    """(...,3,3) rotation matrix -> (...,3) axis-angle (rotation vector)."""
    # angle from trace, axis from skew part; numerically stable near 0 and pi.
    cos = (np.trace(r, axis1=-2, axis2=-1) - 1.0) / 2.0
    cos = np.clip(cos, -1.0, 1.0)
    angle = np.arccos(cos)
    axis = np.stack(
        [
            r[..., 2, 1] - r[..., 1, 2],
            r[..., 0, 2] - r[..., 2, 0],
            r[..., 1, 0] - r[..., 0, 1],
        ],
        axis=-1,
    )
    small = angle < 1e-6
    sin = np.sin(angle)
    # default (generic) branch
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(np.abs(sin) < 1e-8, 0.0, angle / (2.0 * sin))
    out = axis * scale[..., None]
    # small-angle: axis*angle ~ 0.5 * skew vector
    out = np.where(small[..., None], 0.5 * axis, out)
    return out.astype(np.float64)


def relative_rotation_axisangle(
    q_cur_xyzw: np.ndarray, q_tgt_xyzw: np.ndarray
) -> np.ndarray:
    """Exact base-frame delta axis-angle: R_delta = R_tgt @ R_cur.T.

    This is the value the IK ``pose_delta_ori`` controller inverts
    (``R_tgt = R_delta @ R_cur``), NOT the first-order orientation_error.
    """
    r_cur = quat2mat_xyzw(q_cur_xyzw)
    r_tgt = quat2mat_xyzw(q_tgt_xyzw)
    r_delta = r_tgt @ np.swapaxes(r_cur, -1, -2)
    return mat2axisangle(r_delta)


@dataclasses.dataclass
class ArmEefPose:
    pos: np.ndarray  # (3,)
    quat_xyzw: np.ndarray  # (4,)


def compose_pose(
    p_parent: np.ndarray,
    q_parent_xyzw: np.ndarray,
    p_child: np.ndarray,
    q_child_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose child offset expressed in parent frame: world = parent (X) child."""
    r_parent = quat2mat_xyzw(q_parent_xyzw)
    p = p_parent + r_parent @ p_child
    r = r_parent @ quat2mat_xyzw(q_child_xyzw)
    # matrix -> quat (xyzw)
    q = _mat2quat_xyzw(r)
    return p, q


def _mat2quat_xyzw(r: np.ndarray) -> np.ndarray:
    m00, m11, m22 = r[0, 0], r[1, 1], r[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif m00 > m11 and m00 > m22:
        s = np.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif m11 > m22:
        s = np.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def cspace_vector(trunk_qpos: np.ndarray, arm_left: np.ndarray, arm_right: np.ndarray):
    """Stack joint values in CSPACE_JOINTS order (trunk, left arm, right arm)."""
    return np.concatenate(
        [
            np.asarray(trunk_qpos, dtype=np.float64).reshape(4),
            np.asarray(arm_left, dtype=np.float64).reshape(7),
            np.asarray(arm_right, dtype=np.float64).reshape(7),
        ]
    )


def action_joint_to_eef_delta(
    action23: np.ndarray,
    state256: np.ndarray,
    fk_eval_fn,
) -> np.ndarray:
    """Convert one frame's 23-dim joint action to the 21-dim delta-EEF action.

    Args:
        action23: recorded 23-dim action (native controller order).
        state256: recorded 256-dim proprio observation.state.
        fk_eval_fn: callable(cspace_q_2x?) -> per-arm base-frame EEF (pos, quat).
            Concretely ``fk_eval_fn(cspace_q)`` returns
            ``{"left": ArmEefPose, "right": ArmEefPose}`` for a single cspace
            vector. Called twice (target from action, current from state).

    Returns:
        21-dim delta-EEF action (float64): base[3], trunk[4], left_eef[6],
        gripper_left[1], right_eef[6], gripper_right[1].
    """
    action23 = np.asarray(action23, dtype=np.float64).reshape(OLD_ACTION_DIM)
    state256 = np.asarray(state256, dtype=np.float64).reshape(256)

    trunk_target = action23[_ACT["trunk"]]
    arm_left_target = action23[_ACT["arm_left"]]
    arm_right_target = action23[_ACT["arm_right"]]
    trunk_achieved = state256[_PROPRIO["trunk_qpos"]]
    arm_left_achieved = state256[_PROPRIO["arm_left_qpos"]]
    arm_right_achieved = state256[_PROPRIO["arm_right_qpos"]]

    eef_target = fk_eval_fn(
        cspace_vector(trunk_target, arm_left_target, arm_right_target)
    )
    eef_current = fk_eval_fn(
        cspace_vector(trunk_achieved, arm_left_achieved, arm_right_achieved)
    )

    out = np.empty(NEW_ACTION_DIM, dtype=np.float64)
    out[0:3] = action23[_ACT["base"]]
    out[3:7] = trunk_target
    for lo, arm, gidx_new, gidx_old in (
        (7, "left", 13, _ACT["gripper_left"]),
        (14, "right", 20, _ACT["gripper_right"]),
    ):
        tgt, cur = eef_target[arm], eef_current[arm]
        out[lo : lo + 3] = tgt.pos - cur.pos
        out[lo + 3 : lo + 6] = relative_rotation_axisangle(cur.quat_xyzw, tgt.quat_xyzw)
        out[gidx_new] = action23[gidx_old]
    return out
