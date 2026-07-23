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

"""State-based conversion of BEHAVIOR R1Pro joint actions to end-effector (EEF)
actions, for the ``absolute_eef`` and ``delta_eef`` control modes.

Unlike the FK-based ``convert_to_eef_delta`` path, this reads the achieved EEF
pose DIRECTLY from the recorded proprioception (the 256-dim ``observation.state``
already carries each arm's base-frame EEF pose), so no forward kinematics is
needed. The EEF poses are in the ROBOT BASE frame -- the exact frame OmniGibson's
``InverseKinematicsController`` controls in.

Per arm the EEF slice is 6-dim ``[x, y, z, ax, ay, az]`` (position + axis-angle
orientation). The two action modes differ only in what the arm slice encodes:

  absolute_eef  (controller: absolute_pose, re-anchors every step):
      arm = [ p_{t+1},                 quat2axisangle(q_{t+1}) ]
  delta_eef     (controller: pose_delta_ori, state-to-state achieved motion):
      arm = [ p_{t+1} - p_t,           relative_rotation_axisangle(q_t, q_{t+1}) ]

where (p_t, q_t) is the arm's achieved EEF pose at step t (from the state) and
(p_{t+1}, q_{t+1}) is the achieved EEF pose at the NEXT step. Using the achieved
NEXT state (rather than FK of the commanded joint action) avoids the
"command not reached" mismatch of the old FK(action)-FK(state) delta, and makes
the stored action exactly the motion the robot actually executed. base(3,
velocity), trunk(4, abs joint) and the two grippers(1) are copied from the
recorded 23-dim action unchanged, giving a 21-dim EEF action.

Pure numpy (reuses the vendored xyzw rotation math from convert_to_eef_delta);
importing this module never pulls in torch / OmniGibson / Isaac.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as _R

# EEF pose slices inside the 256-dim proprio observation.state (base frame).
# Verified to match the analytical URDF FK (0.0 mm / 0.0 deg) on real frames.
EEF_STATE = {
    "left_pos": slice(186, 189),
    "left_quat": slice(189, 193),  # xyzw
    "right_pos": slice(225, 228),
    "right_quat": slice(228, 232),  # xyzw
}

# Recorded 23-dim action slices (native OmniGibson controller order); the fixed
# base/trunk/gripper slices are copied through to the 21-dim EEF action.
_ACT = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "gripper_left": 14,
    "gripper_right": 22,
}

EEF_ACTION_DIM = 21
OLD_ACTION_DIM = 23


def quat2axisangle(quat_xyzw: np.ndarray) -> np.ndarray:
    """(...,4) xyzw quaternion -> (...,3) axis-angle, EXACTLY as OmniGibson does it
    (``transform_utils.quat2axisangle`` = ``scipy Rotation.from_quat(q).as_rotvec()``).

    Using scipy (not the FK module's mat2axisangle) is deliberate: mat2axisangle is
    inaccurate at the +/-pi singularity, which the R1Pro downward-gripper
    orientation sits on -- scipy is robust there, and it is the exact inverse of the
    controller's ``axisangle2quat`` so the absolute EEF orientation round-trips."""
    return _R.from_quat(np.asarray(quat_xyzw, dtype=np.float64)).as_rotvec()


def relative_rotation_axisangle(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray) -> np.ndarray:
    """Base-frame delta axis-angle ``R_delta = R1 @ R0^-1`` (the value the IK
    ``pose_delta_ori`` controller inverts: ``R1 = R_delta @ R0``), via scipy so it
    matches OmniGibson and is robust at the singularity."""
    r0 = _R.from_quat(np.asarray(q0_xyzw, dtype=np.float64))
    r1 = _R.from_quat(np.asarray(q1_xyzw, dtype=np.float64))
    return (r1 * r0.inv()).as_rotvec()


def read_eef(state256: np.ndarray, arm: str):
    """Return ``(pos[...,3], quat_xyzw[...,4])`` for ``arm`` from the proprio state."""
    s = np.asarray(state256, dtype=np.float64)
    return s[..., EEF_STATE[f"{arm}_pos"]], s[..., EEF_STATE[f"{arm}_quat"]]


def state_to_abs_eef(state256: np.ndarray) -> np.ndarray:
    """The abs_eef STATE token: per arm ``[p_t, axisangle(q_t)]`` (base frame).

    Returns a 12-dim ``[left_pos(3), left_ax(3), right_pos(3), right_ax(3)]`` block
    (the caller assembles it with base/trunk/grippers into the 21-dim state).
    """
    out = np.empty(state256.shape[:-1] + (12,), dtype=np.float64)
    for lo, arm in ((0, "left"), (6, "right")):
        p, q = read_eef(state256, arm)
        out[..., lo : lo + 3] = p
        out[..., lo + 3 : lo + 6] = quat2axisangle(q)
    return out


def _assemble(action23: np.ndarray, left6: np.ndarray, right6: np.ndarray) -> np.ndarray:
    """base(3)+trunk(4)+left_eef(6)+gripL(1)+right_eef(6)+gripR(1) = 21."""
    a = np.asarray(action23, dtype=np.float64)
    out = np.empty(EEF_ACTION_DIM, dtype=np.float64)
    out[0:3] = a[_ACT["base"]]
    out[3:7] = a[_ACT["trunk"]]
    out[7:13] = left6
    out[13] = a[_ACT["gripper_left"]]
    out[14:20] = right6
    out[20] = a[_ACT["gripper_right"]]
    return out


def action_to_absolute_eef(
    action23: np.ndarray, state_t: np.ndarray, state_t1: np.ndarray
) -> np.ndarray:
    """21-dim absolute_eef action: arm = [p_{t+1}, axisangle(q_{t+1})] (next state)."""
    six = []
    for arm in ("left", "right"):
        p1, q1 = read_eef(state_t1, arm)
        six.append(np.concatenate([p1, quat2axisangle(q1)]))
    return _assemble(action23, six[0], six[1])


def action_to_delta_eef(
    action23: np.ndarray, state_t: np.ndarray, state_t1: np.ndarray
) -> np.ndarray:
    """21-dim delta_eef action: arm = [p_{t+1}-p_t, relative_rotation_axisangle(q_t,q_{t+1})]."""
    six = []
    for arm in ("left", "right"):
        p0, q0 = read_eef(state_t, arm)
        p1, q1 = read_eef(state_t1, arm)
        six.append(np.concatenate([p1 - p0, relative_rotation_axisangle(q0, q1)]))
    return _assemble(action23, six[0], six[1])


def convert_episode_actions(
    action23: np.ndarray, state256: np.ndarray, mode: str
) -> np.ndarray:
    """Convert a whole episode (F,23)+(F,256) -> (F,21) for ``absolute_eef`` /
    ``delta_eef``. The last frame has no t+1, so it holds (target = current)."""
    a = np.asarray(action23, dtype=np.float64).reshape(-1, OLD_ACTION_DIM)
    s = np.asarray(state256, dtype=np.float64).reshape(-1, 256)
    f = a.shape[0]
    fn = {"absolute_eef": action_to_absolute_eef, "delta_eef": action_to_delta_eef}[mode]
    out = np.empty((f, EEF_ACTION_DIM), dtype=np.float64)
    for t in range(f):
        t1 = min(t + 1, f - 1)
        out[t] = fn(a[t], s[t], s[t1])
    return out
