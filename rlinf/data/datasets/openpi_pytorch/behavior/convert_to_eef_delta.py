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
        fk_eval_fn: callable(cspace_q) -> {"left": ArmEefPose, "right": ArmEefPose}
            for a single cspace vector (base-frame EEF). Called twice (target
            from action, current from state).

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


# --------------------------------------------------------------------------- #
# Analytical URDF forward kinematics (pure numpy, batched, Isaac-free).
#
# The R1Pro arm chains base_link -> torso -> arm -> gripper_link have only
# translational joint origins (all rpy == 0) plus single-axis revolute joints,
# so FK is a product of translate(xyz) @ axis_rotation(axis, q) transforms. This
# is validated once against a live OmniGibson robot's get_relative_eef_pose
# (see validate_fk_against_live_robot) before any bulk conversion.
# --------------------------------------------------------------------------- #
def load_arm_chains(urdf_path: str) -> dict:
    """Parse base_link -> {arm}_gripper_link chains from the R1Pro URDF.

    Returns ``{"left": [joint, ...], "right": [...]}`` where each joint is
    ``{"name", "type", "xyz"(3), "rpy"(3), "axis"(3)}`` in root->tip order.
    """
    import xml.etree.ElementTree as ET

    root = ET.parse(urdf_path).getroot()
    by_child = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin is not None:
            if origin.get("xyz"):
                xyz = [float(v) for v in origin.get("xyz").split()]
            if origin.get("rpy"):
                rpy = [float(v) for v in origin.get("rpy").split()]
        axis_el = j.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz"):
            axis = [float(v) for v in axis_el.get("xyz").split()]
        by_child[j.find("child").get("link")] = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "xyz": xyz,
            "rpy": rpy,
            "axis": axis,
        }

    chains = {}
    for side in ("left", "right"):
        seq, link = [], f"{side}_gripper_link"
        while link in by_child:
            seq.append(by_child[link])
            link = by_child[link]["parent"]
        seq.reverse()
        chains[side] = seq
    return chains


def _rpy_matrix(rpy) -> np.ndarray:
    rx, ry, rz = rpy
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    rxm = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rym = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rzm = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rzm @ rym @ rxm


def _axis_rotation(axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotation about a fixed unit axis for a batch of angles.

    axis: (3,); theta: (N,) -> (N,3,3).
    """
    a = np.asarray(axis, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.broadcast_to(np.eye(3), theta.shape + (3, 3)).copy()
    a = a / n
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    theta = np.asarray(theta, dtype=np.float64)
    s = np.sin(theta)[..., None, None]
    c = np.cos(theta)[..., None, None]
    eye = np.eye(3)
    return eye + s * k + (1.0 - c) * (k @ k)


class UrdfArmFK:
    """Batched analytical FK for the two R1Pro arm chains, base-frame EEF.

    Call ``fk(cspace_q)`` with cspace_q shaped (N, 15) or (15,) in CSPACE_JOINTS
    order; returns ``{"left": (pos[N,3], quat[N,4]), "right": ...}`` after
    composing the fixed gripper_link->eef_link offset.
    """

    def __init__(self, urdf_path: str):
        self.chains = load_arm_chains(urdf_path)
        self._cspace_index = {name: i for i, name in enumerate(CSPACE_JOINTS)}
        # Precompute fixed local transforms and the moving-joint metadata.
        self._plan = {}
        for side, seq in self.chains.items():
            steps = []
            for jd in seq:
                t = np.eye(4)
                t[:3, :3] = _rpy_matrix(jd["rpy"])
                t[:3, 3] = jd["xyz"]
                moving = jd["type"] in ("revolute", "continuous", "prismatic")
                steps.append(
                    {
                        "T_origin": t,
                        "moving": moving,
                        "prismatic": jd["type"] == "prismatic",
                        "axis": np.asarray(jd["axis"], dtype=np.float64),
                        "cspace_idx": self._cspace_index.get(jd["name"]),
                    }
                )
            self._plan[side] = steps

    def __call__(self, cspace_q: np.ndarray) -> dict:
        q = np.asarray(cspace_q, dtype=np.float64)
        squeeze = q.ndim == 1
        if squeeze:
            q = q[None, :]
        n = q.shape[0]
        out = {}
        for side, steps in self._plan.items():
            acc = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
            for st in steps:
                local = np.broadcast_to(st["T_origin"], (n, 4, 4)).copy()
                if st["moving"] and st["cspace_idx"] is not None:
                    ang = q[:, st["cspace_idx"]]
                    if st["prismatic"]:
                        # translate along axis by q (not used by arm chains)
                        local[:, :3, 3] += ang[:, None] * st["axis"]
                    else:
                        rot = _axis_rotation(st["axis"], ang)  # (n,3,3)
                        local[:, :3, :3] = local[:, :3, :3] @ rot
                acc = acc @ local
            # apply fixed gripper_link -> eef_link offset
            p_g = acc[:, :3, 3]
            r_g = acc[:, :3, :3]
            r_off = quat2mat_xyzw(EEF_OFFSET_QUAT_XYZW)
            p_eef = p_g + (r_g @ EEF_OFFSET_POS)
            r_eef = r_g @ r_off
            quats = np.stack([_mat2quat_xyzw(r_eef[i]) for i in range(n)], axis=0)
            if squeeze:
                out[side] = ArmEefPose(pos=p_eef[0], quat_xyzw=quats[0])
            else:
                out[side] = (p_eef, quats)
        return out


# --------------------------------------------------------------------------- #
# Batched episode conversion + per-episode action statistics.
# --------------------------------------------------------------------------- #
def convert_episode_actions(
    action23: np.ndarray, state256: np.ndarray, fk: "UrdfArmFK"
) -> np.ndarray:
    """Convert a whole episode's actions (F,23) + states (F,256) -> (F,21).

    Batched: two FK passes (target from action, current from state) per episode.
    """
    action23 = np.asarray(action23, dtype=np.float64).reshape(-1, OLD_ACTION_DIM)
    state256 = np.asarray(state256, dtype=np.float64).reshape(-1, 256)
    f = action23.shape[0]

    # cspace vectors (F,15): trunk + left arm + right arm.
    q_target = np.concatenate(
        [action23[:, _ACT["trunk"]], action23[:, _ACT["arm_left"]], action23[:, _ACT["arm_right"]]],
        axis=1,
    )
    q_current = np.concatenate(
        [
            state256[:, _PROPRIO["trunk_qpos"]],
            state256[:, _PROPRIO["arm_left_qpos"]],
            state256[:, _PROPRIO["arm_right_qpos"]],
        ],
        axis=1,
    )
    eef_t = fk(q_target)
    eef_c = fk(q_current)

    out = np.empty((f, NEW_ACTION_DIM), dtype=np.float64)
    out[:, 0:3] = action23[:, _ACT["base"]]
    out[:, 3:7] = action23[:, _ACT["trunk"]]
    for lo, arm, gidx_new, gidx_old in (
        (7, "left", 13, _ACT["gripper_left"]),
        (14, "right", 20, _ACT["gripper_right"]),
    ):
        (pt, qt), (pc, qc) = eef_t[arm], eef_c[arm]
        out[:, lo : lo + 3] = pt - pc
        out[:, lo + 3 : lo + 6] = relative_rotation_axisangle(qc, qt)
        out[:, gidx_new] = action23[:, gidx_old]
    return out


def action_stats(action: np.ndarray) -> dict:
    """LeRobot-style per-episode stats for an (F, D) action array."""
    a = np.asarray(action, dtype=np.float64)
    return {
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "mean": a.mean(axis=0).tolist(),
        "std": a.std(axis=0).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).tolist(),
        "count": [int(a.shape[0])],
    }


def aggregate_feature_stats(stats_ft_list: list) -> dict:
    """Count-weighted aggregation of per-episode feature stats.

    Faithful numpy re-implementation of OmniGibson's
    ``omnigibson.learning.utils.lerobot_utils.aggregate_feature_stats`` (mean is
    count-weighted, variance via the parallel algorithm, quantiles are the
    percentile-of-per-episode-quantiles), so a delta-EEF norm-stats asset built
    here matches the methodology of the original assets WITHOUT importing the
    heavy OmniGibson/Isaac stack just to read ``meta/**``.
    """
    means = np.stack([np.asarray(s["mean"], dtype=np.float64) for s in stats_ft_list])
    variances = np.stack([np.asarray(s["std"], dtype=np.float64) ** 2 for s in stats_ft_list])
    counts = np.stack([np.asarray(s["count"], dtype=np.float64) for s in stats_ft_list])
    q01 = np.stack([np.asarray(s["q01"], dtype=np.float64) for s in stats_ft_list])
    q99 = np.stack([np.asarray(s["q99"], dtype=np.float64) for s in stats_ft_list])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    delta = means - total_mean
    total_var = ((variances + delta**2) * counts).sum(axis=0) / total_count
    return {
        "min": np.min(np.stack([np.asarray(s["min"], dtype=np.float64) for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([np.asarray(s["max"], dtype=np.float64) for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_var),
        "q01": np.percentile(q01, 1, axis=0),
        "q99": np.percentile(q99, 99, axis=0),
        "count": total_count,
    }


def aggregate_episode_stats_from_jsonl(
    episodes_stats_path: str, feature_keys=("action", "observation.state")
) -> dict:
    """Aggregate the selected features across all episodes in an ``episodes_stats.jsonl``.

    Returns ``{feature_key: {min/max/mean/std/q01/q99/count: np.ndarray}}``.
    """
    import json

    per_feature = {k: [] for k in feature_keys}
    with open(episodes_stats_path) as fh:
        for line in fh:
            rec = json.loads(line)
            for k in feature_keys:
                per_feature[k].append(rec["stats"][k])
    return {k: aggregate_feature_stats(v) for k, v in per_feature.items() if per_feature[k]}


def validate_converted_dataset(
    dataset_root: str, control_mode: str, tasks: list | None = None
) -> dict:
    """Validate a dataset root is consistent with the selected control mode.

    For ``eef_delta_pose`` this requires ``dataset_root`` to be a converted
    delta-EEF dataset: ``meta/info.json`` action shape must equal 21, a
    ``meta/eef_delta_provenance.json`` must be present with matching
    ``control_mode`` / ``action_env_dim`` (and, when ``tasks`` is given, its task
    set must cover the requested tasks). For ``joint_absolute`` it requires the
    original 23-dim action shape and rejects a delta provenance file. Returns the
    provenance dict (or ``None`` for joint mode). Raises ``ValueError`` on any
    mismatch so a mode/dataset mix-up fails before streaming.
    """
    import json
    import os

    info_path = os.path.join(dataset_root, "meta", "info.json")
    if not os.path.isfile(info_path):
        raise ValueError(f"dataset root {dataset_root} has no meta/info.json.")
    info = json.load(open(info_path))
    action_shape = info["features"]["action"]["shape"]
    action_len = int(action_shape[-1])
    prov_path = os.path.join(dataset_root, "meta", "eef_delta_provenance.json")
    has_prov = os.path.isfile(prov_path)

    if control_mode == "joint_absolute":
        if action_len != OLD_ACTION_DIM:
            raise ValueError(
                f"control_mode=joint_absolute expects a {OLD_ACTION_DIM}-dim "
                f"action dataset, but {dataset_root} has action shape "
                f"{action_shape}."
            )
        if has_prov:
            raise ValueError(
                f"control_mode=joint_absolute but {dataset_root} carries a "
                f"delta-EEF provenance file; point at the original dataset."
            )
        return None

    # eef_delta_pose
    if action_len != NEW_ACTION_DIM:
        raise ValueError(
            f"control_mode=eef_delta_pose expects a {NEW_ACTION_DIM}-dim action "
            f"dataset, but {dataset_root} has action shape {action_shape}. Point "
            f"at the converted delta-EEF dataset."
        )
    if not has_prov:
        raise ValueError(
            f"control_mode=eef_delta_pose but {dataset_root} has no "
            f"meta/eef_delta_provenance.json; it is not a converted delta-EEF "
            f"dataset."
        )
    prov = json.load(open(prov_path))
    if prov.get("control_mode") != "eef_delta_pose":
        raise ValueError(
            f"{prov_path} control_mode={prov.get('control_mode')!r}, expected "
            f"'eef_delta_pose'."
        )
    if int(prov.get("action_env_dim", -1)) != NEW_ACTION_DIM:
        raise ValueError(
            f"{prov_path} action_env_dim={prov.get('action_env_dim')}, expected "
            f"{NEW_ACTION_DIM}."
        )
    if tasks:
        prov_tasks = set(prov.get("tasks") or [])
        missing = [t for t in tasks if t not in prov_tasks]
        if missing:
            raise ValueError(
                f"data.tasks {missing} are not in the converted dataset "
                f"{dataset_root} (provenance tasks: {sorted(prov_tasks)}). Convert "
                f"those tasks or point at the matching converted root."
            )
    return prov


def resolve_behavior_paths(data_cfg, openpi_cfg, control_mode: str) -> dict:
    """Resolve dataset root + norm-stats asset for the selected control mode.

    ``control_mode`` is the switch: for ``eef_delta_pose`` the resolver prefers
    the mode-specific config fields when present (``data.behavior_dataset_root_eef_delta``,
    ``data.train_data_paths_eef_delta``, ``openpi.assets_dir_eef_delta``,
    ``openpi.asset_id_eef_delta``), otherwise falls back to the base fields. This
    keeps all paths in YAML (no hardcoded filesystem defaults in code, per the
    repo convention) while letting one field select the dataset/stats. The strong
    consistency checks (:func:`validate_converted_dataset`,
    ``validate_norm_stats_for_control_mode``) then reject any residual mismatch,
    so flipping ``control_mode`` without matching paths fails loudly rather than
    silently training on the wrong data.

    Returns ``{behavior_dataset_root, train_data_paths, assets_dir, asset_id}``.
    """
    suffix = "_eef_delta" if control_mode == "eef_delta_pose" else ""

    def pick(cfg, base_key):
        if suffix:
            mode_key = base_key + suffix
            if mode_key in cfg and cfg.get(mode_key) is not None:
                return cfg.get(mode_key)
        return cfg.get(base_key)

    return {
        "behavior_dataset_root": pick(data_cfg, "behavior_dataset_root"),
        "train_data_paths": pick(data_cfg, "train_data_paths"),
        "assets_dir": pick(openpi_cfg, "assets_dir"),
        "asset_id": pick(openpi_cfg, "asset_id"),
    }


# --------------------------------------------------------------------------- #
# Dataset-level conversion: writes a new LeRobot dataset with 21-dim delta-EEF
# actions, symlinking the (large) original videos, regenerating meta for the
# converted tasks, and recording provenance. Refuses to touch the original.
# --------------------------------------------------------------------------- #
def _task_indices_for_names(src_root, task_names):
    import json

    name_to_idx = {}
    with open(f"{src_root}/meta/tasks.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            name_to_idx[d["task_name"]] = d["task_index"]
    missing = [t for t in task_names if t not in name_to_idx]
    if missing:
        raise ValueError(f"Unknown task name(s) {missing}; not in {src_root}/meta/tasks.jsonl.")
    return {t: name_to_idx[t] for t in task_names}


def convert_dataset(
    src_root: str,
    dst_root: str,
    task_names: list,
    urdf_path: str,
    *,
    converter_version: str = "1",
    omnigibson_version: str = "unknown",
    overwrite: bool = False,
) -> dict:
    """Convert the selected tasks of a BEHAVIOR LeRobot dataset to delta-EEF.

    Writes ``dst_root`` with 21-dim actions for the requested tasks; symlinks
    the original ``videos/`` (never copies); regenerates ``meta/`` filtered to
    the converted tasks with recomputed 21-dim action stats; and writes
    ``meta/eef_delta_provenance.json``. The original dataset is never modified.
    """
    import json
    import os
    import shutil

    import pyarrow.parquet as pq

    src_root = os.path.abspath(src_root)
    dst_root = os.path.abspath(dst_root)
    if dst_root == src_root or dst_root.startswith(src_root + os.sep):
        raise ValueError(
            f"Refusing in-place / nested conversion: dst_root ({dst_root}) must "
            f"be outside src_root ({src_root})."
        )
    if os.path.exists(dst_root):
        if not overwrite:
            raise FileExistsError(f"dst_root exists: {dst_root} (pass overwrite=True).")
        shutil.rmtree(dst_root)

    fk = UrdfArmFK(urdf_path)
    task_idx = _task_indices_for_names(src_root, task_names)
    selected_task_indices = set(task_idx.values())

    os.makedirs(f"{dst_root}/meta", exist_ok=True)
    os.makedirs(f"{dst_root}/data", exist_ok=True)
    # Reuse videos by symlink — never a second physical copy (they are large).
    os.symlink(f"{src_root}/videos", f"{dst_root}/videos")
    # Per-episode meta (meta/episodes/task-XXXX/episode_*.json) is action-
    # independent env-config metadata that the LeRobot loader asserts exists.
    # Symlink the selected tasks' per-episode meta dirs (small JSON, unchanged by
    # the action conversion) rather than copy.
    src_ep_meta = f"{src_root}/meta/episodes"
    if os.path.isdir(src_ep_meta):
        os.makedirs(f"{dst_root}/meta/episodes", exist_ok=True)
        for tidx in sorted(_task_indices_for_names(src_root, task_names).values()):
            task_dir = f"task-{tidx:04d}"
            src_task = f"{src_ep_meta}/{task_dir}"
            if os.path.isdir(src_task):
                os.symlink(src_task, f"{dst_root}/meta/episodes/{task_dir}")

    info = json.load(open(f"{src_root}/meta/info.json"))
    chunks_size = int(info.get("chunks_size", 10000))

    # Filter episodes.jsonl / episodes_stats.jsonl to the selected tasks.
    sel_episodes = []
    with open(f"{src_root}/meta/episodes.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            if d["episode_index"] // chunks_size in selected_task_indices:
                sel_episodes.append(d)
    sel_ep_set = {d["episode_index"] for d in sel_episodes}

    # Convert each episode's parquet, collect new action stats.
    new_action_stats = {}
    total_frames = 0
    data_tmpl = info["data_path"]
    for d in sel_episodes:
        ei = d["episode_index"]
        chunk = ei // chunks_size
        rel = data_tmpl.format(episode_chunk=chunk, episode_index=ei)
        src_parquet = f"{src_root}/{rel}"
        dst_parquet = f"{dst_root}/{rel}"
        os.makedirs(os.path.dirname(dst_parquet), exist_ok=True)

        table = pq.read_table(src_parquet)
        cols = table.column_names
        pdf = table.to_pandas()
        act23 = np.stack([np.asarray(a, dtype=np.float64) for a in pdf["action"]])
        st256 = np.stack([np.asarray(s, dtype=np.float64) for s in pdf["observation.state"]])
        act21 = convert_episode_actions(act23, st256, fk).astype(np.float32)
        pdf["action"] = list(act21)
        # Preserve original column order and schema (only action changes width).
        import pyarrow as pa

        out_table = pa.Table.from_pandas(pdf[cols], preserve_index=False)
        pq.write_table(out_table, dst_parquet)

        new_action_stats[ei] = action_stats(act21)
        total_frames += int(d.get("length", act21.shape[0]))

    # Regenerate meta/episodes_stats.jsonl: copy each selected episode's stats,
    # replacing only the "action" entry (state and image stats are unchanged).
    with open(f"{src_root}/meta/episodes_stats.jsonl") as fin, open(
        f"{dst_root}/meta/episodes_stats.jsonl", "w"
    ) as fout:
        for line in fin:
            rec = json.loads(line)
            if rec["episode_index"] in sel_ep_set:
                rec["stats"]["action"] = new_action_stats[rec["episode_index"]]
                fout.write(json.dumps(rec) + "\n")

    # episodes.jsonl (filtered), tasks.jsonl (filtered), info.json (action 21).
    with open(f"{dst_root}/meta/episodes.jsonl", "w") as fout:
        for d in sel_episodes:
            fout.write(json.dumps(d) + "\n")
    with open(f"{src_root}/meta/tasks.jsonl") as fin, open(
        f"{dst_root}/meta/tasks.jsonl", "w"
    ) as fout:
        for line in fin:
            if json.loads(line)["task_index"] in selected_task_indices:
                fout.write(line if line.endswith("\n") else line + "\n")

    info["features"]["action"]["shape"] = [NEW_ACTION_DIM]
    info["total_episodes"] = len(sel_episodes)
    info["total_frames"] = total_frames
    info["total_tasks"] = len(selected_task_indices)
    # Recompute total_videos for the filtered task set (one clip per video key
    # per episode) so the metadata matches the selected episodes rather than the
    # full source dataset. The clips themselves are reused via the videos/ symlink.
    if "total_videos" in info:
        num_video_keys = len(info.get("video_keys") or []) or (
            sum(
                1
                for f in info.get("features", {}).values()
                if f.get("dtype") == "video"
            )
        )
        info["total_videos"] = len(sel_episodes) * num_video_keys
    info["total_chunks"] = len(selected_task_indices)
    json.dump(info, open(f"{dst_root}/meta/info.json", "w"), indent=4)

    provenance = {
        "control_mode": "eef_delta_pose",
        "action_env_dim": NEW_ACTION_DIM,
        "model_action_dim": 32,
        "source_action_dim": OLD_ACTION_DIM,
        "controller": {"arms": "InverseKinematicsController", "mode": "pose_delta_ori",
                       "command_input_limits": None, "command_output_limits": None},
        "conversion_source": "dpos=FK(action_target)-FK(state_achieved); "
                             "dori=quat2axisangle(mat2quat(R_target @ R_current.T))",
        "orientation_convention": "exact_relative_rotation_axisangle_base_frame_left_multiply",
        "cspace_joints": CSPACE_JOINTS,
        "eef_offset_pos": EEF_OFFSET_POS.tolist(),
        "eef_offset_quat_xyzw": EEF_OFFSET_QUAT_XYZW.tolist(),
        "tasks": list(task_names),
        "task_indices": task_idx,
        "converter_version": converter_version,
        "omnigibson_version": omnigibson_version,
        "urdf_path": urdf_path,
        "source_root": src_root,
    }
    json.dump(provenance, open(f"{dst_root}/meta/eef_delta_provenance.json", "w"), indent=2)
    return {"episodes": len(sel_episodes), "frames": total_frames, "dst_root": dst_root}



