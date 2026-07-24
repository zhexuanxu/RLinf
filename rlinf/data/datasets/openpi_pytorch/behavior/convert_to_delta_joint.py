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

"""State-based conversion of BEHAVIOR R1Pro absolute-joint actions to
joint-space DELTA actions, for the ``delta_joint`` control mode.

``delta_joint`` is the joint-space sibling of ``delta_eef``: the prompt state stays
the absolute joint state (``abs_joint``), and only the two 7-DoF arms become the
state-to-state achieved-joint delta

    arm = qpos_{t+1} - qpos_t

where ``qpos_t`` is the arm's ACHIEVED joint position at step t, read directly from
the 256-dim ``observation.state`` proprio (no FK). base(3, velocity), trunk(4, abs
joint) and the two grippers(1) are copied from the recorded 23-dim action unchanged,
giving a **23-dim** action with the same slice layout as ``joint_absolute`` -- the
arms are the only channels that change.

At eval the two arm ``JointController``s run with ``use_delta_commands: True`` and
``command_*_limits: null`` (raw radians), so the controller applies
``q_target = q_current + delta``: feeding ``qpos_{t+1} - qpos_t`` drives the arm to
the achieved next state, mirroring how ``delta_eef`` drives the IK ``pose_delta_ori``
controller. The last frame has no t+1, so it holds (delta = 0).

Pure numpy; importing this module never pulls in torch / OmniGibson / Isaac. It
reuses the shared LeRobot dataset machinery in ``convert_to_eef_delta``
(videos/meta symlinks, meta regeneration, provenance).
"""

from __future__ import annotations

import numpy as np

# Achieved arm joint-position slices inside the 256-dim proprio observation.state
# (same slices behavior_policy uses for the abs_joint state token).
_PROPRIO = {
    "arm_left_qpos": slice(158, 165),  # 7, joint1..7
    "arm_right_qpos": slice(197, 204),  # 7
}

# Recorded 23-dim action slices (native OmniGibson controller order); base/trunk and
# the two grippers are copied through unchanged to the 23-dim delta-joint action.
_ACT = {
    "base": slice(0, 3),
    "trunk": slice(3, 7),
    "gripper_left": 14,
    "gripper_right": 22,
}

JOINT_ACTION_DIM = 23
OLD_ACTION_DIM = 23


def action_to_delta_joint(
    action23: np.ndarray, state_t: np.ndarray, state_t1: np.ndarray
) -> np.ndarray:
    """23-dim delta_joint action: arms = achieved qpos_{t+1} - qpos_t; base/trunk/
    grippers copied from the recorded absolute action.

    Layout (identical slices to joint_absolute):
        out[0:3]   = action base (velocity, passthrough)
        out[3:7]   = action trunk (abs joint, passthrough)
        out[7:14]  = state_t1 arm_left_qpos  - state_t arm_left_qpos   (delta)
        out[14]    = action gripper_left (passthrough)
        out[15:22] = state_t1 arm_right_qpos - state_t arm_right_qpos  (delta)
        out[22]    = action gripper_right (passthrough)
    """
    a = np.asarray(action23, dtype=np.float64)
    s0 = np.asarray(state_t, dtype=np.float64)
    s1 = np.asarray(state_t1, dtype=np.float64)
    out = np.empty(JOINT_ACTION_DIM, dtype=np.float64)
    out[0:3] = a[_ACT["base"]]
    out[3:7] = a[_ACT["trunk"]]
    out[7:14] = s1[_PROPRIO["arm_left_qpos"]] - s0[_PROPRIO["arm_left_qpos"]]
    out[14] = a[_ACT["gripper_left"]]
    out[15:22] = s1[_PROPRIO["arm_right_qpos"]] - s0[_PROPRIO["arm_right_qpos"]]
    out[22] = a[_ACT["gripper_right"]]
    return out


def convert_episode_actions(action23: np.ndarray, state256: np.ndarray) -> np.ndarray:
    """Convert a whole episode (F,23)+(F,256) -> (F,23) delta-joint. The last frame
    has no t+1, so it holds (delta = 0)."""
    a = np.asarray(action23, dtype=np.float64).reshape(-1, OLD_ACTION_DIM)
    s = np.asarray(state256, dtype=np.float64).reshape(-1, 256)
    f = a.shape[0]
    out = np.empty((f, JOINT_ACTION_DIM), dtype=np.float64)
    for t in range(f):
        t1 = min(t + 1, f - 1)
        out[t] = action_to_delta_joint(a[t], s[t], s[t1])
    return out


# --------------------------------------------------------------------------- #
# Dataset-level conversion (reuses the shared LeRobot machinery in
# convert_to_eef_delta: videos/meta symlinks, meta regeneration, provenance).
# --------------------------------------------------------------------------- #
def convert_dataset(
    src_root: str,
    dst_root: str,
    task_names: list,
    *,
    converter_version: str = "1",
    omnigibson_version: str = "unknown",
    overwrite: bool = False,
    progress_every: int = 0,
) -> dict:
    """Convert the selected tasks of a BEHAVIOR LeRobot dataset to ``delta_joint``
    (state-based, no FK). Symlinks videos/ + per-task episode meta, regenerates meta
    with 23-dim action stats, writes ``eef_delta_provenance.json``.
    """
    from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef_delta import (
        _convert_dataset_impl,
    )

    provenance = {
        "control_mode": "delta_joint",
        "action_env_dim": JOINT_ACTION_DIM,
        "model_action_dim": 32,
        "source_action_dim": OLD_ACTION_DIM,
        "controller": {
            "arms": "JointController",
            "use_delta_commands": True,
            "command_input_limits": None,
            "command_output_limits": None,
        },
        "conversion_source": (
            "arm = qpos_{t+1} - qpos_t (achieved joint delta) from proprio state; "
            "base/trunk/grippers passthrough from recorded action"
        ),
        "orientation_convention": "n/a (joint-space)",
        "proprio_arm_slices": {k: [s.start, s.stop] for k, s in _PROPRIO.items()},
        "converter_version": converter_version,
        "omnigibson_version": omnigibson_version,
    }
    return _convert_dataset_impl(
        src_root,
        dst_root,
        task_names,
        lambda a23, s256: convert_episode_actions(a23, s256),
        provenance,
        action_env_dim=JOINT_ACTION_DIM,
        overwrite=overwrite,
        progress_every=progress_every,
    )


def _all_task_names(src_root: str) -> list:
    import json

    names = []
    with open(f"{src_root}/meta/tasks.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                names.append(json.loads(line)["task_name"])
    if not names:
        raise ValueError(f"No tasks found in {src_root}/meta/tasks.jsonl.")
    return names


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Convert a BEHAVIOR R1Pro LeRobot dataset from 23-dim absolute "
        "joint actions to 23-dim delta_joint actions (state-based, no FK): arms "
        "become qpos_{t+1}-qpos_t, base/trunk/grippers pass through.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--src-root", required=True, help="Original 23-dim joint dataset root."
    )
    p.add_argument(
        "--dst-root", required=True, help="Destination dataset root (outside src)."
    )
    p.add_argument("--mode", default="delta_joint", choices=("delta_joint",))
    p.add_argument(
        "--tasks", nargs="*", default=None, help="Task name(s); omit for ALL."
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--omnigibson-version", default="unknown")
    p.add_argument("--converter-version", default="1")
    p.add_argument("--progress-every", type=int, default=200)
    args = p.parse_args()

    task_names = args.tasks if args.tasks else _all_task_names(args.src_root)
    print(
        f"[convert-delta-joint] {len(task_names)} task(s): "
        f"{args.src_root} -> {args.dst_root}",
        flush=True,
    )
    r = convert_dataset(
        args.src_root,
        args.dst_root,
        task_names,
        converter_version=args.converter_version,
        omnigibson_version=args.omnigibson_version,
        overwrite=args.overwrite,
        progress_every=args.progress_every,
    )
    print(
        f"[convert-delta-joint] DONE: {r['episodes']} episodes, {r['frames']} frames "
        f"-> {r['dst_root']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
