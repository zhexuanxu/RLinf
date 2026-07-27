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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf
from openpi.models import model as openpi_model

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
    _Repack,
)
from rlinf.envs.behavior.control_modes import (
    CONTROL_MODES,
    R1PRO_PROPRIO_INDICES,
)
from rlinf.envs.behavior.utils import (
    merge_robot_override,
    select_control_mode_robot_override,
)
from rlinf.models.embodiment.openpi.policies.behavior_policy import BehaviorInputs

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _r1pro_robot_override():
    env_cfg = OmegaConf.load(
        _REPO_ROOT / "examples/embodiment/config/env/behavior_r1pro.yaml"
    )
    return env_cfg.omni_config.robots[0]


@pytest.mark.parametrize(
    ("control_mode", "controller_name", "controller_mode"),
    [
        ("abs_joint", "JointController", False),
        ("delta_joint", "JointController", True),
        ("abs_eef", "InverseKinematicsController", "absolute_pose"),
        ("delta_eef", "InverseKinematicsController", "pose_delta_ori"),
    ],
)
def test_control_selector_maps_detached_arm_leaves(
    control_mode, controller_name, controller_mode
):
    robot_override = _r1pro_robot_override()
    original = OmegaConf.to_container(robot_override, resolve=True)

    selected = select_control_mode_robot_override(robot_override, control_mode)

    assert OmegaConf.to_container(robot_override, resolve=True) == original
    controllers = selected.controller_config
    assert controllers.base.name == "HolonomicBaseJointController"
    assert controllers.trunk.name == "JointController"
    for arm in ("arm_left", "arm_right"):
        controller = controllers[arm]
        assert controller.name == controller_name
        assert not set(CONTROL_MODES).intersection(controller)
        if controller_name == "JointController":
            assert controller.use_delta_commands is controller_mode
            assert "mode" not in controller
        else:
            assert controller.mode == controller_mode
            assert "motor_type" not in controller
            assert "pos_kp" not in controller


@pytest.mark.parametrize(
    "legacy_name",
    ["joint_absolute", "absolute_eef", "eef_delta_pose", "ABS_JOINT"],
)
def test_control_selector_rejects_aliases(legacy_name):
    with pytest.raises(ValueError, match="does not define control_mode"):
        select_control_mode_robot_override(_r1pro_robot_override(), legacy_name)


@pytest.mark.parametrize(
    ("model_group", "control_mode"),
    [
        ("actor", "delta_eef"),
        ("rollout", "delta_joint"),
    ],
)
def test_behavior_env_config_selects_model_control_mode(model_group, control_mode):
    env_cfg = OmegaConf.load(
        _REPO_ROOT / "examples/embodiment/config/env/behavior_r1pro.yaml"
    )
    root_cfg = OmegaConf.create(
        {
            model_group: {"model": {"openpi": {"control_mode": control_mode}}},
            "env": {"eval": env_cfg},
        }
    )

    assert root_cfg.env.eval.control_mode == control_mode


def test_model_templates_derive_action_width_from_exact_control_map():
    for relative_path in (
        "examples/sft/config/model/pi0_5_pytorch.yaml",
        "examples/embodiment/config/model/pi0_5_pytorch.yaml",
    ):
        cfg = OmegaConf.load(_REPO_ROOT / relative_path)
        assert tuple(cfg.openpi.control_mode_action_dims) == CONTROL_MODES
        for mode, expected_dim in {
            "abs_joint": 23,
            "delta_joint": 23,
            "abs_eef": 21,
            "delta_eef": 21,
        }.items():
            cfg.openpi.control_mode = mode
            assert cfg.action_dim == expected_dim
            assert cfg.openpi.action_env_dim == expected_dim


def test_behavior_input_uses_selected_state_layout():
    state = np.zeros(256, dtype=np.float32)
    state[R1PRO_PROPRIO_INDICES["base_qvel"]] = [1, 2, 3]
    state[R1PRO_PROPRIO_INDICES["trunk_qpos"]] = [4, 5, 6, 7]
    state[R1PRO_PROPRIO_INDICES["arm_left_qpos"]] = np.arange(10, 17)
    state[R1PRO_PROPRIO_INDICES["arm_right_qpos"]] = np.arange(20, 27)
    state[R1PRO_PROPRIO_INDICES["gripper_left_qpos"]] = [0.1, 0.2]
    state[R1PRO_PROPRIO_INDICES["gripper_right_qpos"]] = [0.4, 0.5]
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    wrist = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    transform = BehaviorInputs(
        model_type=openpi_model.ModelType.PI0,
        extract_state_from_proprio=True,
        use_all_wrist_images=True,
        state_token="abs_joint",
    )

    output = transform(
        {
            "observation/image": image,
            "observation/wrist_image": wrist,
            "observation/state": state,
        }
    )

    assert output["state"].shape == (23,)
    assert output["state"][14] == pytest.approx(0.3)
    np.testing.assert_array_equal(output["state"][15:22], np.arange(20, 27))


def test_repack_rejects_dataset_from_another_control_mode():
    frame = {
        "observation.images.rgb.head": np.zeros((3, 4, 4)),
        "observation.images.rgb.left_wrist": np.zeros((3, 4, 4)),
        "observation.images.rgb.right_wrist": np.zeros((3, 4, 4)),
        "observation.state": np.zeros(256),
        "action": np.zeros((32, 23)),
        "task": "turn on the radio",
    }

    with pytest.raises(ValueError, match="expected 21"):
        _Repack(action_env_dim=21)(frame)


def test_robot_merge_replaces_controller_when_class_changes():
    omni_cfg = OmegaConf.create(
        {
            "robots": [
                {
                    "type": "R1Pro",
                    "controller_config": {
                        "arm_left": {
                            "name": "JointController",
                            "motor_type": "position",
                            "pos_kp": 150,
                        },
                        "trunk": {
                            "name": "JointController",
                            "pos_kp": 150,
                        },
                    },
                }
            ]
        }
    )
    override = OmegaConf.create(
        {
            "controller_config": {
                "arm_left": {
                    "name": "InverseKinematicsController",
                    "mode": "absolute_pose",
                },
                "trunk": {
                    "name": "JointController",
                    "command_input_limits": None,
                },
            }
        }
    )

    merge_robot_override(omni_cfg, override)

    assert OmegaConf.to_container(
        omni_cfg.robots[0].controller_config.arm_left, resolve=True
    ) == {
        "name": "InverseKinematicsController",
        "mode": "absolute_pose",
    }
    assert omni_cfg.robots[0].controller_config.trunk.pos_kp == 150
