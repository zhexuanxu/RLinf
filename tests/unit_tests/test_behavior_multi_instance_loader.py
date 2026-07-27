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
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.envs.behavior.instance_loader as instance_loader_module
from rlinf.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
)
from rlinf.envs.behavior.utils import (
    clear_robot_grasp_state,
    reset_robot_joint_state_to_reset_pose,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ACTIVITY_NAME = "turning_on_radio"


def _write_instances(instance_dir: Path, instance_ids: list[int]) -> None:
    instance_dir.mkdir()
    for instance_id in instance_ids:
        filename = (
            "house_double_floor_lower_task_turning_on_radio_"
            f"0_{instance_id}_template-tro_state.json"
        )
        (instance_dir / filename).write_text("{}", encoding="utf-8")


def _omni_cfg(
    instance_dir: Path | None,
    instance_ids,
    *,
    instance_resample_mode: str = "offline",
):
    return OmegaConf.create(
        {
            "task": {
                "activity_name": _ACTIVITY_NAME,
                "activity_definition_id": 0,
                "activity_instance_id": instance_ids,
                "activity_instance_dir": (
                    str(instance_dir) if instance_dir is not None else None
                ),
                "instance_file_format": "tro_state",
                "instance_resample_mode": instance_resample_mode,
                "online_object_sampling": instance_resample_mode == "online",
                "use_presampled_robot_pose": instance_resample_mode != "online",
            }
        }
    )


def test_loader_filters_ordered_test_ids_and_bootstraps_template_zero(tmp_path):
    _write_instances(tmp_path / "instances", [1, 2, 3, 4])
    cfg = _omni_cfg(tmp_path / "instances", [4, 2, 3])

    loader = ActivityInstanceLoader.from_omni_cfg(
        cfg,
        seed_offset=1,
        total_num_workers=3,
    )
    initial_cfg = loader.build_initial_omni_cfg()

    assert [entry.instance_id for entry in loader.activity_instances] == [4, 2, 3]
    assert loader.activity_instance_id == 4
    assert initial_cfg.task.activity_instance_id == 0
    assert list(cfg.task.activity_instance_id) == [4, 2, 3]


@pytest.mark.parametrize(
    ("instance_ids", "mode", "error"),
    [
        ([], "offline", "must not be empty"),
        ([1, 2], "disabled", "requires task.instance_resample_mode='offline'"),
        ([1, 2], "online", "requires task.instance_resample_mode='offline'"),
        ([1, "2"], "offline", "list entry must be an integer"),
    ],
)
def test_loader_rejects_invalid_instance_id_lists(
    tmp_path,
    instance_ids,
    mode,
    error,
):
    _write_instances(tmp_path / "instances", [1, 2])
    cfg = _omni_cfg(tmp_path / "instances", instance_ids, instance_resample_mode=mode)

    with pytest.raises(ValueError, match=error):
        ActivityInstanceLoader.from_omni_cfg(cfg)


def test_loader_rejects_missing_requested_ids(tmp_path):
    _write_instances(tmp_path / "instances", [1, 2])
    cfg = _omni_cfg(tmp_path / "instances", [2, 7])

    with pytest.raises(ValueError, match=r"not present.*\[7\]"):
        ActivityInstanceLoader.from_omni_cfg(cfg)


def test_offline_selection_is_global_round_robin_across_workers_and_resets():
    instance_files = tuple(
        ActivityInstanceFile(
            instance_id=instance_id,
            path=f"/cache/{instance_id}.json",
            file_format="tro_state",
        )
        for instance_id in range(1, 6)
    )
    vec_env = SimpleNamespace(envs=[object(), object()])

    selections = []
    for worker_index in range(2):
        loader = ActivityInstanceLoader(
            omni_cfg=OmegaConf.create({"task": {}}),
            activity_name=_ACTIVITY_NAME,
            activity_instance_id=1,
            instance_resample_mode="offline",
            activity_instances=instance_files,
            seed_offset=worker_index,
            total_num_workers=2,
        )
        worker_selections = []
        loader._apply_instance_files = (  # noqa: SLF001 - isolate scheduling logic
            lambda _vec_env, files: worker_selections.append(
                [entry.instance_id for entry in files]
            )
        )
        loader.prepare_reset(vec_env)
        loader.prepare_reset(vec_env)
        selections.append(worker_selections)

    assert selections == [
        [[1, 2], [5, 1]],
        [[3, 4], [2, 3]],
    ]


def test_tro_state_reload_resets_scene_before_rollout(monkeypatch):
    calls = []

    def _load(env, instance_id, tro_file_path, reset_scene):
        calls.append((env, instance_id, tro_file_path, reset_scene))

    monkeypatch.setattr(
        instance_loader_module,
        "load_activity_instance_tro_state",
        _load,
    )
    env = object()
    instance_file = ActivityInstanceFile(
        instance_id=8,
        path="/cache/8.json",
        file_format="tro_state",
    )
    loader = ActivityInstanceLoader(
        omni_cfg=OmegaConf.create({"task": {}}),
        activity_name=_ACTIVITY_NAME,
        activity_instance_id=8,
        instance_resample_mode="disabled",
        activity_instances=(instance_file,),
    )

    loader._load_tro_state_instances(  # noqa: SLF001 - isolate reset contract
        SimpleNamespace(envs=[env]),
        [instance_file],
    )

    assert calls == [(env, 8, "/cache/8.json", True)]


def test_robot_joint_reset_preserves_base_and_zeros_velocity():
    class _Robot:
        reset_joint_pos = torch.arange(10, dtype=torch.float32)

        def __init__(self):
            self.positions = torch.full((10,), 9.0)
            self.updated_positions = None
            self.updated_velocities = None
            self.keep_still_calls = 0

        def get_joint_positions(self):
            return self.positions

        def set_joint_positions(self, positions, drive):
            assert drive is False
            self.updated_positions = positions

        def set_joint_velocities(self, velocities, drive):
            assert drive is False
            self.updated_velocities = velocities

        def keep_still(self):
            self.keep_still_calls += 1

    robot = _Robot()

    reset_robot_joint_state_to_reset_pose(
        robot,
        preserve_base_pose=True,
        base_joint_dim=3,
    )

    torch.testing.assert_close(robot.updated_positions[:3], torch.full((3,), 9.0))
    torch.testing.assert_close(
        robot.updated_positions[3:],
        torch.arange(3, 10, dtype=torch.float32),
    )
    torch.testing.assert_close(robot.updated_velocities, torch.zeros(10))
    assert robot.keep_still_calls == 2


def test_assisted_grasp_state_is_cleared_for_every_arm():
    class _Robot:
        is_manipulation = True
        arm_names = ["left", "right"]

        def __init__(self):
            self._ag_obj_in_hand = {"left": object(), "right": object()}
            self._ag_obj_constraints = {"left": object(), "right": object()}
            self._ag_obj_constraint_params = {"left": {}, "right": {}}
            self._ag_release_counter = {"left": 3, "right": 4}
            self._ag_grasp_counter = {"left": 5, "right": 6}
            self.released = []
            self.keep_still_calls = 0

        def release_grasp_immediately(self, arm):
            self.released.append(arm)

        def keep_still(self):
            self.keep_still_calls += 1

    robot = _Robot()

    clear_robot_grasp_state(robot)

    assert robot.released == ["left", "right"]
    assert robot._ag_obj_in_hand == {"left": None, "right": None}
    assert robot._ag_obj_constraints == {"left": None, "right": None}
    assert robot._ag_obj_constraint_params == {"left": None, "right": None}
    assert robot._ag_release_counter == {"left": 0, "right": 0}
    assert robot._ag_grasp_counter == {"left": 0, "right": 0}
    assert robot.keep_still_calls == 1


def test_test_set_eval_yaml_uses_ordered_cached_instances():
    config_path = (
        _REPO_ROOT
        / "evaluations/behavior/behavior_openpi_pi05_pytorch_vlm_vla_eval_test.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.env.eval.total_num_envs == 8
    assert cfg.env.eval.rollout_epoch == 2
    assert cfg.env.eval.total_num_envs * cfg.env.eval.rollout_epoch >= 10
    assert list(cfg.env.eval.omni_config.task.activity_instance_id) == list(
        range(1, 11)
    )
    assert cfg.env.eval.omni_config.task.instance_resample_mode == "offline"
    assert cfg.env.eval.omni_config.task.instance_file_format == "tro_state"
    assert cfg.rollout.model.openpi.control_mode == "abs_joint"
