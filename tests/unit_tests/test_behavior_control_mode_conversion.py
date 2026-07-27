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

"""Fast, simulator-free tests for BEHAVIOR control-mode data tooling."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.data.datasets.openpi_pytorch.behavior.compute_norm_stats import (
    compute_norm_stats,
    write_norm_stats,
)
from rlinf.data.datasets.openpi_pytorch.behavior.convert_control_mode import (
    action_statistics,
    convert_behavior_dataset,
)
from rlinf.envs.behavior.control_modes import (
    CONTROL_MODE_ACTION_DIMS,
    CONTROL_MODES,
    STATE_TOKENS,
    convert_action,
    convert_episode_actions,
    extract_state_from_proprio,
    quaternion_to_axis_angle,
    relative_rotation_axis_angle,
    validate_control_mode,
    validate_control_mode_dataset,
    validate_norm_stats_asset,
    validate_state_token,
)


def _episode_arrays(offset: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    frames = 4
    actions = np.arange(frames * 23, dtype=np.float64).reshape(frames, 23) / 20
    actions += offset
    states = np.zeros((frames, 256), dtype=np.float64)
    for frame in range(frames):
        states[frame, 158:165] = offset + frame + np.arange(7) / 10
        states[frame, 197:204] = offset - frame + np.arange(7) / 20
        states[frame, 186:189] = [0.1 * frame, 0.2 + offset, -0.3]
        states[frame, 225:228] = [-0.2 * frame, -0.1, 0.4 + offset]
        states[frame, 189:193] = Rotation.from_euler("z", 0.1 * frame).as_quat()
        states[frame, 228:232] = Rotation.from_euler("x", -0.2 * frame).as_quat()
        states[frame, 193:195] = [0.01 * frame, 0.02 * frame]
        states[frame, 232:234] = [0.03 * frame, 0.04 * frame]
        states[frame, 236:240] = offset + frame + np.arange(4)
        states[frame, 253:256] = [frame, frame + 1, frame + 2]
    return actions, states


def _write_source_dataset(root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    (root / "meta").mkdir(parents=True)
    (root / "videos").mkdir()
    (root / "annotations").mkdir()
    data = {
        10: _episode_arrays(0.0),
        10010: _episode_arrays(10.0),
    }
    tasks = [
        {"task_index": 0, "task_name": "task_zero", "task": "do zero"},
        {"task_index": 1, "task_name": "task_one", "task": "do one"},
    ]
    episodes = [
        {"episode_index": episode, "tasks": [f"do {index}"], "length": 4}
        for index, episode in enumerate(data)
    ]
    info = {
        "codebase_version": "v2.1",
        "robot_type": "R1Pro",
        "total_episodes": 2,
        "total_frames": 8,
        "total_tasks": 2,
        "total_videos": 0,
        "chunks_size": 10000,
        "fps": 30,
        "splits": {"train": "0:2"},
        "data_path": (
            "data/task-{episode_chunk:04d}/episode_{episode_index:08d}.parquet"
        ),
        "features": {
            "observation.state": {
                "dtype": "float64",
                "shape": [256],
                "names": None,
            },
            "action": {"dtype": "float32", "shape": [23], "names": None},
        },
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in tasks)
    )
    (root / "meta/episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episodes)
    )
    episode_stats = []
    for episode_index, (actions, states) in data.items():
        episode_stats.append(
            {
                "episode_index": episode_index,
                "stats": {
                    "action": action_statistics(actions),
                    "observation.state": action_statistics(states),
                },
            }
        )
        output = (
            root
            / f"data/task-{episode_index // 10000:04d}"
            / f"episode_{episode_index:08d}.parquet"
        )
        output.parent.mkdir(parents=True)
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float64())
                ),
                "action": pa.array(
                    actions.astype(np.float32).tolist(),
                    type=pa.list_(pa.float32()),
                ),
            }
        )
        pq.write_table(table, output)
    (root / "meta/episodes_stats.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in episode_stats)
    )
    return data


class TestExactPublicSurface:
    def test_names_and_dimensions_are_exact(self):
        assert CONTROL_MODES == (
            "abs_joint",
            "delta_joint",
            "abs_eef",
            "delta_eef",
        )
        assert CONTROL_MODE_ACTION_DIMS == {
            "abs_joint": 23,
            "delta_joint": 23,
            "abs_eef": 21,
            "delta_eef": 21,
        }
        assert STATE_TOKENS == (
            "none",
            "abs_joint_old",
            "abs_joint",
            "abs_eef",
        )

    @pytest.mark.parametrize(
        "legacy_name",
        ["joint_absolute", "absolute_eef", "eef_delta_pose", "delta_pose"],
    )
    def test_control_mode_aliases_are_rejected(self, legacy_name):
        with pytest.raises(ValueError, match="Legacy aliases"):
            validate_control_mode(legacy_name)

    @pytest.mark.parametrize("legacy_name", ["comet", "align", "no_state"])
    def test_state_token_aliases_are_rejected(self, legacy_name):
        with pytest.raises(ValueError, match="Legacy aliases"):
            validate_state_token(legacy_name)


class TestConversionMath:
    def test_state_layouts(self):
        _, states = _episode_arrays()
        assert extract_state_from_proprio(states, "none").shape == (4, 23)
        assert extract_state_from_proprio(states, "abs_joint_old").shape == (4, 23)
        assert extract_state_from_proprio(states, "abs_joint").shape == (4, 23)
        assert extract_state_from_proprio(states, "abs_eef").shape == (4, 21)
        np.testing.assert_array_equal(
            extract_state_from_proprio(states, "none"),
            extract_state_from_proprio(states, "abs_joint_old"),
        )
        old = extract_state_from_proprio(states[1], "abs_joint_old")
        aligned = extract_state_from_proprio(states[1], "abs_joint")
        assert old[21] == aligned[14]
        np.testing.assert_array_equal(old[14:21], aligned[15:22])

    def test_delta_joint_uses_achieved_next_state(self):
        actions, states = _episode_arrays()
        converted = convert_episode_actions(actions, states, "delta_joint")
        np.testing.assert_allclose(
            converted[0, 7:14], states[1, 158:165] - states[0, 158:165]
        )
        np.testing.assert_allclose(
            converted[0, 15:22], states[1, 197:204] - states[0, 197:204]
        )
        np.testing.assert_allclose(converted[-1, 7:14], 0)
        np.testing.assert_allclose(converted[-1, 15:22], 0)
        np.testing.assert_allclose(converted[:, :7], actions[:, :7])
        np.testing.assert_allclose(converted[:, 14], actions[:, 14])
        np.testing.assert_allclose(converted[:, 22], actions[:, 22])

    def test_eef_modes_and_rotation_convention(self):
        actions, states = _episode_arrays()
        absolute = convert_action(actions[0], states[0], states[1], "abs_eef")
        delta = convert_action(actions[0], states[0], states[1], "delta_eef")
        assert absolute.shape == (21,)
        assert delta.shape == (21,)
        np.testing.assert_allclose(absolute[7:10], states[1, 186:189])
        np.testing.assert_allclose(delta[7:10], states[1, 186:189] - states[0, 186:189])

        reconstructed = Rotation.from_rotvec(delta[10:13]) * Rotation.from_quat(
            states[0, 189:193]
        )
        np.testing.assert_allclose(
            reconstructed.as_matrix(),
            Rotation.from_quat(states[1, 189:193]).as_matrix(),
            atol=1e-12,
        )

    def test_axis_angle_is_stable_at_pi(self):
        quaternion = np.array([-0.0222, 0.9998, -0.0003, 0.0012])
        quaternion /= np.linalg.norm(quaternion)
        axis_angle = quaternion_to_axis_angle(quaternion)
        np.testing.assert_allclose(
            Rotation.from_rotvec(axis_angle).as_matrix(),
            Rotation.from_quat(quaternion).as_matrix(),
            atol=1e-12,
        )
        relative = relative_rotation_axis_angle(
            Rotation.identity().as_quat(), quaternion
        )
        np.testing.assert_allclose(relative, axis_angle, atol=1e-12)


class TestOnDiskConversion:
    def test_episode_subset_is_written_with_consistent_metadata(self, tmp_path):
        pq = pytest.importorskip("pyarrow.parquet")
        source = tmp_path / "source"
        original = _write_source_dataset(source)
        destination = tmp_path / "converted"

        result = convert_behavior_dataset(
            source,
            destination,
            "delta_joint",
            task_names=["task_zero"],
            episode_indices=[10],
            progress_every=0,
        )
        assert result["episodes"] == 1
        assert result["frames"] == 4
        assert result["tasks"] == ["task_zero"]

        info = json.loads((destination / "meta/info.json").read_text())
        assert info["features"]["action"]["shape"] == [23]
        assert info["total_episodes"] == 1
        assert info["total_frames"] == 4
        assert info["total_tasks"] == 1
        assert len((destination / "meta/episodes.jsonl").read_text().splitlines()) == 1
        manifest = validate_control_mode_dataset(
            destination, "delta_joint", ["task_zero"]
        )
        assert manifest["episodes"] == [10]
        assert manifest["control_mode"] == "delta_joint"

        output_actions = np.stack(
            pq.read_table(
                destination / "data/task-0000/episode_00000010.parquet",
                columns=["action"],
            )["action"].to_pylist()
        )
        expected = convert_episode_actions(*original[10], "delta_joint")
        np.testing.assert_allclose(output_actions, expected, atol=1e-6)

        source_actions = np.stack(
            pq.read_table(
                source / "data/task-0000/episode_00000010.parquet",
                columns=["action"],
            )["action"].to_pylist()
        )
        np.testing.assert_allclose(source_actions, original[10][0], atol=1e-6)

    @pytest.mark.parametrize("control_mode", CONTROL_MODES)
    def test_all_modes_write_data_and_norm_stats(self, tmp_path, control_mode):
        source = tmp_path / "source"
        _write_source_dataset(source)
        dataset = tmp_path / control_mode
        convert_behavior_dataset(
            source,
            dataset,
            control_mode,
            task_names=["task_zero"],
            episode_indices=[10],
            progress_every=0,
        )
        assert validate_control_mode_dataset(dataset, control_mode)

        state_token = "abs_eef" if control_mode.endswith("eef") else "abs_joint"
        payload = compute_norm_stats(
            dataset,
            control_mode,
            state_token,
            model_action_dim=32,
            task_names=["task_zero"],
            episode_indices=[10],
        )
        assert payload["metadata"]["control_mode"] == control_mode
        assert payload["metadata"]["state_token"] == state_token
        assert payload["metadata"]["state_layout"] == state_token
        assert payload["metadata"]["episodes"] == [10]
        for feature in ("state", "actions"):
            for key in ("mean", "std", "q01", "q99"):
                assert len(payload["norm_stats"][feature][key]) == 32
        assert payload["norm_stats"]["actions"]["mean"][
            CONTROL_MODE_ACTION_DIMS[control_mode] :
        ] == [0.0] * (32 - CONTROL_MODE_ACTION_DIMS[control_mode])

        stats_path = write_norm_stats(
            payload, tmp_path / "assets", f"nested/{control_mode}"
        )
        assert stats_path.is_file()
        metadata = validate_norm_stats_asset(
            tmp_path / "assets",
            f"nested/{control_mode}",
            control_mode,
            state_token,
            model_action_dim=32,
        )
        assert metadata["action_env_dim"] == CONTROL_MODE_ACTION_DIMS[control_mode]

    def test_wrong_mode_and_unselected_episode_fail(self, tmp_path):
        source = tmp_path / "source"
        _write_source_dataset(source)
        with pytest.raises(ValueError, match="episode ids"):
            convert_behavior_dataset(
                source,
                tmp_path / "missing",
                "delta_eef",
                task_names=["task_zero"],
                episode_indices=[10010],
                progress_every=0,
            )
        with pytest.raises(ValueError, match="requires"):
            validate_control_mode_dataset(source, "delta_joint")
        (source / "meta/eef_delta_provenance.json").write_text(
            json.dumps({"control_mode": "delta_joint"})
        )
        with pytest.raises(ValueError, match="legacy provenance"):
            validate_control_mode_dataset(source, "abs_joint")

    def test_converter_rejects_ancestor_and_symlinked_nested_destinations(
        self, tmp_path
    ):
        source = tmp_path / "source"
        _write_source_dataset(source)
        source_info = source / "meta/info.json"

        with pytest.raises(ValueError, match="must be disjoint"):
            convert_behavior_dataset(
                source,
                tmp_path,
                "abs_joint",
                overwrite=True,
                progress_every=0,
            )
        assert source_info.is_file()

        alias = tmp_path / "source_alias"
        alias.symlink_to(source, target_is_directory=True)
        with pytest.raises(ValueError, match="must be disjoint"):
            convert_behavior_dataset(
                source,
                alias / "nested",
                "abs_joint",
                overwrite=True,
                progress_every=0,
            )
        assert source_info.is_file()

    def test_converter_overwrite_requires_converter_owned_destination(self, tmp_path):
        source = tmp_path / "source"
        _write_source_dataset(source)
        unrelated = tmp_path / "unrelated"
        unrelated.mkdir()
        sentinel = unrelated / "keep.txt"
        sentinel.write_text("do not delete")

        with pytest.raises(ValueError, match="refusing to overwrite"):
            convert_behavior_dataset(
                source,
                unrelated,
                "abs_joint",
                overwrite=True,
                progress_every=0,
            )
        assert sentinel.read_text() == "do not delete"

        converted = tmp_path / "converted"
        convert_behavior_dataset(
            source,
            converted,
            "abs_joint",
            task_names=["task_zero"],
            episode_indices=[10],
            progress_every=0,
        )
        convert_behavior_dataset(
            source,
            converted,
            "delta_eef",
            task_names=["task_zero"],
            episode_indices=[10],
            overwrite=True,
            progress_every=0,
        )
        manifest = json.loads((converted / "meta/control_mode.json").read_text())
        assert manifest["control_mode"] == "delta_eef"

    def test_converter_rejects_same_width_delta_joint_source(self, tmp_path):
        source = tmp_path / "source"
        _write_source_dataset(source)
        delta_joint = tmp_path / "delta_joint"
        convert_behavior_dataset(
            source,
            delta_joint,
            "delta_joint",
            progress_every=0,
        )

        with pytest.raises(ValueError, match="was built for 'delta_joint'"):
            convert_behavior_dataset(
                delta_joint,
                tmp_path / "mislabeled_abs_joint",
                "abs_joint",
                progress_every=0,
            )
        assert not (tmp_path / "mislabeled_abs_joint").exists()

    def test_legacy_norm_asset_honors_recorded_state_layout(self, tmp_path):
        legacy_asset = tmp_path / "assets/legacy"
        legacy_asset.mkdir(parents=True)
        payload = {
            "norm_stats": {
                feature: {key: [0.0] * 32 for key in ("mean", "std", "q01", "q99")}
                for feature in ("state", "actions")
            },
            "metadata": {"state_order": "align", "action_dim": 32},
        }
        (legacy_asset / "norm_stats.json").write_text(json.dumps(payload))
        validate_norm_stats_asset(tmp_path / "assets", "legacy", "abs_joint", "none")
        validate_norm_stats_asset(
            tmp_path / "assets", "legacy", "abs_joint", "abs_joint"
        )
        for token in ("abs_joint_old", "abs_eef"):
            with pytest.raises(ValueError, match="legacy state layout"):
                validate_norm_stats_asset(
                    tmp_path / "assets", "legacy", "abs_joint", token
                )

    @pytest.mark.parametrize(
        ("asset_token", "run_token"),
        [("none", "abs_joint_old"), ("abs_joint_old", "none")],
    )
    def test_none_and_old_joint_share_continuous_layout(
        self, tmp_path, asset_token, run_token
    ):
        payload = {
            "norm_stats": {
                feature: {key: [0.0] * 32 for key in ("mean", "std", "q01", "q99")}
                for feature in ("state", "actions")
            },
            "metadata": {
                "control_mode": "abs_joint",
                "action_env_dim": 23,
                "model_action_dim": 32,
                "state_token": asset_token,
                "state_layout": "abs_joint_old",
            },
        }
        write_norm_stats(payload, tmp_path / "assets", "shared_layout")
        validate_norm_stats_asset(
            tmp_path / "assets",
            "shared_layout",
            "abs_joint",
            run_token,
            model_action_dim=32,
        )

    def test_canonical_norm_asset_must_cover_all_requested_tasks(self, tmp_path):
        payload = {
            "norm_stats": {
                feature: {key: [0.0] * 32 for key in ("mean", "std", "q01", "q99")}
                for feature in ("state", "actions")
            },
            "metadata": {
                "control_mode": "abs_joint",
                "action_env_dim": 23,
                "model_action_dim": 32,
                "state_token": "none",
                "state_layout": "abs_joint_old",
                "tasks": ["turning_on_radio"],
            },
        }
        write_norm_stats(payload, tmp_path / "assets", "single_task")

        validate_norm_stats_asset(
            tmp_path / "assets",
            "single_task",
            "abs_joint",
            "none",
            model_action_dim=32,
            tasks=["turning_on_radio"],
        )
        with pytest.raises(ValueError, match="does not cover requested tasks"):
            validate_norm_stats_asset(
                tmp_path / "assets",
                "single_task",
                "abs_joint",
                "none",
                model_action_dim=32,
                tasks=["turning_on_radio", "picking_up_trash"],
            )

    def test_cli_scripts_are_executable_and_mode_generic(self):
        repo_root = Path(__file__).resolve().parents[2]
        for script_name in (
            "convert_openpi_control_mode.sh",
            "compute_openpi_norm_stats.sh",
        ):
            path = repo_root / "toolkits/behavior" / script_name
            assert path.is_file()
            assert os.access(path, os.X_OK)
            contents = path.read_text()
            assert '"$@"' in contents
            assert "assets_dir_delta_joint" not in contents
            assert "behavior_dataset_root_abs_eef" not in contents
