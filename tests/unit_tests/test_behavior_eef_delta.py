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

"""Config-surface + converter-math contract for the delta-EEF control mode.

Fast, Isaac-free coverage of the ``control_mode`` machinery:
- the canonical mode/dim constants;
- the eval controller mapping + dimensional-consistency checks in
  ``validate_embodied_cfg`` (arms become IK ``pose_delta_ori`` for delta mode);
- the converter's exact-rotation labeling and the 21-dim action layout;
- the norm-stats manifest / control-mode guard.

The "assert the real R1Pro env action space is 21 and ``env.step`` accepts a
21-dim vector" check requires the Isaac runtime and lives in the environment-
gated smoke script, not here.
"""

import numpy as np
import pytest

from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    CONTROL_MODE_ACTION_ENV_DIM,
    CONTROL_MODES,
)


class TestControlModeConstants:
    def test_modes_and_dims(self):
        assert CONTROL_MODES == ("joint_absolute", "eef_delta_pose")
        assert CONTROL_MODE_ACTION_ENV_DIM == {
            "joint_absolute": 23,
            "eef_delta_pose": 21,
        }

    def test_delta_arm_dims_are_six_each(self):
        # 21 = base(3) + trunk(4) + arm_left_eef(6) + gripper(1)
        #      + arm_right_eef(6) + gripper(1)
        fixed = 3 + 4 + 1 + 1
        assert CONTROL_MODE_ACTION_ENV_DIM["eef_delta_pose"] - fixed == 12  # 6 + 6


class TestNormStatsManifestGuard:
    def _write_asset(self, tmp_path, control_mode, action_env_dim):
        import json

        d = tmp_path / "asset"
        d.mkdir()
        payload = {
            "norm_stats": {
                "state": {k: [0.0] * 32 for k in ("mean", "std", "q01", "q99")},
                "actions": {k: [0.0] * 32 for k in ("mean", "std", "q01", "q99")},
            }
        }
        if control_mode is not None:
            payload["metadata"] = {
                "control_mode": control_mode,
                "action_env_dim": action_env_dim,
            }
        (d / "norm_stats.json").write_text(json.dumps(payload))
        return str(tmp_path), "asset"

    def test_delta_asset_accepts_matching_run(self, tmp_path):
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, "eef_delta_pose", 21)
        validate_norm_stats_for_control_mode(ad, aid, "eef_delta_pose", 21)

    def test_legacy_asset_without_manifest_passes(self, tmp_path):
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, None, None)
        validate_norm_stats_for_control_mode(ad, aid, "joint_absolute", 23)

    @pytest.mark.parametrize(
        ("asset_mode", "asset_dim", "run_mode", "run_dim"),
        [
            ("eef_delta_pose", 21, "joint_absolute", 23),
            ("joint_absolute", 23, "eef_delta_pose", 21),
            ("eef_delta_pose", 23, "eef_delta_pose", 21),
        ],
    )
    def test_mismatch_rejected(self, tmp_path, asset_mode, asset_dim, run_mode, run_dim):
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, asset_mode, asset_dim)
        with pytest.raises(ValueError):
            validate_norm_stats_for_control_mode(ad, aid, run_mode, run_dim)

    def test_delta_run_rejects_manifestless_legacy_asset(self, tmp_path):
        # A manifest-less asset is a legacy joint stats file; delta mode must
        # refuse it (not silently normalize 21-dim actions against 23-dim stats).
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, None, None)
        with pytest.raises(ValueError):
            validate_norm_stats_for_control_mode(ad, aid, "eef_delta_pose", 21)


class TestControlModeResolver:
    def _cvt(self):
        import importlib.util as u

        spec = u.spec_from_file_location(
            "cvt_res",
            "rlinf/data/datasets/openpi_pytorch/behavior/convert_to_eef_delta.py",
        )
        m = u.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_mode_selects_suffixed_fields(self):
        from omegaconf import OmegaConf

        cvt = self._cvt()
        data = OmegaConf.create(
            {
                "behavior_dataset_root": "/orig",
                "train_data_paths": "/orig",
                "behavior_dataset_root_eef_delta": "/delta",
                "train_data_paths_eef_delta": "/delta",
            }
        )
        op = OmegaConf.create(
            {
                "assets_dir": "/a",
                "asset_id": "joint",
                "assets_dir_eef_delta": "/a",
                "asset_id_eef_delta": "delta_asset",
            }
        )
        r = cvt.resolve_behavior_paths(data, op, "eef_delta_pose")
        assert r["behavior_dataset_root"] == "/delta"
        assert r["asset_id"] == "delta_asset"
        rj = cvt.resolve_behavior_paths(data, op, "joint_absolute")
        assert rj["behavior_dataset_root"] == "/orig"
        assert rj["asset_id"] == "joint"

    def test_missing_delta_fields_error_for_delta_mode(self):
        # eef_delta_pose REQUIRES the *_eef_delta fields; a config with only base
        # fields must fail loudly rather than silently training delta actions
        # against the joint dataset/stats.
        import pytest
        from omegaconf import OmegaConf

        cvt = self._cvt()
        data = OmegaConf.create(
            {"behavior_dataset_root": "/only", "train_data_paths": "/only"}
        )
        op = OmegaConf.create({"assets_dir": "/a", "asset_id": "only_asset"})
        with pytest.raises(ValueError):
            cvt.resolve_behavior_paths(data, op, "eef_delta_pose")

    def test_joint_mode_uses_base_fields(self):
        from omegaconf import OmegaConf

        cvt = self._cvt()
        data = OmegaConf.create(
            {
                "behavior_dataset_root": "/orig",
                "train_data_paths": "/orig",
                "behavior_dataset_root_eef_delta": "/delta",
                "train_data_paths_eef_delta": "/delta",
            }
        )
        op = OmegaConf.create(
            {"assets_dir": "/a", "asset_id": "joint", "asset_id_eef_delta": "delta_asset"}
        )
        r = cvt.resolve_behavior_paths(data, op, "joint_absolute")
        assert r["behavior_dataset_root"] == "/orig"
        assert r["asset_id"] == "joint"


class TestNormStatsManifestTaskList:
    """The all-50 stats manifest must record the real task list (not null) when
    aggregated from a converted dataset via --from-episodes-stats (no --tasks)."""

    def test_dataset_task_names_reads_tasks_jsonl(self, tmp_path):
        import json

        from rlinf.data.datasets.openpi_pytorch.behavior.compute_norm_stats import (
            _dataset_task_names,
        )

        meta = tmp_path / "meta"
        meta.mkdir()
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task_name": "turning_on_radio"}) + "\n"
            + json.dumps({"task_index": 1, "task_name": "picking_up_trash"}) + "\n"
        )
        assert _dataset_task_names(str(tmp_path)) == [
            "turning_on_radio",
            "picking_up_trash",
        ]

    def test_dataset_task_names_missing_returns_none(self, tmp_path):
        from rlinf.data.datasets.openpi_pytorch.behavior.compute_norm_stats import (
            _dataset_task_names,
        )

        assert _dataset_task_names(str(tmp_path)) is None


class TestConverterHelpers:
    """Guards against the resolve_behavior_paths edit that once shadowed
    _task_indices_for_names (leaving it undefined -> convert_dataset NameError)."""

    def _cvt(self):
        import importlib.util as u

        spec = u.spec_from_file_location(
            "cvt_help",
            "rlinf/data/datasets/openpi_pytorch/behavior/convert_to_eef_delta.py",
        )
        m = u.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_task_indices_for_names_is_module_level_and_callable(self, tmp_path):
        import json

        cvt = self._cvt()
        assert callable(getattr(cvt, "_task_indices_for_names", None))
        meta = tmp_path / "meta"
        meta.mkdir()
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task_name": "turning_on_radio"}) + "\n"
            + json.dumps({"task_index": 1, "task_name": "picking_up_trash"}) + "\n"
        )
        idx = cvt._task_indices_for_names(str(tmp_path), ["turning_on_radio"])
        assert idx == {"turning_on_radio": 0}
        import pytest

        with pytest.raises(ValueError):
            cvt._task_indices_for_names(str(tmp_path), ["no_such_task"])


class TestPaddedStatValidation:
    def _write_asset(self, tmp_path, meaningful_len, padded_len, model_dim):
        import json

        d = tmp_path / "asset"
        d.mkdir()
        arr = [1.0] * meaningful_len + [0.0] * (padded_len - meaningful_len)
        payload = {
            "norm_stats": {
                "state": {k: list(arr) for k in ("mean", "std", "q01", "q99")},
                "actions": {k: list(arr) for k in ("mean", "std", "q01", "q99")},
            },
            "metadata": {
                "control_mode": "eef_delta_pose",
                "action_env_dim": 21,
                "model_action_dim": model_dim,
            },
        }
        (d / "norm_stats.json").write_text(json.dumps(payload))
        return str(tmp_path), "asset"

    def test_correct_padding_passes(self, tmp_path):
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, 21, 32, 32)
        validate_norm_stats_for_control_mode(
            ad, aid, "eef_delta_pose", 21, model_action_dim=32
        )

    def test_wrong_model_action_dim_rejected(self, tmp_path):
        import pytest

        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        ad, aid = self._write_asset(tmp_path, 21, 32, 64)
        with pytest.raises(ValueError):
            validate_norm_stats_for_control_mode(
                ad, aid, "eef_delta_pose", 21, model_action_dim=32
            )

    def test_too_short_arrays_rejected(self, tmp_path):
        import pytest

        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            validate_norm_stats_for_control_mode,
        )

        # arrays padded to 21, but the run pads to 32
        ad, aid = self._write_asset(tmp_path, 21, 21, 32)
        with pytest.raises(ValueError):
            validate_norm_stats_for_control_mode(
                ad, aid, "eef_delta_pose", 21, model_action_dim=32
            )


class TestConverterMath:
    def _cvt(self):
        import importlib.util as u

        spec = u.spec_from_file_location(
            "cvt_mod",
            "rlinf/data/datasets/openpi_pytorch/behavior/convert_to_eef_delta.py",
        )
        m = u.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def test_relative_rotation_is_exact_left_multiply(self):
        cvt = self._cvt()
        rng = np.random.default_rng(0)
        for _ in range(200):
            # random unit quats (xyzw)
            qc = rng.normal(size=4)
            qc /= np.linalg.norm(qc)
            qt = rng.normal(size=4)
            qt /= np.linalg.norm(qt)
            dori = cvt.relative_rotation_axisangle(qc, qt)
            # R_delta @ R_cur must equal R_tgt
            r_delta = cvt.quat2mat_xyzw(cvt.axisangle_to_quat(dori)) if hasattr(
                cvt, "axisangle_to_quat"
            ) else _rotvec_to_mat(dori)
            applied = r_delta @ cvt.quat2mat_xyzw(qc)
            target = cvt.quat2mat_xyzw(qt)
            assert np.abs(applied - target).max() < 1e-6

    def test_action_layout_preserves_fixed_slices(self):
        cvt = self._cvt()

        # identity FK: EEF == a fixed pose regardless of q, so all deltas are 0
        class _IdFK:
            def __call__(self, q):
                q = np.asarray(q, dtype=float)
                n = 1 if q.ndim == 1 else q.shape[0]
                pos = np.zeros((n, 3))
                quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
                if q.ndim == 1:
                    return {
                        "left": cvt.ArmEefPose(pos[0], quat[0]),
                        "right": cvt.ArmEefPose(pos[0], quat[0]),
                    }
                return {"left": (pos, quat), "right": (pos, quat)}

        rng = np.random.default_rng(1)
        act23 = rng.normal(size=23)
        st256 = rng.normal(size=256)
        out = cvt.action_joint_to_eef_delta(act23, st256, _IdFK())
        assert out.shape == (21,)
        # base + trunk copied through
        assert np.allclose(out[0:3], act23[0:3])
        assert np.allclose(out[3:7], act23[3:7])
        # grippers copied to new positions 13 and 20
        assert out[13] == act23[14]
        assert out[20] == act23[22]
        # identity FK -> zero arm deltas
        assert np.allclose(out[7:13], 0.0)
        assert np.allclose(out[14:20], 0.0)


def _rotvec_to_mat(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-12:
        return np.eye(3)
    k = rotvec / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


class TestEvalControllerMappingAndDecode:
    """AC-6 without Isaac: exercise the REAL validate_embodied_cfg controller
    mapping (apply_behavior_control_mode) and the BehaviorEvalProcessor decode
    [B, chunk, 32] -> [B, chunk, 21]. The live env.step is proven separately by
    the Isaac EVAL_SMOKE_OK run."""

    def _behavior_cfg(self, control_mode, action_dim):
        from omegaconf import OmegaConf

        joint_arm = {
            "name": "JointController",
            "motor_type": "position",
            "use_delta_commands": False,
            "pos_kp": 150,
        }
        robot = {
            "type": "R1Pro",
            "controller_config": {
                "base": {"name": "HolonomicBaseJointController"},
                "trunk": {"name": "JointController", "use_delta_commands": False},
                "arm_left": dict(joint_arm),
                "arm_right": dict(joint_arm),
                "gripper_left": {"name": "MultiFingerGripperController"},
                "gripper_right": {"name": "MultiFingerGripperController"},
            },
        }
        env_split = {"omni_config": {"robots": [robot]}}
        return OmegaConf.create(
            {
                "actor": {
                    "model": {
                        "action_dim": action_dim,
                        "openpi": {
                            "control_mode": control_mode,
                            "action_env_dim": action_dim,
                        },
                    }
                },
                "env": {"train": dict(env_split), "eval": dict(env_split)},
            }
        )

    def test_delta_maps_arms_to_ik_pose_delta_ori(self):
        from rlinf.config import apply_behavior_control_mode

        cfg = self._behavior_cfg("eef_delta_pose", 21)
        apply_behavior_control_mode(cfg)
        for split in ("train", "eval"):
            cc = cfg.env[split].omni_config.robots[0].controller_config
            for arm in ("arm_left", "arm_right"):
                assert cc[arm].name == "InverseKinematicsController"
                assert cc[arm].mode == "pose_delta_ori"
                assert cc[arm].command_input_limits is None
                assert "motor_type" not in cc[arm]  # no stale JointController keys
            # base/trunk/grippers unchanged
            assert cc.trunk.name == "JointController"

    def test_joint_leaves_arms_and_delta_dim_mismatch_rejected(self):
        import pytest

        from rlinf.config import apply_behavior_control_mode

        cfg = self._behavior_cfg("joint_absolute", 23)
        apply_behavior_control_mode(cfg)
        assert (
            cfg.env.eval.omni_config.robots[0].controller_config.arm_left.name
            == "JointController"
        )
        # delta mode with action_dim 23 must be rejected
        with pytest.raises(ValueError):
            apply_behavior_control_mode(self._behavior_cfg("eef_delta_pose", 23))

    def test_behavior_eval_processor_decodes_to_21(self):
        # Decode a normalized [B, chunk, 32] model action through the real
        # BehaviorEvalProcessor and assert the env action is [B, chunk, 21].
        import torch

        from rlinf.data.datasets.openpi_pytorch.behavior.processing import (
            BehaviorEvalProcessor,
        )
        from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
            load_norm_stats,
        )

        asset = "/mnt/public/xzxuan/repos/RLinf/outputs/norm_stats"
        try:
            norm_stats = load_norm_stats(asset, "turn_on_radio_eef_delta")
        except FileNotFoundError:
            import pytest

            pytest.skip("delta norm stats asset not present")

        proc = BehaviorEvalProcessor(
            norm_stats,
            tokenizer=None,
            action_chunk=32,
            action_env_dim=21,
            model_action_dim=32,
            state_order="align",
        )
        B, chunk = 2, 32
        model_actions = torch.zeros(B, chunk, 32, dtype=torch.float32)
        out = proc.postprocess_actions(model_actions)
        assert out.shape == (B, chunk, 21), out.shape
