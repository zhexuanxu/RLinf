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

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_replay_worker import (
    BehaviorReplayRolloutWorker,
)
from rlinf.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
)
from rlinf.envs.behavior.replay import (
    BehaviorReplayActionSource,
    resolve_replay_episode,
)


def _write_jsonlines(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _make_dataset(
    root: Path,
    *,
    episode_index: int = 7,
    control_mode: str = "abs_joint",
) -> np.ndarray:
    import pyarrow as pa
    import pyarrow.parquet as pq

    action_dim = 21 if control_mode.endswith("eef") else 23
    actions = np.arange(3 * action_dim, dtype=np.float32).reshape(3, action_dim)
    states = np.arange(3 * 256, dtype=np.float32).reshape(3, 256)
    (root / "meta").mkdir(parents=True)
    episode_dir = root / "data/chunk-000"
    episode_dir.mkdir(parents=True)
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "chunks_size": 100,
                "features": {"action": {"shape": [action_dim]}},
            }
        ),
        encoding="utf-8",
    )
    _write_jsonlines(
        root / "meta/tasks.jsonl",
        [
            {
                "task_index": 0,
                "task_name": "turning_on_radio",
                "task": "Turn on the radio.",
            }
        ],
    )
    _write_jsonlines(
        root / "meta/episodes.jsonl",
        [
            {
                "episode_index": episode_index,
                "tasks": ["Turn on the radio."],
                "length": len(actions),
            }
        ],
    )
    table = pa.table(
        {
            "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
            "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
        }
    )
    pq.write_table(
        table,
        episode_dir / f"episode_{episode_index:06d}.parquet",
    )
    if control_mode != "abs_joint":
        (root / "meta/control_mode.json").write_text(
            json.dumps(
                {
                    "control_mode": control_mode,
                    "action_env_dim": action_dim,
                    "tasks": ["turning_on_radio"],
                }
            ),
            encoding="utf-8",
        )
    return actions


def test_replay_loads_explicit_episode_and_holds_last_action(tmp_path):
    expected = _make_dataset(tmp_path)
    source = BehaviorReplayActionSource(
        tmp_path,
        control_mode="abs_joint",
        num_action_chunks=2,
        num_envs=2,
        activity_name="turning_on_radio",
    )

    actions = source.load_episode(7)
    chunks = source.episode_to_chunks(actions, n_eval_chunk_steps=2)

    np.testing.assert_array_equal(actions, expected)
    assert chunks.shape == (2, 2, 2, 23)
    np.testing.assert_array_equal(chunks[1, 0, 0], expected[-1])
    np.testing.assert_array_equal(chunks[1, 0, 1, :3], np.zeros(3))
    np.testing.assert_array_equal(chunks[1, 0, 1, 3:], expected[-1, 3:])
    np.testing.assert_array_equal(chunks[:, 0], chunks[:, 1])


def test_replay_validates_episode_activity_and_empty_actions(tmp_path):
    _make_dataset(tmp_path)
    source = BehaviorReplayActionSource(
        tmp_path,
        control_mode="abs_joint",
        num_action_chunks=2,
        num_envs=1,
        activity_name="turning_on_radio",
    )
    with pytest.raises(ValueError, match="no episode_index=70"):
        source.load_episode(70)
    with pytest.raises(ValueError, match="empty action sequence"):
        source.episode_to_chunks(np.empty((0, 23), dtype=np.float32), 1)


def test_resolve_replay_episode_requires_fixed_matching_instance():
    replay = {"activity_instance_id": 4, "episode_index": 123}
    task = {
        "activity_instance_id": 4,
        "instance_resample_mode": "disabled",
        "online_object_sampling": False,
    }
    assert resolve_replay_episode(replay, task).episode_index == 123

    with pytest.raises(ValueError, match="instance mismatch"):
        resolve_replay_episode(replay, {**task, "activity_instance_id": 5})
    with pytest.raises(ValueError, match="instance_resample_mode=disabled"):
        resolve_replay_episode(replay, {**task, "instance_resample_mode": "offline"})


def test_replay_worker_preserves_eval_channel_lifecycle():
    class _ReceiveHandle:
        async def async_wait(self):
            return {"obs": "ignored"}

    worker = object.__new__(BehaviorReplayRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"group_name": "EnvGroup"}})
    worker.eval_rollout_epoch = 2
    worker.eval_batch_size = 1
    worker.n_eval_chunk_steps = 2
    worker._replay_chunks = np.arange(12, dtype=np.float32).reshape(2, 1, 2, 3)
    receives = []
    sends = []

    def recv_from(**kwargs):
        receives.append(kwargs)
        return _ReceiveHandle()

    def send_to(**kwargs):
        sends.append(kwargs)

    worker.recv_from = recv_from
    worker.send_to = send_to
    asyncio.run(worker.evaluate(object(), object()))

    assert len(receives) == len(sends) == 4
    assert [send["route_key"] for send in sends] == [0, 0, 0, 0]
    assert all(send["tag"] == "eval_rollout_results" for send in sends)
    assert all(isinstance(send["data"], torch.Tensor) for send in sends)
    torch.testing.assert_close(sends[0]["data"], sends[2]["data"])
    torch.testing.assert_close(sends[1]["data"], sends[3]["data"])


def test_cached_instance_loader_constructs_from_base_template():
    omni_cfg = OmegaConf.create(
        {
            "task": {
                "activity_name": "turning_on_radio",
                "activity_instance_id": 7,
            }
        }
    )
    loader = ActivityInstanceLoader(
        omni_cfg=omni_cfg,
        activity_name="turning_on_radio",
        activity_instance_id=7,
        instance_resample_mode="disabled",
        activity_instances=(
            ActivityInstanceFile(
                instance_id=7,
                path="/cache/turning_on_radio_0_7_template-tro_state.json",
                file_format="tro_state",
            ),
        ),
    )

    initial_cfg = loader.build_initial_omni_cfg()

    assert initial_cfg.task.activity_instance_id == 0
    assert loader.activity_instance_id == 7
    assert loader.omni_cfg.task.activity_instance_id == 7


def test_single_replay_yaml_uses_generic_explicit_selectors():
    repo_root = Path(__file__).resolve().parents[2]
    replay_configs = sorted((repo_root / "evaluations/behavior").glob("*replay*.yaml"))
    assert [path.name for path in replay_configs] == [
        "behavior_openpi_pi05_pytorch_replay.yaml"
    ]
    config_text = replay_configs[0].read_text(encoding="utf-8")
    cfg = OmegaConf.load(replay_configs[0])

    assert cfg.env.eval.replay.enabled
    assert cfg.env.eval.replay.dataset_root
    assert cfg.env.eval.replay.activity_instance_id == 1
    assert cfg.env.eval.replay.episode_index == 10
    assert cfg.env.eval.omni_config.task.activity_instance_id == 1
    assert cfg.env.eval.omni_config.task.instance_resample_mode == "disabled"
    assert cfg.rollout.model.openpi.control_mode == "abs_joint"
    assert cfg.rollout.model.openpi.state_token == "none"
    assert "model_path" not in config_text
    assert "eval_processor" not in config_text
    assert "episode_stride" not in config_text
    assert "assets_dir_" not in config_text
    assert "asset_id_" not in config_text

    launcher_text = (repo_root / "toolkits/behavior/run_behavior_replay.sh").read_text(
        encoding="utf-8"
    )
    assert (
        '"env.eval.replay.activity_instance_id=${ACTIVITY_INSTANCE_ID}"'
        in launcher_text
    )
    assert (
        '"env.eval.omni_config.task.activity_instance_id=${ACTIVITY_INSTANCE_ID}"'
        in launcher_text
    )
    assert (
        'RLINF_EVAL_ENTRYPOINT="${REPO_ROOT}/evaluations/behavior/'
        'replay_embodied_agent.py"' in launcher_text
    )
    eval_launcher_text = (repo_root / "evaluations/run_eval.sh").read_text(
        encoding="utf-8"
    )
    assert (
        'SRC_FILE="${RLINF_EVAL_ENTRYPOINT:-${EVALUATIONS_PATH}/'
        'eval_embodied_agent.py}"' in eval_launcher_text
    )

    eval_entry = repo_root / "evaluations/eval_embodied_agent.py"
    replay_entry = repo_root / "evaluations/behavior/replay_embodied_agent.py"
    assert "BehaviorReplayRolloutWorker" not in eval_entry.read_text(encoding="utf-8")
    replay_entry_text = replay_entry.read_text(encoding="utf-8")
    assert "BehaviorReplayRolloutWorker.create_group(cfg)" in replay_entry_text
    assert (
        "rlinf.data.datasets.openpi_pytorch.behavior.behavior_replay_worker"
        in replay_entry_text
    )
