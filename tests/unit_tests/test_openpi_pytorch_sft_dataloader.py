# Copyright (c) 2026, RLinf contributors.
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

import pathlib

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import NormStats
from rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer import (
    PaligemmaTokenizer,
)

# Reference per-frame skill labels for the turning_on_radio task-0000 local-skill
# recipe (== examples/sft/config/env/behavior_skill_library.yaml).
_REFERENCE_SUBTASKS = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]


def _norm_stats():
    stats = NormStats(
        mean=np.zeros(32, dtype=np.float32),
        std=np.ones(32, dtype=np.float32),
        q01=np.zeros(32, dtype=np.float32),
        q99=np.ones(32, dtype=np.float32),
    )
    return {"state": stats, "actions": stats}


def _raw_item():
    image = np.zeros((3, 16, 12), dtype=np.uint8)
    proprio = np.full(256, 0.5, dtype=np.float32)
    proprio[193:195] = 0.25
    proprio[232:234] = 0.25
    return {
        "observation.images.rgb.head": image,
        "observation.images.rgb.left_wrist": image + 1,
        "observation.images.rgb.right_wrist": image + 2,
        "observation.state": proprio,
        "action": np.full((32, 23), 0.5, dtype=np.float32),
        "task": "turning_on_radio",
    }


def test_behavior_sft_transform_and_collate_contract():
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        collate_behavior_sft_items,
    )
    from rlinf.data.datasets.behavior.behavior_sft_transform import (
        BehaviorSftTransform,
        transform_behavior_sft_item,
    )

    transform = BehaviorSftTransform(
        norm_stats=_norm_stats(),
        action_dim=32,
        tokenizer=PaligemmaTokenizer(max_len=200),
    )
    item = transform_behavior_sft_item(_raw_item(), transform)
    observation, actions = collate_behavior_sft_items([item, item])

    assert isinstance(observation, Observation)
    assert tuple(actions.shape) == (2, 32, 32)
    assert actions.dtype == torch.float32
    torch.testing.assert_close(actions[..., :23], torch.zeros(2, 32, 23))
    torch.testing.assert_close(actions[..., 23:], torch.zeros(2, 32, 9))

    assert tuple(observation.state.shape) == (2, 32)
    torch.testing.assert_close(observation.state[:, :23], torch.zeros(2, 23))
    torch.testing.assert_close(observation.state[:, 23:], torch.zeros(2, 9))
    assert tuple(observation.tokenized_prompt.shape) == (2, 200)
    assert tuple(observation.tokenized_prompt_mask.shape) == (2, 200)
    assert observation.tokenized_prompt.dtype == torch.long
    assert observation.tokenized_prompt_mask.dtype == torch.bool

    for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"):
        assert tuple(observation.images[key].shape) == (2, 224, 224, 3)
        assert observation.images[key].dtype == torch.float32
        assert tuple(observation.image_masks[key].shape) == (2,)
        assert observation.image_masks[key].dtype == torch.bool


def test_behavior_sft_transform_rejects_missing_required_field():
    from rlinf.data.datasets.behavior.behavior_sft_transform import (
        BehaviorSftTransform,
        transform_behavior_sft_item,
    )

    raw = _raw_item()
    raw.pop("observation.images.rgb.head")
    with pytest.raises(KeyError, match="observation.images.rgb.head"):
        transform_behavior_sft_item(
            raw,
            BehaviorSftTransform(
                norm_stats=_norm_stats(),
                tokenizer=PaligemmaTokenizer(max_len=200),
            ),
        )


def _openpi_pytorch_sft_worker(openpi_overrides, *, world_size=1, rank=0, data=None):
    """An OPENPI_PYTORCH SFT worker whose actor.model.openpi block is overridable."""
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    data_block = {"train_data_paths": "/data/behavior", "num_workers": 0}
    if data:
        data_block.update(data)
    worker = FSDPVlaSftWorker.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_type": "openpi_pytorch", "openpi": openpi_overrides},
                "micro_batch_size": 32,
                "eval_batch_size": 4,
            },
            "data": data_block,
        }
    )
    worker._world_size = world_size
    worker._rank = rank
    worker.micro_batch_size = 32
    worker.eval_batch_size = 4
    return worker


def test_fsdp_vla_worker_dispatches_openpi_pytorch_dataloader(monkeypatch):
    from rlinf.data.datasets.behavior import behavior_sft_data_loader

    calls = {}

    class _FakeLoader:
        def data_config(self):
            return {"dataset": "behavior_b1k_direct"}

        def __len__(self):
            return 8

    def _fake_loader(**kwargs):
        calls.update(kwargs)
        return _FakeLoader()

    monkeypatch.setattr(
        behavior_sft_data_loader, "create_behavior_sft_data_loader", _fake_loader
    )

    # Norm stats are resolved STRICTLY from YAML assets_dir + asset_id (AC-8): the
    # experiment config supplies the canonical task-0000 asset location, which must
    # propagate verbatim to the loader (no model_path/norm_stats_path fallback).
    worker = _openpi_pytorch_sft_worker(
        {
            "assets_dir": "/data/assets",
            "asset_id": "behavior-1k/2025-challenge-demos",
        },
        world_size=8,
        rank=3,
    )

    loader, data_config = worker.build_dataloader("/data/behavior")
    assert isinstance(loader, _FakeLoader)
    assert data_config == {"dataset": "behavior_b1k_direct"}
    assert calls["behavior_dataset_root"] == "/data/behavior"
    assert calls["assets_dir"] == "/data/assets"
    assert calls["asset_id"] == "behavior-1k/2025-challenge-demos"
    assert calls["repo_id"] == "behavior-1k/2025-challenge-demos"
    assert calls["tasks"] == ["turning_on_radio"]
    assert calls["modalities"] == ["rgb"]
    assert calls["action_dim"] == 32
    assert calls["action_horizon"] == 32
    assert calls["max_token_len"] == 200
    assert calls["batch_size"] == 32
    assert calls["num_workers"] == 0
    assert calls["shuffle"] is True
    # Default (no cfg.data.use_skill) trains on the main-task text (AC-10).
    assert calls["use_skill"] is False


def test_fsdp_vla_worker_use_skill_true_propagates(monkeypatch):
    """cfg.data.use_skill/tasks are the production source of truth: use_skill: true
    propagates the reference skill window recipe to the loader (AC-10)."""
    from rlinf.data.datasets.behavior import behavior_sft_data_loader

    calls = {}

    class _FakeLoader:
        def data_config(self):
            return {"dataset": "behavior_b1k_direct"}

        def __len__(self):
            return 8

    monkeypatch.setattr(
        behavior_sft_data_loader,
        "create_behavior_sft_data_loader",
        lambda **kwargs: calls.update(kwargs) or _FakeLoader(),
    )

    worker = _openpi_pytorch_sft_worker(
        {"assets_dir": "/data/assets", "asset_id": "behavior-1k/2025-challenge-demos"},
        data={
            "use_skill": True,
            "tasks": ["turning_on_radio"],
            "task_subtasks": {"turning_on_radio": _REFERENCE_SUBTASKS},
        },
    )
    worker.build_dataloader("/data/behavior")
    assert calls["use_skill"] is True
    # The skill labels are the REFERENCE subtask list from config, NOT derived.
    assert calls["skill_labels"] == dict(enumerate(_REFERENCE_SUBTASKS))
    # Fixed reference skill recipe (pi05_b1k-task0000_sft_local_skill).
    assert calls["enable_gap"] is True
    assert calls["allow_left"] == 100
    assert calls["allow_right"] == 100
    assert calls["tasks"] == ["turning_on_radio"]


@pytest.mark.parametrize(
    "data",
    [
        # use_skill:true but NO configured subtask labels -> raise (no orchestrator
        # fallback to task text).
        {"use_skill": True, "tasks": ["turning_on_radio"]},
        {
            "use_skill": True,
            "tasks": ["turning_on_radio"],
            "task_subtasks": {"some_other_task": _REFERENCE_SUBTASKS},
        },
        # use_skill:true with more than one task is unsupported (task-0000 recipe).
        {
            "use_skill": True,
            "tasks": ["turning_on_radio", "extra_task"],
            "task_subtasks": {"turning_on_radio": _REFERENCE_SUBTASKS},
        },
    ],
)
def test_sft_builder_use_skill_requires_reference_labels(data):
    """use_skill:true without configured reference labels (or with >1 task) raises
    loudly instead of silently deriving task-text labels (AC-10/AC-12)."""
    worker = _openpi_pytorch_sft_worker(
        {"assets_dir": "/data/assets", "asset_id": "behavior-1k/2025-challenge-demos"},
        data=data,
    )
    with pytest.raises(ValueError, match="use_skill"):
        worker.build_dataloader("/data/behavior")


# Real BEHAVIOR dataset + canonical task-0000 norm stats for the integrated AC-12 gate.
_DATA_ROOT = pathlib.Path("/mnt/public/xzxuan/data/2025-challenge-demos")
_ASSETS_DIR = pathlib.Path("/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets")
_NORM_STATS = _ASSETS_DIR / "behavior-1k/2025-challenge-demos/norm_stats.json"


@pytest.mark.skipif(
    not (_DATA_ROOT.is_dir() and _NORM_STATS.is_file()),
    reason="real BEHAVIOR dataset / canonical task-0000 norm stats not available",
)
def test_use_skill_true_real_loader_emits_reference_skill_prompt():
    """Integrated AC-10/AC-12 gate on the REAL dataset: the production
    `use_skill: true` builder selects a REFERENCE skill label (not the task text) as
    the per-frame prompt, and a built batch is a valid (Observation, actions)."""
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi_pytorch",
                    "num_action_chunks": 32,
                    "openpi": {
                        "assets_dir": str(_ASSETS_DIR),
                        "asset_id": "behavior-1k/2025-challenge-demos",
                        "model_action_dim": 32,
                        "max_token_len": 200,
                    },
                },
                "micro_batch_size": 1,
                "eval_batch_size": 1,
                "seed": 42,
            },
            "data": {
                "train_data_paths": str(_DATA_ROOT),
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": True,
                "task_subtasks": {"turning_on_radio": _REFERENCE_SUBTASKS},
            },
        }
    )
    loader, _ = build_behavior_sft_dataloader(cfg, 1, 0, str(_DATA_ROOT))

    # A raw frame from the underlying dataset carries a reference SKILL label as the
    # prompt (window-resolved), distinct from the full main-task text.
    raw = loader.torch_loader.dataset._dataset[0]
    assert raw["prompt"] in set(_REFERENCE_SUBTASKS)
    assert raw["prompt"] != raw["task"]

    # Iterating yields a well-formed (Observation, actions) batch.
    observation, actions = next(iter(loader))
    assert isinstance(observation, Observation)
    assert tuple(actions.shape)[1:] == (32, 32)
    assert tuple(observation.tokenized_prompt.shape)[1:] == (200,)


@pytest.mark.parametrize(
    ("openpi_overrides", "missing"),
    [
        # Omitted fields.
        ({"asset_id": "behavior-1k/2025-challenge-demos"}, "assets_dir"),
        ({"assets_dir": "/data/assets"}, "asset_id"),
        # Blank / whitespace-only values must be rejected like omitted ones (AC-8):
        # a blank YAML value is not a value and must not fall back to bare stats.
        (
            {"assets_dir": "", "asset_id": "behavior-1k/2025-challenge-demos"},
            "assets_dir",
        ),
        ({"assets_dir": "/data/assets", "asset_id": ""}, "asset_id"),
        ({"assets_dir": "/data/assets", "asset_id": "   "}, "asset_id"),
    ],
)
def test_sft_builder_requires_assets_dir_and_asset_id(openpi_overrides, missing):
    """Missing OR blank openpi.assets_dir/openpi.asset_id fails loudly (no fallback)."""
    worker = _openpi_pytorch_sft_worker(openpi_overrides)
    with pytest.raises(ValueError, match=missing):
        worker.build_dataloader("/data/behavior")
