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

"""Runtime-label tests without constructing OmniGibson or decoding videos."""

import types

import pytest
import torch

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
    _Repack,
    _sft_collate,
    _validate_task_names,
    create_behavior_sft_data_loader,
)
from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
    partition_chunk_indices,
)
from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    behavior_task_names,
)


def _window(skill_idx, skill, *objects):
    return {
        "skill_idx": skill_idx,
        "skill_description": [skill],
        "object_id": [list(objects)],
    }


def _resolver(annotations, episodes):
    stub = types.SimpleNamespace(
        _subtask_text_cache={},
        meta=types.SimpleNamespace(annotations=annotations, episodes=episodes),
    )
    return stub, BehaviorSftDataset._resolve_subtask_text.__get__(stub)


def test_same_skill_index_resolves_from_each_episode_annotation():
    annotations = {
        20010: {
            "skill_annotation": [
                _window(3, "pick up from", "pillar_candle_89", "floors_ulujpr_0")
            ]
        },
        20020: {
            "skill_annotation": [
                _window(
                    3,
                    "place on next to",
                    "cauldron_92",
                    "floors_ulujpr_0",
                    "coffee_table_koagbh_0",
                )
            ]
        },
    }
    episodes = {
        episode: {"tasks": ["putting_away_Halloween_decorations"]}
        for episode in annotations
    }
    _, resolve = _resolver(annotations, episodes)

    assert resolve(20010, 3) == "pick up pillar candle from floors"
    assert resolve(20020, 3) == ("place cauldron on floors next to coffee table")


def test_resolved_text_is_cached_and_missing_window_fails():
    annotation = {"skill_annotation": [_window(0, "move to", "radio_89")]}
    stub, resolve = _resolver({1: annotation}, {1: {"tasks": ["turning_on_radio"]}})
    assert resolve(1, 0) == "move to radio"
    assert stub._subtask_text_cache[(1, 0)] == "move to radio"
    with pytest.raises(ValueError, match="skill_idx 2"):
        resolve(1, 2)


def test_repack_preserves_main_prompt_and_subtask_response():
    image = torch.zeros(3, 4, 4)
    frame = {
        "observation.images.rgb.head": image,
        "observation.images.rgb.left_wrist": image,
        "observation.images.rgb.right_wrist": image,
        "observation.state": torch.zeros(256),
        "action": torch.zeros(32, 23),
        "task": "turn on the radio",
        "response": "move to radio",
    }
    repacked = _Repack()(frame)
    assert repacked["prompt"] == "turn on the radio"
    assert repacked["response"] == "move to radio"


def _transformed_item(*, with_vlm_masks: bool):
    item = {
        "image": {
            key: torch.zeros(4, 4, 3, dtype=torch.uint8).numpy()
            for key in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        },
        "image_mask": dict.fromkeys(
            ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"), True
        ),
        "state": torch.zeros(32).numpy(),
        "actions": torch.zeros(32, 32).numpy(),
        "tokenized_prompt": torch.arange(8).numpy(),
        "tokenized_prompt_mask": torch.ones(8, dtype=torch.bool).numpy(),
    }
    if with_vlm_masks:
        item.update(
            {
                "token_ar_mask": torch.zeros(8, dtype=torch.bool).numpy(),
                "token_loss_mask": torch.ones(8, dtype=torch.bool).numpy(),
                "token_kv_cache_mask": torch.ones(8, dtype=torch.bool).numpy(),
            }
        )
    return item


def test_collate_includes_all_vlm_masks():
    observation, actions = _sft_collate(
        [_transformed_item(with_vlm_masks=True) for _ in range(2)]
    )
    assert observation.token_ar_mask.shape == (2, 8)
    assert observation.token_loss_mask.shape == (2, 8)
    assert observation.token_kv_cache_mask.shape == (2, 8)
    assert actions.shape == (2, 32, 32)


def test_collate_keeps_vla_masks_absent():
    observation, _ = _sft_collate(
        [_transformed_item(with_vlm_masks=False) for _ in range(2)]
    )
    assert observation.token_ar_mask is None
    assert observation.token_loss_mask is None
    assert observation.token_kv_cache_mask is None


def test_legacy_vla_prompt_and_use_skill_paths_are_preserved():
    direct = types.SimpleNamespace(
        _get_fine_grained_task=lambda item: "main task",
        skill_labels=None,
        use_skill=False,
    )
    item = {}
    BehaviorSftDataset._set_prompt(direct, item)
    assert item == {"task": "main task"}

    skill = types.SimpleNamespace(
        _get_fine_grained_task=lambda item: "main task",
        _get_skill_label=lambda item: "local skill",
        skill_labels={0: "local skill"},
        use_skill=True,
    )
    item = {}
    BehaviorSftDataset._set_prompt(skill, item)
    assert item == {
        "task": "main task",
        "skill_label": "local skill",
        "prompt": "local skill",
    }


def test_vlm_task_name_validation_accepts_all_tasks_and_rejects_typos():
    _validate_task_names(behavior_task_names())
    with pytest.raises(ValueError, match="flying_to_mars"):
        _validate_task_names(["turning_on_radio", "flying_to_mars"])


def test_chunk_partition_covers_chunks_and_can_be_empty():
    covered = {
        index
        for rank in range(4)
        for index in partition_chunk_indices(
            8, rank=rank, world_size=4, worker_id=0, num_workers=1
        )
    }
    assert covered == set(range(8))
    assert (
        partition_chunk_indices(1, rank=3, world_size=4, worker_id=0, num_workers=1)
        == []
    )


@pytest.mark.parametrize("shuffle", [True, False])
def test_empty_rank_partition_falls_back_to_available_chunks(shuffle):
    chunk = (0, 10, 0)
    stub = types.SimpleNamespace(
        chunks=[chunk],
        shuffle=shuffle,
        seed=0,
        _active_chunks=None,
        _dist_rank=3,
        _dist_world_size=4,
    )
    BehaviorSftDataset._select_streaming_chunk(stub)
    assert stub._active_chunks == [chunk]
    assert stub.current_streaming_chunk_idx == 0
    assert stub.current_streaming_frame_idx == 0


def test_outer_loader_never_builds_frame_scale_random_sampler(monkeypatch):
    """Chunk streaming owns shuffle; torch must not permute all frame indices."""

    class _Dataset:
        def __init__(self, **kwargs):
            del kwargs
            self.hf_dataset = range(10)

        def __getitem__(self, index):
            return index

    captured = {}

    def _data_loader(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "rlinf.data.datasets.openpi_pytorch.behavior."
        "behavior_sft_data_loader.BehaviorSftDataset",
        _Dataset,
    )
    monkeypatch.setattr(
        "rlinf.data.datasets.openpi_pytorch.behavior."
        "behavior_sft_data_loader.build_openpi_transforms",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(torch.utils.data, "DataLoader", _data_loader)

    create_behavior_sft_data_loader(
        behavior_dataset_root="/dataset",
        assets_dir="/assets",
        asset_id="behavior",
        model_path="/model",
        config_name="pi05_behavior",
        repo_id="behavior/repo",
        tasks=["turning_on_radio"],
        modalities=["rgb"],
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=1,
        num_workers=0,
        fine_grained_level=1,
        tolerance_s=1e-4,
        shuffle=True,
        seed=42,
        skill_labels=None,
        use_skill=False,
        enable_gap=False,
        allow_left=0,
        allow_right=0,
        dist_rank=0,
        dist_world_size=1,
        mode="vlm_vla",
        hf_cache_dir="/large-cache",
    )

    assert captured["shuffle"] is False
    assert captured["sampler"] is None
