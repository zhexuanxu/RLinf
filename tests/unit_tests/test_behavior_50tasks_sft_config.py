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

"""Tests for the behavior_50tasks_pi05_vlm_vla SFT config.

Verifies the 50-task config's step counts, mode/level/state-order, norm-stat
asset wiring, and that its task list is exactly the authoritative
``TASK_NAMES_TO_INDICES`` order (AC-1, AC-2, AC-6 wiring).
"""

import os

import pytest
from hydra import compose, initialize_config_dir

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
    TASK_NAMES_TO_INDICES,
)
from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    behavior_task_names,
)

_CONFIG_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sft", "config")
)
_CONFIG_NAME = "behavior_50tasks_pi05_vlm_vla"


@pytest.fixture(scope="module")
def cfg():
    # EMBODIED_PATH feeds the config's Hydra searchpath (file://${oc.env:...}).
    os.environ.setdefault(
        "EMBODIED_PATH",
        os.path.abspath(os.path.join(_CONFIG_DIR, "..")),
    )
    with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):
        return compose(config_name=_CONFIG_NAME)


class TestStepCountsAndMode:
    """AC-1: step counts and the vlm_vla / level-1 / align knobs."""

    def test_step_counts_are_50000(self, cfg):
        assert cfg.runner.max_steps == 50000
        assert cfg.actor.optim.total_training_steps == 50000

    def test_not_the_old_step_count(self, cfg):
        # Negative: the derived-from config must not keep the 30000 default.
        assert cfg.runner.max_steps != 30000

    def test_mode_level_state_order(self, cfg):
        assert cfg.actor.model.openpi.mode == "vlm_vla"
        assert cfg.data.fine_grained_level == 1
        assert cfg.actor.model.openpi.state_order == "align"


class TestTaskList:
    """AC-2: explicit 50 names, index-ordered, consistent with the map."""

    def test_lists_all_50_index_ordered(self, cfg):
        tasks = list(cfg.data.tasks)
        assert tasks == behavior_task_names()
        assert len(tasks) == 50
        assert len(set(tasks)) == 50

    def test_every_name_is_known(self, cfg):
        for name in cfg.data.tasks:
            assert name in TASK_NAMES_TO_INDICES

    def test_first_and_last(self, cfg):
        assert cfg.data.tasks[0] == "turning_on_radio"
        assert cfg.data.tasks[-1] == "make_pizza"

    def test_no_inline_task_subtasks(self, cfg):
        # Runtime resolution supersedes the old inline per-task label list.
        assert "task_subtasks" not in cfg.data


class TestNormStatWiring:
    """AC-6 (config side): the config points at the 50-task align asset."""

    def test_asset_points_at_50tasks_reorder(self, cfg):
        assert cfg.actor.model.openpi.asset_id == "50tasks_reorder"
        assert not str(cfg.actor.model.openpi.asset_id).startswith("turn_on_radio")

    def test_assets_dir_set(self, cfg):
        assert str(cfg.actor.model.openpi.assets_dir).endswith("outputs/norm_stats")
