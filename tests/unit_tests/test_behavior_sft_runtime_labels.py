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

"""Tests for multi-task subtask-supervision runtime label resolution.

These exercise the per-episode subtask-text resolution
(:meth:`BehaviorSftDataset._resolve_subtask_text`) and the multi-task task-name
validation (:func:`_validate_task_names`) without constructing a full streaming
dataset (which needs OmniGibson and the video assets). The resolver is exercised
by binding the unbound method to a light stub carrying only the attributes it
reads, which mirrors exactly what the real object provides at runtime.
"""

import types

import pytest

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
    _validate_task_names,
)
from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
    TASK_NAMES_TO_INDICES,
    BehaviorSftDataset,
)
from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    behavior_task_names,
)


def _make_resolver(annotations, episodes):
    """Bind ``_resolve_subtask_text`` to a stub with the attributes it reads."""
    stub = types.SimpleNamespace(
        _subtask_text_cache={},
        meta=types.SimpleNamespace(annotations=annotations, episodes=episodes),
    )
    return stub, BehaviorSftDataset._resolve_subtask_text.__get__(stub)


def _window(skill_idx, skill_description, *slots):
    return {
        "skill_idx": skill_idx,
        "skill_description": [skill_description],
        "object_id": [list(slots)],
    }


class TestBackwardCompatAnchor:
    """AC-8: the old task-0000 recipe yields the same 4 labels under runtime resolution."""

    def test_turning_on_radio_labels_unchanged(self):
        # The real task-0000 skill sequence (all episodes share it).
        annotation = {
            "skill_annotation": [
                _window(0, "move to", "radio_89"),
                _window(1, "pick up from", "radio_89", "coffee_table_koagbh_0"),
                _window(2, "press", "radio_89"),
                _window(3, "place on", "radio_89", "coffee_table_koagbh_0"),
            ]
        }
        episodes = {0: {"tasks": ["turning_on_radio"]}}
        _, resolve = _make_resolver({0: annotation}, episodes)
        labels = [resolve(0, i) for i in range(4)]
        assert labels == [
            "move to radio",
            "pick up radio from coffee table",
            "press radio",
            "place radio on coffee table",
        ]


class TestVariableSequenceCorrectness:
    """AC-5: the same skill_idx in two episodes resolves to each episode's own label.

    Mirrors real task-0002 episodes 20010 and 20020, where skill_idx=3 is
    "pick up from" in one and "place on next to" in the other -- a case a single
    static per-task label list cannot represent.
    """

    def _annotations(self):
        ep_a = {  # episode 20010: skill_idx=3 is a pick-up
            "skill_annotation": [
                _window(3, "pick up from", "pillar_candle_89", "floors_ulujpr_0"),
            ]
        }
        ep_b = {  # episode 20020: skill_idx=3 is a place-next-to
            "skill_annotation": [
                _window(
                    3,
                    "place on next to",
                    "cauldron_92",
                    "floors_ulujpr_0",
                    "coffee_table_koagbh_0",
                ),
            ]
        }
        return {20010: ep_a, 20020: ep_b}

    def test_same_index_resolves_per_episode(self):
        episodes = {
            20010: {"tasks": ["putting_away_Halloween_decorations"]},
            20020: {"tasks": ["putting_away_Halloween_decorations"]},
        }
        # Note: skill_idx=3 lives at position 0 of each single-window fixture; use
        # the real per-episode window list so the index maps to that episode's own
        # entry. Here each fixture holds exactly the skill_idx=3 window.
        _, resolve = _make_resolver(self._annotations(), episodes)
        label_a = resolve(20010, 0)
        label_b = resolve(20020, 0)
        assert label_a == "pick up pillar candle from floors"
        assert label_b == "place cauldron on floors next to coffee table"
        assert label_a != label_b

    def test_static_shared_list_would_be_wrong(self):
        # Demonstrate why a static per-task list fails: one shared label array
        # cannot equal both episodes' distinct texts at the same index.
        _, resolve = _make_resolver(
            self._annotations(),
            {
                20010: {"tasks": ["putting_away_Halloween_decorations"]},
                20020: {"tasks": ["putting_away_Halloween_decorations"]},
            },
        )
        shared_static_label = resolve(20010, 0)  # what a fixed list would cache
        # The other episode's true label differs from the shared static one.
        assert resolve(20020, 0) != shared_static_label


class TestResolverFailFast:
    """AC-4: missing annotation / out-of-range skill_idx fail loudly with context."""

    def test_missing_annotation_raises(self):
        _, resolve = _make_resolver({}, {5: {"tasks": ["turning_on_radio"]}})
        with pytest.raises(ValueError) as exc:
            resolve(5, 0)
        assert "5" in str(exc.value)

    def test_out_of_range_skill_idx_raises(self):
        annotation = {"skill_annotation": [_window(0, "move to", "radio_89")]}
        _, resolve = _make_resolver(
            {7: annotation}, {7: {"tasks": ["turning_on_radio"]}}
        )
        with pytest.raises(ValueError) as exc:
            resolve(7, 5)
        message = str(exc.value)
        assert "7" in message
        assert "5" in message

    def test_cache_returns_same_text(self):
        annotation = {"skill_annotation": [_window(0, "move to", "radio_89")]}
        stub, resolve = _make_resolver(
            {1: annotation}, {1: {"tasks": ["turning_on_radio"]}}
        )
        first = resolve(1, 0)
        assert (1, 0) in stub._subtask_text_cache
        assert resolve(1, 0) == first


class TestValidateTaskNames:
    """AC-2, AC-4, AC-8: explicit, known, non-empty task lists at level 1."""

    def test_all_50_names_accepted(self):
        _validate_task_names(behavior_task_names())  # no raise

    def test_two_tasks_accepted(self):
        _validate_task_names(["turning_on_radio", "picking_up_trash"])  # no raise

    def test_empty_list_rejected(self):
        with pytest.raises(ValueError) as exc:
            _validate_task_names([])
        assert (
            "empty" in str(exc.value).lower() or "non-empty" in str(exc.value).lower()
        )

    def test_unknown_name_rejected_by_name(self):
        with pytest.raises(ValueError) as exc:
            _validate_task_names(["turning_on_radio", "flying_to_mars"])
        assert "flying_to_mars" in str(exc.value)

    def test_registry_is_single_source_of_truth(self):
        # The helper's notion of "known" is exactly TASK_NAMES_TO_INDICES.
        assert set(behavior_task_names()) == set(TASK_NAMES_TO_INDICES)
