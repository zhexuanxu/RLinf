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

"""Tests for BEHAVIOR annotation-to-language conversion."""

import pytest

from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    behavior_task_names,
    clean_object_name,
    entry_to_subtask_text,
)


def _entry(skill, *objects):
    return {"skill_description": [skill], "object_id": [list(objects)]}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("coffee_table_koagbh_0", "coffee table"),
        ("beer_bottle_267", "beer bottle"),
        ("half_head_cabbage_212_1", "half head cabbage"),
        ("diced__chili", "diced chili"),
        ("half-log-176-0", "half log"),
        ("robot", "robot"),
    ],
)
def test_object_names_are_cleaned_without_losing_categories(raw, expected):
    assert clean_object_name(raw) == expected


def test_task_zero_anchor_language():
    entries = [
        _entry("move to", "radio_89"),
        _entry("pick up from", "radio_89", "coffee_table_koagbh_0"),
        _entry("press", "radio_89"),
        _entry("place on", "radio_89", "coffee_table_koagbh_0"),
    ]
    assert [entry_to_subtask_text(entry) for entry in entries] == [
        "move to radio",
        "pick up radio from coffee table",
        "press radio",
        "place radio on coffee table",
    ]


def test_list_slots_are_deduplicated_and_joined():
    entry = {
        "skill_description": ["pour"],
        "object_id": [
            [
                ["candy_cane_224", "candy_cane_225", "wreath_227"],
                "wicker_basket_218",
                "sofa_lugrhk_1",
            ]
        ],
    }
    assert (
        entry_to_subtask_text(entry) == "pour candy cane and wreath into wicker basket"
    )


def test_unknown_template_includes_episode_context():
    with pytest.raises(ValueError, match="episode 42.*teleport"):
        entry_to_subtask_text(
            _entry("teleport", "radio_89"),
            task_name="turning_on_radio",
            episode_id=42,
            skill_idx=3,
        )


def test_task_registry_is_complete_and_index_ordered():
    names = behavior_task_names()
    assert len(names) == len(set(names)) == 50
    assert names[0] == "turning_on_radio"
    assert names[-1] == "make_pizza"
