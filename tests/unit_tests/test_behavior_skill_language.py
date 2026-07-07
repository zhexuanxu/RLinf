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

"""Unit tests for the BEHAVIOR subtask-language converter and task registry.

Most tests are pure and dataset-free. The corpus-coverage test is guarded by the
presence of the local BEHAVIOR annotations and skipped otherwise, so CI without
the dataset still passes while a machine with the data exercises every real entry.
"""

import glob
import json
import os

import pytest

from rlinf.data.datasets.openpi_pytorch.behavior.skill_language import (
    SKILL_TEMPLATES,
    behavior_task_names,
    clean_object_name,
    entry_to_subtask_text,
)

_BEHAVIOR_ANNOTATIONS = "/mnt/public/xzxuan/data/2025-challenge-demos/annotations"


def _entry(skill_description, *slots):
    """Build a one-entry annotation with the canonical ``[[slot, ...]]`` shape."""
    return {"skill_description": [skill_description], "object_id": [list(slots)]}


class TestCleanObjectName:
    """AC-3.1: deterministic object-name cleaning."""

    @pytest.mark.parametrize(
        "token,expected",
        [
            # <category>_<int>
            ("radio_89", "radio"),
            ("broom_172", "broom"),
            ("apple_pie_234", "apple pie"),
            # <category>_<model-hash>_<int>
            ("coffee_table_koagbh_0", "coffee table"),
            ("countertop_kelzer_0", "countertop"),
            ("fridge_petcxr_0", "fridge"),
            ("multi_station_furniture_sink_upwldu_0", "multi station furniture sink"),
            # real six-letter trailing words must be KEPT (not treated as a hash)
            ("beer_bottle_267", "beer bottle"),
            ("camera_tripod_86", "camera tripod"),
            ("digital_camera_87", "digital camera"),
            ("bell_pepper_213", "bell pepper"),
            ("boxing_gloves_188", "boxing gloves"),
            ("butter_cookie_81", "butter cookie"),
            ("allen_wrench_189", "allen wrench"),
            # cut-piece <category>_<int>_<int>
            ("half_head_cabbage_212_1", "half head cabbage"),
            ("half_beet_211_0", "half beet"),
            # dirty separators
            ("diced__chili", "diced chili"),
            ("half-log-176-0", "half log"),
            # bare tokens pass through unchanged
            ("mud", "mud"),
            ("left", "left"),
            ("right", "right"),
            ("robot", "robot"),
            ("grated_cheese", "grated cheese"),
            # a model hash that happens to look word-ish is still stripped
            ("wall_nail_wlnail_1", "wall nail"),
        ],
    )
    def test_clean_object_name(self, token, expected):
        assert clean_object_name(token) == expected

    def test_bare_hand_side_tokens_not_emptied(self):
        # The negative case from the AC: cleaning must never blank a bare word.
        for token in ("left", "right", "mud"):
            assert clean_object_name(token) != ""

    def test_multiword_stem_preserved(self):
        # "coffee table" must not collapse to "coffee".
        assert clean_object_name("coffee_table_koagbh_0") == "coffee table"

    @pytest.mark.parametrize("bad", ["", "   ", "89", "_0", 123, None])
    def test_invalid_tokens_raise(self, bad):
        with pytest.raises(ValueError):
            clean_object_name(bad)


class TestTurningOnRadioAnchor:
    """AC-3.2: reproduce the existing hand-written task-0000 labels exactly."""

    def test_anchor_labels_exact(self):
        entries = [
            _entry("move to", "radio_89"),
            _entry("pick up from", "radio_89", "coffee_table_koagbh_0"),
            _entry("press", "radio_89"),
            _entry("place on", "radio_89", "coffee_table_koagbh_0"),
        ]
        assert [entry_to_subtask_text(e) for e in entries] == [
            "move to radio",
            "pick up radio from coffee table",
            "press radio",
            "place radio on coffee table",
        ]


class TestEntryToSubtaskText:
    """AC-3, AC-3.3: template coverage, arities, list slots, fail-fast."""

    def test_one_object_templates(self):
        assert entry_to_subtask_text(_entry("press", "radio_89")) == "press radio"
        assert (
            entry_to_subtask_text(_entry("open door", "fridge_petcxr_0"))
            == "open door of fridge"
        )
        assert (
            entry_to_subtask_text(_entry("turn on switch", "lighter_73"))
            == "turn on lighter"
        )

    def test_two_object_templates(self):
        assert (
            entry_to_subtask_text(
                _entry("chop", "carving_knife_209", "head_cabbage_212")
            )
            == "chop head cabbage with carving knife"
        )
        assert (
            entry_to_subtask_text(_entry("spray", "atomizer_174", "pot_plant_173"))
            == "spray atomizer on pot plant"
        )

    def test_three_object_templates(self):
        assert (
            entry_to_subtask_text(
                _entry(
                    "place on next to",
                    "cauldron_92",
                    "floors_ulujpr_0",
                    "coffee_table_koagbh_0",
                )
            )
            == "place cauldron on floors next to coffee table"
        )

    def test_turn_to_drops_recipient(self):
        # slot 1 ("robot") carries no linguistic content and is dropped.
        assert (
            entry_to_subtask_text(_entry("turn to", "hinged_jar_236", "robot"))
            == "turn hinged jar"
        )

    def test_hand_over_drops_hand_sides(self):
        assert (
            entry_to_subtask_text(
                _entry("hand over", "hinged_jar_235", "right", "left")
            )
            == "hand over hinged jar"
        )

    def test_list_valued_slot_joined_and_deduped(self):
        # pour's first slot is a list of poured items; repeats collapse, order kept.
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
            entry_to_subtask_text(entry)
            == "pour candy cane and wreath into wicker basket"
        )

    def test_sweep_off_list_slot(self):
        entry = {
            "skill_description": ["sweep off"],
            "object_id": [[["half_log_176_0", "half_log_176_1"], "driveway_umalys_0"]],
        }
        assert entry_to_subtask_text(entry) == "sweep half log off driveway"

    def test_unknown_template_fails_fast_with_context(self):
        with pytest.raises(ValueError) as exc:
            entry_to_subtask_text(
                _entry("teleport", "radio_89"),
                task_name="task-0000",
                episode_id="episode_00000010.json",
                skill_idx=2,
            )
        message = str(exc.value)
        assert "teleport" in message
        assert "task-0000" in message
        assert "episode_00000010.json" in message
        assert "2" in message

    def test_too_few_slots_fails(self):
        with pytest.raises(ValueError):
            entry_to_subtask_text(_entry("pick up from", "radio_89"))

    def test_malformed_object_id_fails(self):
        # object_id must be the one-element wrapper list.
        with pytest.raises(ValueError):
            entry_to_subtask_text(
                {"skill_description": ["move to"], "object_id": ["radio_89"]}
            )

    def test_missing_skill_description_fails(self):
        with pytest.raises(ValueError):
            entry_to_subtask_text({"object_id": [["radio_89"]]})


class TestTaskNameRegistry:
    """AC-2.1: the 50 task names derive from TASK_NAMES_TO_INDICES, index-ordered."""

    def test_registry_matches_index_ordered_map(self):
        from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
            TASK_NAMES_TO_INDICES,
        )

        expected = [
            name
            for name, _ in sorted(TASK_NAMES_TO_INDICES.items(), key=lambda kv: kv[1])
        ]
        names = behavior_task_names()
        assert names == expected
        assert len(names) == 50
        assert len(set(names)) == 50
        assert names[0] == "turning_on_radio"
        assert names[-1] == "make_pizza"


class TestTemplateCoverageGuard:
    """AC-3.3: the template table covers every observed skill_description.

    Skipped when the local BEHAVIOR annotations are unavailable (e.g. CI).
    """

    @pytest.mark.skipif(
        not os.path.isdir(_BEHAVIOR_ANNOTATIONS),
        reason="BEHAVIOR annotations not available on this machine",
    )
    def test_all_observed_templates_convert(self):
        observed = set()
        failures = []
        converted = 0
        for task_dir in sorted(
            glob.glob(os.path.join(_BEHAVIOR_ANNOTATIONS, "task-*"))
        ):
            task_name = os.path.basename(task_dir)
            for path in sorted(glob.glob(os.path.join(task_dir, "*.json"))):
                episode_id = os.path.basename(path)
                with open(path) as handle:
                    annotation = json.load(handle)
                for entry in annotation.get("skill_annotation") or []:
                    observed.add(entry["skill_description"][0])
                    try:
                        text = entry_to_subtask_text(
                            entry,
                            task_name=task_name,
                            episode_id=episode_id,
                            skill_idx=entry.get("skill_idx"),
                        )
                    except ValueError as exc:  # pragma: no cover - failure path
                        failures.append(str(exc))
                        continue
                    assert text and text.strip()
                    converted += 1
        assert not failures, "\n".join(failures[:20])
        # Every observed template is in the table, and the table has no dead entries.
        assert observed == set(SKILL_TEMPLATES)
        assert converted > 0
