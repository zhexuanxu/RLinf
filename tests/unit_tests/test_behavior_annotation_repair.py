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

"""Unit tests for the BEHAVIOR skill-annotation repair rules.

Exercises the pure ``repair_skill_annotation`` function from the maintenance
script one rule at a time (dedup, ambiguous-label delete-all, union merge,
staircase trim, nonscalar split, the two bespoke fixes) plus the ``skill_idx ==
position`` invariant and the byte-faithful JSON encoder. Every repaired
annotation is asserted to pass ``build_skill_segments``.
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../toolkits/eval_scripts_openpi")
)
from fix_behavior_annotations import (  # noqa: E402
    build_skill_segments,
    encode_annotation,
    repair_skill_annotation,
)


def _entry(skill_idx, frame_duration, description, objects):
    """Build one skill_annotation entry with the fields the pipeline reads."""
    return {
        "skill_idx": skill_idx,
        "skill_description": [description],
        "object_id": [list(objects)],
        "frame_duration": frame_duration,
    }


def _annotation(entries, valid):
    """Wrap skill_annotation entries in a minimal episode annotation dict."""
    return {
        "skill_annotation": entries,
        "meta_data": {"valid_duration": list(valid)},
    }


def _validate(repaired, episode_id):
    """Assert the repaired annotation passes build_skill_segments."""
    build_skill_segments(
        repaired, len(repaired["skill_annotation"]), episode_id=episode_id
    )


def _spans(repaired):
    """Return the (start, end) windows of a repaired annotation, in order."""
    return [tuple(e["frame_duration"]) for e in repaired["skill_annotation"]]


class TestRenumberInvariant:
    """skill_idx must equal list position after any repair (dataset invariant)."""

    def test_skill_idx_matches_position(self):
        entries = [
            _entry(0, [0, 100], "move to", ["radio_1"]),
            _entry(1, [100, 200], "press", ["radio_1"]),
            _entry(2, [200, 300], "move to", ["radio_1"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (0, 300)), 1
        )
        assert [e["skill_idx"] for e in repaired["skill_annotation"]] == [0, 1, 2]
        assert not deleted
        _validate(repaired, 1)


class TestRule1DedupIdentical:
    """Identical frame window rendering to the SAME subtask -> keep one."""

    def test_byte_identical_duplicate_dropped(self):
        entries = [
            _entry(0, [0, 100], "move to", ["radio_1"]),
            _entry(1, [100, 200], "place in", ["radio_1", "box_2"]),
            _entry(2, [100, 200], "place in", ["radio_1", "box_2"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (0, 200)), 20020
        )
        assert ops["dedup_identical"] == 1
        assert not deleted
        assert _spans(repaired) == [(0, 100), (100, 200)]
        _validate(repaired, 20020)


class TestRule2DeleteAmbiguous:
    """Identical frame window, DIFFERENT subtasks -> delete all (untrained)."""

    def test_all_windows_on_ambiguous_span_removed(self):
        entries = [
            _entry(0, [0, 100], "move to", ["beet_1"]),
            _entry(1, [100, 150], "chop", ["parer_1", "beet_1"]),
            _entry(2, [100, 150], "chop", ["parer_1", "zucchini_2"]),
            _entry(3, [150, 200], "move to", ["parer_1"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (0, 200)), 430800
        )
        assert ops["delete_ambiguous"] == 1
        assert deleted == [[100, 150]]
        # Both contested windows are gone; the frame span is left unowned.
        assert _spans(repaired) == [(0, 100), (150, 200)]
        _validate(repaired, 430800)


class TestRule2aMergeUnion:
    """Partial overlap rendering the SAME subtask -> merge to the union."""

    def test_same_text_partial_overlap_merged(self):
        entries = [
            _entry(0, [7751, 8267], "hand over", ["jar_1", "right", "left"]),
            _entry(1, [7997, 8267], "hand over", ["jar_1", "left", "right"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (7000, 9000)), 40240
        )
        assert ops["merge_union"] == 1
        assert not deleted
        assert _spans(repaired) == [(7751, 8267)]
        _validate(repaired, 40240)


class TestRule2bTrimStaircase:
    """Small (<=20 frame) staircase overlap of different subtasks -> trim both."""

    def test_small_staircase_trimmed(self):
        entries = [
            _entry(0, [4821, 5253], "place in", ["candle_1", "table_2"]),
            _entry(1, [5252, 5532], "move to", ["candle_1"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (4000, 6000)), 92250
        )
        assert ops["trim_staircase"] == 1
        assert not deleted
        # The shared frame [5252, 5253) is trimmed out of both windows.
        assert _spans(repaired) == [(4821, 5252), (5253, 5532)]
        _validate(repaired, 92250)

    def test_large_overlap_not_trimmed(self):
        # A >20-frame different-text overlap is NOT a staircase-trim case; it must
        # be left unresolved (and would be reported), not silently mangled.
        entries = [
            _entry(0, [6396, 7427], "ignite", ["lighter_1", "firewood_2"]),
            _entry(1, [6540, 7050], "place in", ["firewood_2", "fireplace_3"]),
        ]
        _, ops, _ = repair_skill_annotation(_annotation(entries, (0, 8000)), 301710)
        assert ops["trim_staircase"] == 0
        assert ops["unresolved_overlap"] == 1


class TestRule4SplitNonscalar:
    """A frame_duration list of intervals -> one window per interval."""

    def test_interleaved_skill_split_into_two(self):
        entries = [
            _entry(0, [[2795, 3007], [3209, 3638]], "move to", ["bratwurst_1"]),
            _entry(1, [3008, 3209], "open lid", ["jar_2"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (2000, 4000)), 40170
        )
        assert ops["split_nonscalar"] == 1
        assert not deleted
        # move to [2795,3007], open lid [3008,3209], move to [3209,3638].
        assert _spans(repaired) == [(2795, 3007), (3008, 3209), (3209, 3638)]
        assert [e["skill_idx"] for e in repaired["skill_annotation"]] == [0, 1, 2]
        _validate(repaired, 40170)


class TestBespokeFixes:
    """The two hand-specified episode repairs (rules 5 and 6)."""

    def test_490320_oven_triple_reordered(self):
        # Raw: inverted push-tray and an overlapping close-door around the oven.
        entries = [
            _entry(57, [13311, 13973], "place in", ["sheet_1", "oven_2"]),
            _entry(58, [13931, 14180], "close door", ["oven_2"]),
            _entry(59, [13973, 13931], "push tray", ["oven_2"]),
            _entry(60, [14180, 14691], "turn on switch", ["oven_2"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (0, 15000)), 490320
        )
        assert ops["bespoke_490320"] == 1
        assert not deleted
        assert _spans(repaired) == [
            (13311, 13973),
            (13973, 14130),
            (14130, 14550),
            (14550, 14691),
        ]
        _validate(repaired, 490320)

    def test_42230_inverted_split_interval_repaired(self):
        # The second interval of the split move-to is inverted ([6738, 6459]).
        entries = [
            _entry(14, [5531, 5949], "hand over", ["jar_1"]),
            _entry(15, [[5949, 6456], [6738, 6459]], "move to", ["bratwurst_2"]),
            _entry(16, [6459, 6738], "open lid", ["jar_1"]),
            _entry(17, [6939, 7212], "pick up from", ["bratwurst_2", "board_3"]),
        ]
        repaired, ops, deleted = repair_skill_annotation(
            _annotation(entries, (5000, 8000)), 42230
        )
        assert ops["bespoke_42230"] == 1
        assert (6738, 6939) in _spans(repaired)
        _validate(repaired, 42230)


class TestEncoderRoundTrip:
    """The encoder reproduces the dataset's inline-scalar-array JSON format."""

    def test_scalar_arrays_inline_object_arrays_expanded(self):
        annotation = _annotation(
            [_entry(0, [0, 211], "move to", ["radio_89"])], (0, 2038)
        )
        text = encode_annotation(annotation)
        # Scalar arrays stay on one line; object arrays expand one-per-line.
        assert '"frame_duration": [0, 211]' in text
        assert '"skill_description": ["move to"]' in text
        assert '"skill_annotation": [\n' in text
        assert not text.endswith("\n")


class TestIdempotency:
    """Repairing an already-valid annotation is a no-op on its content."""

    def test_clean_annotation_unchanged(self):
        entries = [
            _entry(0, [0, 100], "move to", ["radio_1"]),
            _entry(1, [100, 200], "press", ["radio_1"]),
        ]
        annotation = _annotation(entries, (0, 200))
        repaired, ops, deleted = repair_skill_annotation(annotation, 30)
        assert not ops
        assert not deleted
        assert encode_annotation(repaired) == encode_annotation(annotation)
        _validate(repaired, 30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
