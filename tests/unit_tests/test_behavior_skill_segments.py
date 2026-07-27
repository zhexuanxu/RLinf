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

"""Tests for deterministic BEHAVIOR frame-to-subtask resolution."""

import pytest

from rlinf.data.datasets.openpi_pytorch.behavior.skill_segments import (
    build_skill_segments,
    resolve_frame_subtask,
)


def _annotation(windows, valid=(0, 100)):
    return {
        "skill_annotation": [
            {"skill_idx": index, "frame_duration": [start, end]}
            for start, end, index in windows
        ],
        "meta_data": {"valid_duration": list(valid)},
    }


def test_windows_and_gaps_are_half_open_and_deterministic():
    segments = build_skill_segments(
        _annotation([(0, 10, 0), (20, 30, 1), (30, 40, 2)], valid=(0, 50)),
        num_subtasks=3,
        episode_id=7,
    )

    assert resolve_frame_subtask(segments, 9, True) == 0
    assert resolve_frame_subtask(segments, 10, True) == 1
    assert resolve_frame_subtask(segments, 19, True) == 1
    assert resolve_frame_subtask(segments, 10, False) is None
    assert resolve_frame_subtask(segments, 20, False) == 1
    assert resolve_frame_subtask(segments, 30, False) == 2
    assert resolve_frame_subtask(segments, 40, True) is None


def test_valid_duration_clips_leading_and_trailing_frames():
    segments = build_skill_segments(
        _annotation([(10, 20, 0)], valid=(5, 30)),
        num_subtasks=1,
        episode_id=8,
    )

    assert resolve_frame_subtask(segments, 4, True) is None
    assert resolve_frame_subtask(segments, 5, True) == 0
    assert resolve_frame_subtask(segments, 9, False) is None
    assert resolve_frame_subtask(segments, 20, True) is None
    assert resolve_frame_subtask(segments, 30, True) is None


@pytest.mark.parametrize(
    ("annotation", "pattern"),
    [
        (_annotation([(10, 10, 0)]), "empty or inverted"),
        (_annotation([(0, 20, 0), (10, 30, 1)]), "overlap"),
        (_annotation([(0, 10, 0), (20, 30, 0)]), "duplicate skill_idx"),
        (_annotation([(0, 10, 3)]), "outside"),
        (_annotation([(0, 20, 0)], valid=(5, 30)), "valid_duration"),
        ({"skill_annotation": []}, "skill_annotation"),
    ],
)
def test_malformed_annotations_fail_loudly(annotation, pattern):
    with pytest.raises(ValueError, match=pattern):
        build_skill_segments(annotation, num_subtasks=2, episode_id=9)
