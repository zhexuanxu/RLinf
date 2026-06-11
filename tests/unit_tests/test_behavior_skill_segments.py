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

"""Unit tests for the deterministic BEHAVIOR frame-to-subtask resolver."""

import pytest

from rlinf.data.datasets.openpi_pytorch.behavior.skill_segments import (
    build_skill_segments,
    resolve_frame_subtask,
)


def _annotation(windows, valid, shuffle=False):
    """Build a minimal annotation dict from (start, end, skill_idx) windows."""
    entries = [
        {"skill_idx": idx, "frame_duration": [start, end]}
        for start, end, idx in windows
    ]
    if shuffle:
        entries = entries[::-1]
    return {
        "skill_annotation": entries,
        "meta_data": {"valid_duration": list(valid)},
    }


# The real annotations/task-0000/episode_00000030.json layout: four skills with
# gaps 211-666 and 1408-1416, valid_duration [0, 2038].
EPISODE_30_WINDOWS = [
    (0, 211, 0),
    (666, 999, 1),
    (999, 1408, 2),
    (1416, 2038, 3),
]


@pytest.fixture
def episode_30_segments():
    annotation = _annotation(EPISODE_30_WINDOWS, valid=(0, 2038))
    return build_skill_segments(annotation, num_subtasks=4, episode_id=30)


class TestEpisode30Resolution:
    @pytest.mark.parametrize(
        ("frame", "expected"),
        [
            (0, 0),
            (100, 0),
            (210, 0),
            (211, 1),  # First gap frame belongs to the NEXT ("pick up") skill.
            (300, 1),
            (665, 1),
            (666, 1),
            (700, 1),
            (998, 1),
            (999, 2),  # Contiguous boundary: half-open windows, no gap here.
            (1200, 2),
            (1407, 2),
            (1408, 3),  # Second gap frame belongs to the next skill.
            (1415, 3),
            (1416, 3),
            (1500, 3),
            (2037, 3),
        ],
    )
    def test_enable_gap_true(self, episode_30_segments, frame, expected):
        assert resolve_frame_subtask(episode_30_segments, frame, True) == expected

    @pytest.mark.parametrize(
        ("frame", "expected"),
        [
            (100, 0),
            (210, 0),
            (211, None),  # Gap frames are unused when enable_gap is off.
            (300, None),
            (665, None),
            (666, 1),
            (999, 2),
            (1408, None),
            (1415, None),
            (1416, 3),
        ],
    )
    def test_enable_gap_false(self, episode_30_segments, frame, expected):
        assert resolve_frame_subtask(episode_30_segments, frame, False) == expected

    @pytest.mark.parametrize("enable_gap", [True, False])
    def test_outside_valid_duration(self, episode_30_segments, enable_gap):
        assert resolve_frame_subtask(episode_30_segments, -1, enable_gap) is None
        assert resolve_frame_subtask(episode_30_segments, 2038, enable_gap) is None
        assert resolve_frame_subtask(episode_30_segments, 5000, enable_gap) is None


class TestValidDurationRelativeEdges:
    """Edges are relative to valid_duration, not raw frame zero / episode end.

    Mirrors real episodes with a nonzero valid_start, e.g. episode_00002150
    whose valid_duration is [262, 2170].
    """

    WINDOWS = [(300, 500, 0), (800, 1000, 1)]
    VALID = (262, 2170)

    @pytest.fixture
    def segments(self):
        return build_skill_segments(
            _annotation(self.WINDOWS, self.VALID), num_subtasks=2, episode_id=2150
        )

    @pytest.mark.parametrize("enable_gap", [True, False])
    def test_frames_below_valid_start_never_used(self, segments, enable_gap):
        assert resolve_frame_subtask(segments, 0, enable_gap) is None
        assert resolve_frame_subtask(segments, 261, enable_gap) is None

    def test_leading_gap_goes_to_first_skill_when_gap_enabled(self, segments):
        assert resolve_frame_subtask(segments, 262, True) == 0
        assert resolve_frame_subtask(segments, 299, True) == 0

    def test_leading_gap_unused_when_gap_disabled(self, segments):
        assert resolve_frame_subtask(segments, 262, False) is None
        assert resolve_frame_subtask(segments, 299, False) is None

    @pytest.mark.parametrize("enable_gap", [True, False])
    def test_trailing_gap_always_unused(self, segments, enable_gap):
        assert resolve_frame_subtask(segments, 1000, enable_gap) is None
        assert resolve_frame_subtask(segments, 1500, enable_gap) is None
        assert resolve_frame_subtask(segments, 2169, enable_gap) is None

    def test_inter_window_gap_goes_to_next_skill(self, segments):
        assert resolve_frame_subtask(segments, 500, True) == 1
        assert resolve_frame_subtask(segments, 799, True) == 1
        assert resolve_frame_subtask(segments, 500, False) is None


class TestBuildValidation:
    def test_windows_sorted_by_start_regardless_of_input_order(self):
        segments = build_skill_segments(
            _annotation(EPISODE_30_WINDOWS, valid=(0, 2038), shuffle=True),
            num_subtasks=4,
            episode_id=30,
        )
        assert segments.starts == (0, 666, 999, 1416)
        assert segments.skill_indices == (0, 1, 2, 3)

    def test_skill_idx_out_of_bounds_rejected(self):
        annotation = _annotation([(0, 10, 4)], valid=(0, 100))
        with pytest.raises(ValueError, match="episode 7.*skill_idx 4"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=7)

    def test_duplicate_skill_idx_rejected(self):
        annotation = _annotation([(0, 10, 0), (20, 30, 0)], valid=(0, 100))
        with pytest.raises(ValueError, match="episode 8.*duplicate skill_idx"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=8)

    def test_overlapping_windows_rejected(self):
        annotation = _annotation([(0, 50, 0), (40, 80, 1)], valid=(0, 100))
        with pytest.raises(ValueError, match="episode 9.*overlap"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=9)

    def test_empty_window_rejected(self):
        annotation = _annotation([(10, 10, 0)], valid=(0, 100))
        with pytest.raises(ValueError, match="empty or inverted"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=10)

    def test_missing_valid_duration_rejected(self):
        annotation = {"skill_annotation": [{"skill_idx": 0, "frame_duration": [0, 5]}]}
        with pytest.raises(ValueError, match="valid_duration"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=11)

    def test_empty_skill_annotation_rejected(self):
        annotation = {"skill_annotation": [], "meta_data": {"valid_duration": [0, 5]}}
        with pytest.raises(ValueError, match="skill_annotation"):
            build_skill_segments(annotation, num_subtasks=4, episode_id=12)
