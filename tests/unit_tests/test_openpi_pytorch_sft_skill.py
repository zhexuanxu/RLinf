# Copyright (c) 2025, RLinf contributors.
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

"""Skill-mode parity for the BEHAVIOR SFT dataset vs the JAX openpi-comet logic.

Verifies the ported ``_build_skill_boundaries`` / ``_get_skill_label`` reproduce
openpi-comet's semantics: contiguous boundaries extend outward by
``allow_left`` / ``allow_right`` frames; ``enable_gap`` absorbs a true gap into
both adjacent skills (creating overlap); and a frame inside more than one
window is assigned a skill at random per sample — distinct from the old
gap-midpoint split.
"""

from __future__ import annotations

import random

from rlinf.data.datasets.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)

# Concrete example from the JAX reference:
# valid [0, 1000]; skills 0:[10,110], 1:[200,300], 2:[500,600] (gaps between all).
_ANNOTATION = {
    "skill_annotation": [
        {"skill_idx": 0, "frame_duration": [10, 110]},
        {"skill_idx": 1, "frame_duration": [200, 300]},
        {"skill_idx": 2, "frame_duration": [500, 600]},
    ],
    "meta_data": {"valid_duration": [0, 1000]},
}
_LABELS = {0: "move to radio", 1: "pick up radio", 2: "press radio"}


class _Meta:
    annotations = {0: _ANNOTATION}
    fps = 30  # `BehaviorSftDataset.fps` reads `self.meta.fps`


class _Scalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


def _make_dataset(enable_gap=True, allow_left=100, allow_right=100):
    ds = BehaviorSftDataset.__new__(BehaviorSftDataset)
    ds.enable_gap = enable_gap
    ds.allow_left = allow_left
    ds.allow_right = allow_right
    ds.episodes = [0]
    ds.meta = _Meta()
    ds.skill_labels = _LABELS
    ds._build_skill_boundaries()
    return ds


def _item(frame, fps=30):
    return {"episode_index": _Scalar(0), "timestamp": _Scalar(frame / fps)}


def test_skill_boundaries_match_reference_example():
    ds = _make_dataset(enable_gap=True, allow_left=100, allow_right=100)
    # Skill 0: first-skill left extends 10-100 -> clamp 0; gap-right absorbs to 200.
    # Skill 1: gap on both sides absorbs to [110, 500].
    # Skill 2: gap-left absorbs to 300; last-skill right extends 600+100 = 700.
    assert ds.skill_start_frames[0] == [0, 110, 300]
    assert ds.skill_end_frames[0] == [200, 500, 700]


def test_skill_label_single_window_frames():
    ds = _make_dataset()
    assert ds._get_skill_label(_item(50)) == _LABELS[0]  # only skill 0
    assert ds._get_skill_label(_item(250)) == _LABELS[1]  # only skill 1
    assert ds._get_skill_label(_item(650)) == _LABELS[2]  # only skill 2


def test_overlap_frame_is_random_not_midpoint_split():
    ds = _make_dataset()
    # Frame 150 lies in the absorbed gap [110, 200): both skill 0 and skill 1.
    # Old midpoint split ((110+200)//2 = 155) would always assign skill 0 here;
    # the JAX overlap semantics assign skill 0 OR 1 at random per sample.
    random.seed(0)
    seen = {ds._get_skill_label(_item(150)) for _ in range(200)}
    assert seen == {_LABELS[0], _LABELS[1]}
    # Frame 400 lies in [300, 500): both skill 1 and skill 2.
    seen2 = {ds._get_skill_label(_item(400)) for _ in range(200)}
    assert seen2 == {_LABELS[1], _LABELS[2]}


def test_gap_frame_detection_uses_effective_windows():
    ds = _make_dataset(enable_gap=True, allow_left=100, allow_right=100)
    # With gaps absorbed, the windows tile [0, 700) with overlaps -> no gaps in-range.
    assert not ds._is_gap_frame(0, 150)
    assert not ds._is_gap_frame(0, 400)
    # Past the last effective window end (700) is outside every window.
    assert ds._is_gap_frame(0, 800)


def test_enable_gap_false_does_not_absorb_gaps():
    ds = _make_dataset(enable_gap=False, allow_left=0, allow_right=0)
    # No extension, no absorption -> windows are the raw annotation ranges.
    assert ds.skill_start_frames[0] == [10, 200, 500]
    assert ds.skill_end_frames[0] == [110, 300, 600]
    # A true gap frame (between skill 0 and 1) is now outside every window.
    assert ds._is_gap_frame(0, 150)
