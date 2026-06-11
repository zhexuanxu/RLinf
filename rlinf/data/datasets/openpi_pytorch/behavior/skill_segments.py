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

"""Deterministic frame-to-subtask resolution for BEHAVIOR skill annotations.

A BEHAVIOR episode annotation carries a ``skill_annotation`` list whose entries
hold a ``skill_idx`` and a half-open frame window ``frame_duration = [start,
end)``, plus a ``meta_data.valid_duration = [valid_start, valid_end)`` range of
usable frames. This module maps a frame index to the subtask (skill) index that
owns it — or to "unused" — with no randomness and no window overlap:

* frames outside ``valid_duration`` are never used;
* a frame inside window ``i`` belongs to that window's ``skill_idx``;
* the gap between two windows belongs to the NEXT window when ``enable_gap``
  is set, otherwise the gap frames are unused;
* the leading gap ``[valid_start, first_start)`` belongs to the first window
  when ``enable_gap`` is set, otherwise it is unused;
* the trailing gap ``[last_end, valid_end)`` is always unused (no next window).

The functions are pure (no torch, no dataset state) so the mapping is unit
testable in isolation; :class:`BehaviorSftDataset` builds one
:class:`SkillSegments` per episode and queries it per streamed frame.
"""

from __future__ import annotations

import bisect
import dataclasses

__all__ = ["SkillSegments", "build_skill_segments", "resolve_frame_subtask"]


@dataclasses.dataclass(frozen=True)
class SkillSegments:
    """Validated, start-sorted skill windows of one episode.

    Attributes:
        starts: Window start frames, strictly increasing.
        ends: Half-open window end frames, parallel to ``starts``.
        skill_indices: The annotation ``skill_idx`` of each window, parallel to
            ``starts`` (used to index the configured subtask label list).
        valid_start: First usable frame of the episode (inclusive).
        valid_end: One past the last usable frame of the episode.
    """

    starts: tuple[int, ...]
    ends: tuple[int, ...]
    skill_indices: tuple[int, ...]
    valid_start: int
    valid_end: int


def build_skill_segments(
    annotation: dict, num_subtasks: int, episode_id: object
) -> SkillSegments:
    """Validate an episode annotation and build its :class:`SkillSegments`.

    Args:
        annotation: The parsed episode annotation JSON (must contain
            ``skill_annotation`` and ``meta_data.valid_duration``).
        num_subtasks: Number of configured subtask labels for the episode's
            task; every ``skill_idx`` must index into this range.
        episode_id: Identifier used in error messages only.

    Returns:
        The validated, start-sorted segments.

    Raises:
        ValueError: On any malformed annotation (missing fields, empty or
            inverted windows, overlapping windows, duplicate or out-of-range
            ``skill_idx``). Invalid annotations are never silently skipped or
            clipped.
    """

    prefix = f"episode {episode_id}: invalid skill annotation:"

    skills = annotation.get("skill_annotation")
    if not skills:
        raise ValueError(f"{prefix} missing or empty 'skill_annotation'")
    valid = annotation.get("meta_data", {}).get("valid_duration")
    if not isinstance(valid, (list, tuple)) or len(valid) != 2:
        raise ValueError(f"{prefix} missing 'meta_data.valid_duration' pair")
    valid_start, valid_end = int(valid[0]), int(valid[1])
    if valid_start >= valid_end:
        raise ValueError(f"{prefix} empty valid_duration [{valid_start}, {valid_end})")

    windows: list[tuple[int, int, int]] = []
    for entry in skills:
        duration = entry.get("frame_duration")
        if not isinstance(duration, (list, tuple)) or len(duration) != 2:
            raise ValueError(
                f"{prefix} entry {entry.get('skill_idx')!r} has no [start, end) pair"
            )
        start, end = int(duration[0]), int(duration[1])
        if start >= end:
            raise ValueError(f"{prefix} window [{start}, {end}) is empty or inverted")
        skill_idx = entry.get("skill_idx")
        if not isinstance(skill_idx, int):
            raise ValueError(f"{prefix} non-integer skill_idx {skill_idx!r}")
        if not 0 <= skill_idx < num_subtasks:
            raise ValueError(
                f"{prefix} skill_idx {skill_idx} outside the configured "
                f"{num_subtasks} subtask labels"
            )
        windows.append((start, end, skill_idx))

    windows.sort(key=lambda w: w[0])
    seen_indices = [w[2] for w in windows]
    if len(set(seen_indices)) != len(seen_indices):
        raise ValueError(f"{prefix} duplicate skill_idx values {seen_indices}")
    for (_, prev_end, prev_idx), (cur_start, _, cur_idx) in zip(windows, windows[1:]):
        if cur_start < prev_end:
            raise ValueError(
                f"{prefix} windows of skill_idx {prev_idx} and {cur_idx} overlap "
                f"(start {cur_start} < previous end {prev_end})"
            )

    return SkillSegments(
        starts=tuple(w[0] for w in windows),
        ends=tuple(w[1] for w in windows),
        skill_indices=tuple(w[2] for w in windows),
        valid_start=valid_start,
        valid_end=valid_end,
    )


def resolve_frame_subtask(
    segments: SkillSegments, frame_index: int, enable_gap: bool
) -> int | None:
    """Resolve a frame to the ``skill_idx`` that owns it, or None if unused.

    Args:
        segments: The episode's validated skill windows.
        frame_index: Absolute frame index within the episode.
        enable_gap: Whether gap frames are assigned to the next window
            (True) or unused (False).

    Returns:
        The owning ``skill_idx``, or ``None`` when the frame must not be used
        for training (outside ``valid_duration``, in a gap with
        ``enable_gap=False``, or in the trailing gap).
    """
    if not segments.valid_start <= frame_index < segments.valid_end:
        return None
    # Windows are start-sorted: position points at the first window starting
    # AFTER the frame, so the frame lies either inside the previous window or
    # in the gap before that next window.
    position = bisect.bisect_right(segments.starts, frame_index)
    if position > 0 and frame_index < segments.ends[position - 1]:
        return segments.skill_indices[position - 1]
    if position == len(segments.starts):
        return None  # Trailing gap: there is no next window to absorb it.
    return segments.skill_indices[position] if enable_gap else None
