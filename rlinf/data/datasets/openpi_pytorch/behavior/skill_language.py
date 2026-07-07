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

"""Natural-language subtask text for BEHAVIOR-1K skill annotations.

A BEHAVIOR episode annotation carries a ``skill_annotation`` list; each entry
pairs a ``skill_description`` (a one-element list such as ``["move to"]``) with
an ``object_id`` (a list of *slots*, each slot being either an object token or a
list of object tokens). This module turns one such entry into a single
natural-language subtask string -- the VLM supervision target used by the pi05
``vlm_vla`` SFT path.

Why generate the text here instead of listing it per task
--------------------------------------------------------
The subtask windows of a BEHAVIOR task are NOT fixed across its trajectories:
for most tasks the ``skill_idx`` of a window is only a per-episode positional
index, so the same index means different skills in different episodes. A single
per-task label list is therefore correct for only a handful of tasks. Resolving
the text from each episode's own annotation is correct for every task, and the
annotations are already loaded by the dataset, so no extra IO is needed.

Design contract
---------------
* :func:`clean_object_name` maps a raw object token (e.g. ``coffee_table_koagbh_0``)
  to a readable name (``"coffee table"``). It strips the trailing instance id and
  the OmniGibson model hash, converts separators to spaces, and lowercases. It
  never returns an empty string for a non-empty input.
* :func:`entry_to_subtask_text` looks the ``skill_description`` up in an explicit
  table covering every template observed in the dataset and formats the cleaned
  object names into the template's phrasing.
* Anything unexpected -- an unknown skill template, a malformed entry, a slot
  that cleans to nothing -- raises :class:`ValueError` with task/episode/skill
  context. There is no silent fallback: mislabeled supervision is worse than a
  loud failure.
* :func:`behavior_task_names` exposes the canonical 50 task names (index-ordered)
  derived from the single authoritative ``TASK_NAMES_TO_INDICES`` map, so configs
  never carry a second, drift-prone copy of the list.
"""

from __future__ import annotations

import re

__all__ = [
    "SKILL_TEMPLATES",
    "REAL_WORD_MODEL_HASH_ENDINGS",
    "clean_object_name",
    "entry_to_subtask_text",
    "behavior_task_names",
]


# Slot fillers are referenced positionally in the format strings below:
# ``{0}`` is the first object slot, ``{1}`` the second, ``{2}`` the third. Each
# filler is the cleaned name of that slot (a list-valued slot is joined by
# :func:`_join_names`). Slots that carry no linguistic content (the hand side of
# ``hand over``; the ``robot`` recipient of ``turn to``) are simply not
# referenced, so they are cleaned-and-dropped rather than rendered.
#
# The table is the sole source of truth for which templates are known: a
# ``skill_description`` absent here is rejected by :func:`entry_to_subtask_text`.
SKILL_TEMPLATES: dict[str, str] = {
    # --- single-object motions/manipulations -------------------------------
    "move to": "move to {0}",
    "press": "press {0}",
    "hold": "hold {0}",
    "release": "release {0}",
    "tip over": "tip over {0}",
    "pull tray": "pull tray of {0}",
    "push tray": "push tray of {0}",
    "open door": "open door of {0}",
    "close door": "close door of {0}",
    "open drawer": "open drawer of {0}",
    "close drawer": "close drawer of {0}",
    "open lid": "open lid of {0}",
    "close lid": "close lid of {0}",
    "turn on switch": "turn on {0}",
    "turn off switch": "turn off {0}",
    # --- two-object manipulations ------------------------------------------
    "pick up from": "pick up {0} from {1}",
    "place on": "place {0} on {1}",
    "place in": "place {0} in {1}",
    "place under": "place {0} under {1}",
    "push to": "push {0} to {1}",
    "spray": "spray {0} on {1}",
    "insert": "insert {0} into {1}",
    "attach": "attach {0} to {1}",
    "hang": "hang {0} on {1}",
    "chop": "chop {1} with {0}",
    "wipe hard": "wipe {1} with {0}",
    "ignite": "ignite {1} with {0}",
    "sweep surface": "sweep {1} with {0}",
    "sweep off": "sweep {0} off {1}",
    "turn to": "turn {0}",  # slot 1 is the recipient ("robot"); dropped.
    # --- three-object manipulations ----------------------------------------
    "place on next to": "place {0} on {1} next to {2}",
    "place in next to": "place {0} in {1} next to {2}",
    "pour": "pour {0} into {1}",  # slot 2 is the support location; dropped.
    "hand over": "hand over {0}",  # slots 1,2 are hand sides; dropped.
}

# Six-letter tokens that are genuine trailing words of an object category, not
# OmniGibson model hashes. A model instance token looks like
# ``<category>_<6-random-lowercase>_<int>`` (e.g. ``coffee_table_koagbh_0``), but
# some categories legitimately END in a six-letter word (``beer_bottle_267``,
# ``camera_tripod_86``). Stripping those as if they were hashes would corrupt the
# name, so the six-letter-hash rule skips any chunk listed here. Derived by
# scanning every object token in the dataset annotations.
REAL_WORD_MODEL_HASH_ENDINGS: frozenset[str] = frozenset(
    {
        "basket",
        "bottle",
        "camera",
        "candle",
        "cheese",
        "cookie",
        "coffee",
        "filter",
        "gloves",
        "kettle",
        "pepper",
        "puzzle",
        "racket",
        "tomato",
        "tripod",
        "wrench",
    }
)

# A trailing instance index: ``_89`` or a cut-piece ``_212_1``.
_TRAILING_INSTANCE_RE = re.compile(r"_\d+(?:_\d+)?$")
# A trailing OmniGibson model hash: exactly six lowercase letters between the
# category and the instance index has already been removed, e.g. the ``koagbh``
# left at the end of ``coffee_table_koagbh`` after ``_0`` was stripped.
_TRAILING_HASH_RE = re.compile(r"_([a-z]{6})$")


def clean_object_name(token: str) -> str:
    """Turn a raw BEHAVIOR object token into a readable object name.

    Steps (in order):

    1. Normalize whitespace and case: strip surrounding whitespace, collapse any
       internal runs of whitespace, and lowercase.
    2. Convert hyphens and repeated underscores to single underscores so tokens
       like ``half-log-176-0`` and ``diced__chili`` are handled uniformly.
    3. Strip a trailing instance index (``_89``) or cut-piece index (``_212_1``).
    4. Strip a trailing six-letter model hash (``_koagbh``) unless that chunk is a
       real trailing word (see :data:`REAL_WORD_MODEL_HASH_ENDINGS`).
    5. Replace the remaining underscores with spaces and collapse whitespace.

    Args:
        token: A raw object token from an annotation ``object_id`` slot.

    Returns:
        The cleaned, space-separated, lowercase object name.

    Raises:
        ValueError: If ``token`` is not a string or is empty/blank, or if
            cleaning would yield an empty string.
    """
    if not isinstance(token, str):
        raise ValueError(f"object token must be a string, got {token!r}")
    normalized = re.sub(r"\s+", " ", token).strip().lower()
    if not normalized:
        raise ValueError(f"object token is empty or blank: {token!r}")

    # Unify separators: hyphens -> underscore, collapse repeated underscores.
    unified = normalized.replace(" ", "_").replace("-", "_")
    unified = re.sub(r"_+", "_", unified).strip("_")
    if not unified:
        raise ValueError(f"object token has no content after cleanup: {token!r}")

    # Strip the trailing instance/piece index, then a model hash if present and
    # not a real trailing word.
    stem = _TRAILING_INSTANCE_RE.sub("", unified)
    hash_match = _TRAILING_HASH_RE.search(stem)
    if hash_match and hash_match.group(1) not in REAL_WORD_MODEL_HASH_ENDINGS:
        stem = stem[: hash_match.start()]
    stem = stem.strip("_")

    # A bare token with no instance suffix (e.g. "mud", "left", "robot") survives
    # unchanged; a token that was only an index (e.g. "89", "_0") leaves no
    # category part and is rejected rather than emitting a bare number.
    name = stem.replace("_", " ").strip()
    if not name or not any(ch.isalpha() for ch in name):
        raise ValueError(
            f"object token {token!r} cleaned to a name with no category part: {name!r}"
        )
    return name


def _clean_slot(slot: object, *, context: str) -> str:
    """Clean one ``object_id`` slot (a token or a list of tokens) to a name.

    A list-valued slot (e.g. the poured items of ``pour``) is cleaned
    element-wise and joined by :func:`_join_names`.
    """
    if isinstance(slot, str):
        return clean_object_name(slot)
    if isinstance(slot, (list, tuple)):
        if not slot:
            raise ValueError(f"{context}: empty object slot")
        names = [clean_object_name(tok) for tok in slot]
        return _join_names(names)
    raise ValueError(f"{context}: object slot must be a string or list, got {slot!r}")


def _join_names(names: list[str]) -> str:
    """Join multiple cleaned object names, de-duplicating while keeping order.

    Repeated categories (``candy cane`` appearing several times in a poured-items
    slot) collapse to one mention. Two names join with ``" and "``; three or more
    use a comma series with a trailing ``" and "``.
    """
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _skill_description_text(skill_description: object, *, context: str) -> str:
    """Extract the single template string from a ``skill_description`` field.

    The annotation stores it as a one-element list (``["move to"]``); a bare
    string is also accepted.
    """
    if isinstance(skill_description, str):
        text = skill_description
    elif isinstance(skill_description, (list, tuple)):
        if len(skill_description) != 1 or not isinstance(skill_description[0], str):
            raise ValueError(
                f"{context}: skill_description must be a single string, got "
                f"{skill_description!r}"
            )
        text = skill_description[0]
    else:
        raise ValueError(
            f"{context}: skill_description must be a string or one-element list, "
            f"got {skill_description!r}"
        )
    text = text.strip()
    if not text:
        raise ValueError(f"{context}: skill_description is empty")
    return text


def entry_to_subtask_text(
    entry: dict,
    *,
    task_name: object = None,
    episode_id: object = None,
    skill_idx: object = None,
) -> str:
    """Convert one ``skill_annotation`` entry to its natural-language subtask.

    Args:
        entry: A single ``skill_annotation`` element, carrying at least
            ``skill_description`` and ``object_id``.
        task_name: Optional task name, used only to enrich error messages.
        episode_id: Optional episode id, used only to enrich error messages.
        skill_idx: Optional skill index, used only to enrich error messages.

    Returns:
        The subtask text, e.g. ``"pick up radio from coffee table"``.

    Raises:
        ValueError: If the entry is malformed, the ``skill_description`` is not a
            known template, the number of object slots does not match the
            template, or any slot cleans to an empty name. The message names the
            task/episode/skill and the offending template.
    """
    context = (
        f"task {task_name!r} episode {episode_id!r} skill_idx {skill_idx!r}"
        if task_name is not None or episode_id is not None or skill_idx is not None
        else "skill annotation"
    )
    if not isinstance(entry, dict):
        raise ValueError(f"{context}: skill annotation entry must be a dict")

    template_key = _skill_description_text(
        entry.get("skill_description"), context=context
    )
    template = SKILL_TEMPLATES.get(template_key)
    if template is None:
        raise ValueError(
            f"{context}: unknown skill template {template_key!r}; add it to "
            "SKILL_TEMPLATES with its natural-language phrasing"
        )

    slots = _object_slots(entry.get("object_id"), context=context)

    # The number of referenced slots is the highest ``{i}`` index in the template
    # plus one; extra trailing slots (hand sides, the turn-to recipient) are
    # intentionally not referenced and thus ignored.
    referenced = _referenced_slot_count(template)
    if len(slots) < referenced:
        raise ValueError(
            f"{context}: template {template_key!r} needs {referenced} object "
            f"slot(s) but the annotation has {len(slots)}"
        )

    fillers = [
        _clean_slot(slots[i], context=f"{context} template {template_key!r}")
        for i in range(referenced)
    ]
    return template.format(*fillers)


def _object_slots(object_id: object, *, context: str) -> list:
    """Return the list of object slots from an annotation ``object_id`` field.

    The annotation always wraps the slots in a one-element outer list:
    ``object_id == [[slot0, slot1, ...]]``. This unwraps that single layer and
    returns ``[slot0, slot1, ...]``, where each slot is itself either an object
    token or a list of object tokens. A missing, non-list, or multi-element outer
    list is rejected.
    """
    if not isinstance(object_id, (list, tuple)):
        raise ValueError(f"{context}: object_id must be a list, got {object_id!r}")
    if len(object_id) != 1 or not isinstance(object_id[0], (list, tuple)):
        raise ValueError(
            f"{context}: object_id must be a one-element list wrapping the slot "
            f"list, got {object_id!r}"
        )
    return list(object_id[0])


def _referenced_slot_count(template: str) -> int:
    """Return how many positional slots (``{0}``, ``{1}``, ...) a template uses."""
    indices = [int(m) for m in re.findall(r"\{(\d+)\}", template)]
    return max(indices) + 1 if indices else 0


def behavior_task_names() -> list[str]:
    """Return the 50 BEHAVIOR task names, ordered by their canonical task index.

    Derived from the single authoritative ``TASK_NAMES_TO_INDICES`` map so the
    list never drifts from the dataset's own task ordering. Imported lazily to
    avoid importing the (heavier) dataset module unless the names are needed.
    """
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        TASK_NAMES_TO_INDICES,
    )

    return [
        name for name, _ in sorted(TASK_NAMES_TO_INDICES.items(), key=lambda kv: kv[1])
    ]
