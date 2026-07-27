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

"""Natural-language subtask text for BEHAVIOR skill annotations.

Skill sequences can vary between episodes of the same task. This module derives
the VLM supervision target from each episode's own ``skill_description`` and
``object_id`` instead of relying on a fixed per-task label list.
"""

from __future__ import annotations

import re

__all__ = [
    "REAL_WORD_MODEL_HASH_ENDINGS",
    "SKILL_TEMPLATES",
    "behavior_task_names",
    "clean_object_name",
    "entry_to_subtask_text",
]


SKILL_TEMPLATES: dict[str, str] = {
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
    "turn to": "turn {0}",
    "place on next to": "place {0} on {1} next to {2}",
    "place in next to": "place {0} in {1} next to {2}",
    "pour": "pour {0} into {1}",
    "hand over": "hand over {0}",
}

# Six-letter category words which must not be mistaken for OmniGibson hashes.
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

_TRAILING_INSTANCE_RE = re.compile(r"_\d+(?:_\d+)?$")
_TRAILING_HASH_RE = re.compile(r"_([a-z]{6})$")


def clean_object_name(token: str) -> str:
    """Convert a BEHAVIOR object token to a readable lowercase name."""
    if not isinstance(token, str):
        raise ValueError(f"object token must be a string, got {token!r}")
    normalized = re.sub(r"\s+", " ", token).strip().lower()
    if not normalized:
        raise ValueError(f"object token is empty or blank: {token!r}")

    unified = normalized.replace(" ", "_").replace("-", "_")
    unified = re.sub(r"_+", "_", unified).strip("_")
    if not unified:
        raise ValueError(f"object token has no content after cleanup: {token!r}")

    stem = _TRAILING_INSTANCE_RE.sub("", unified)
    hash_match = _TRAILING_HASH_RE.search(stem)
    if hash_match and hash_match.group(1) not in REAL_WORD_MODEL_HASH_ENDINGS:
        stem = stem[: hash_match.start()]
    name = stem.strip("_").replace("_", " ").strip()
    if not name or not any(character.isalpha() for character in name):
        raise ValueError(
            f"object token {token!r} cleaned to a name with no category part: {name!r}"
        )
    return name


def _join_names(names: list[str]) -> str:
    """Join names after stable de-duplication."""
    unique: list[str] = []
    for name in names:
        if name not in unique:
            unique.append(name)
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return ", ".join(unique[:-1]) + f", and {unique[-1]}"


def _clean_slot(slot: object, *, context: str) -> str:
    """Clean a scalar or list-valued object slot."""
    if isinstance(slot, str):
        return clean_object_name(slot)
    if isinstance(slot, (list, tuple)):
        if not slot:
            raise ValueError(f"{context}: empty object slot")
        return _join_names([clean_object_name(token) for token in slot])
    raise ValueError(f"{context}: object slot must be a string or list, got {slot!r}")


def _skill_description_text(skill_description: object, *, context: str) -> str:
    """Extract the single template key stored by BEHAVIOR annotations."""
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


def _object_slots(object_id: object, *, context: str) -> list:
    """Unwrap the annotation's one-element outer object-slot list."""
    if not isinstance(object_id, (list, tuple)):
        raise ValueError(f"{context}: object_id must be a list, got {object_id!r}")
    if len(object_id) != 1 or not isinstance(object_id[0], (list, tuple)):
        raise ValueError(
            f"{context}: object_id must be a one-element list wrapping the slot "
            f"list, got {object_id!r}"
        )
    return list(object_id[0])


def _referenced_slot_count(template: str) -> int:
    """Return the number of positional object slots referenced by a template."""
    indices = [int(match) for match in re.findall(r"\{(\d+)\}", template)]
    return max(indices) + 1 if indices else 0


def entry_to_subtask_text(
    entry: dict,
    *,
    task_name: object = None,
    episode_id: object = None,
    skill_idx: object = None,
) -> str:
    """Convert one skill annotation entry to natural-language supervision."""
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
    referenced = _referenced_slot_count(template)
    if len(slots) < referenced:
        raise ValueError(
            f"{context}: template {template_key!r} needs {referenced} object "
            f"slot(s) but the annotation has {len(slots)}"
        )
    fillers = [
        _clean_slot(slots[index], context=f"{context} template {template_key!r}")
        for index in range(referenced)
    ]
    return template.format(*fillers)


def behavior_task_names() -> list[str]:
    """Return the canonical 50 BEHAVIOR task names in task-index order."""
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        TASK_NAMES_TO_INDICES,
    )

    return [
        name
        for name, _ in sorted(TASK_NAMES_TO_INDICES.items(), key=lambda item: item[1])
    ]
