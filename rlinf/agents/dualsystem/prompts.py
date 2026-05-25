# Copyright 2025 The RLinf Authors.
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

"""Prompt templates and output parsers for the dual-system embodied pipeline.

This module is the **single source of truth** for VLM input/output format,
shared by both SFT training (``behavior_vlm_data_loader``) and dual-system
eval (``dual_system_agent_loop``).

Three independent boolean flags control the format:

- ``enable_reasoning``: include ``<think>`` block in output
- ``enable_memory``: include Old Memory in input and ``<memory>`` block in output
- ``simple_skill``: subtask is one of 4 short skill labels vs. a full sentence
"""

import logging
import re

logger = logging.getLogger(__name__)


# ======================================================================
# Prompt building (input side)
# ======================================================================


def build_vlm_user_text(
    task_description: str,
    memory: str = "",
    enable_memory: bool = False,
    skill_aug: list[str] | None = None,
    subtask_aug: list[str] | None = None,
    aug_option: str = "none",
) -> str:
    """Build the user-message text for the VLM.

    When ``enable_memory=True``, includes Old Memory line.
    When ``enable_memory=False``, omits it entirely.

    ``aug_option`` controls skill/subtask augmentation in the prompt:

    - ``"skill"``: tells the model which skills are available (global)
    - ``"subtask"``: tells the model which subtasks are valid for this task
    - ``"none"``: no augmentation (original behavior)
    """
    parts = [f"Main Task: {task_description}"]
    if enable_memory:
        parts.append(f"Old Memory: {memory}")

    if aug_option == "skill" and skill_aug:
        skill_list_str = ", ".join(f'"{s}"' for s in skill_aug)
        parts.append(
            f"A subtask is composed of a skill and optional objects. "
            f"Available skills: [{skill_list_str}]"
        )
    elif aug_option == "subtask" and subtask_aug:
        subtask_list_str = ", ".join(f'"{s}"' for s in subtask_aug)
        parts.append(
            f"Your output subtask must be one of: [{subtask_list_str}]"
        )

    parts.append("What the next subtask should being performed now?")
    return "\n".join(parts)


# ======================================================================
# Answer building (output / label side)
# ======================================================================


def build_vlm_answer(
    reasoning: str = "",
    memory: str = "",
    subtask: str = "",
    enable_reasoning: bool = True,
    enable_memory: bool = True,
) -> str:
    """Build the expected assistant answer using XML tags.

    Only the tags enabled by the flags are included; disabled tags are
    omitted entirely (not left empty).
    """
    parts = []
    if enable_reasoning:
        parts.append(f"<think>\n{reasoning}\n</think>")
    if enable_memory:
        parts.append(f"<memory>\n{memory}\n</memory>")
    parts.append(f"<subtask>\n{subtask}\n</subtask>")
    return "\n".join(parts)


# ======================================================================
# Output parsing (eval side)
# ======================================================================


def _extract_tag_content(text: str, tag: str) -> str:
    """Extract text between ``<tag>`` and ``</tag>``, or return empty string."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_vlm_output(text: str) -> tuple[str, str, str]:
    """Parse VLM output with XML tags.

    Extracts content from ``<think>``, ``<memory>``, and ``<subtask>`` tags.
    Missing tags return empty strings.

    Returns:
        ``(reasoning, memory, subtask)``
    """
    text = text.strip()

    reasoning = _extract_tag_content(text, "think")
    memory = _extract_tag_content(text, "memory")
    subtask = _extract_tag_content(text, "subtask")

    # Fallback: if no <subtask> tag found, use the full text (after stripping
    # other tags) as the subtask.
    if not subtask:
        fallback = text
        fallback = re.sub(r"<think>.*?</think>", "", fallback, flags=re.DOTALL | re.IGNORECASE)
        fallback = re.sub(r"<memory>.*?</memory>", "", fallback, flags=re.DOTALL | re.IGNORECASE)
        fallback = fallback.strip()
        if fallback:
            subtask = fallback
        else:
            logger.warning("Failed to parse VLM output; raw text: %s", text[:200])
            subtask = "continue current action"

    return reasoning, memory, subtask
