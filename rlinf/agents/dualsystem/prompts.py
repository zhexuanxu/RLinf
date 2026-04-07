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

"""Prompt templates and output parsers for the dual-system embodied agentloop.

Two modes are supported:

1. **Memoryless** (default): The VLM outputs only a subtask string.
2. **Memory-enabled**: Inspired by the MEM paper
   (pi_HL(l_{t+1}, m_{t+1} | o_t, m_t, g)), the VLM outputs a JSON object
   containing both the next ``subtask`` and an updated ``memory`` summary.
   The memory is a compressed natural-language summary of all semantically
   relevant events so far, allowing the VLM to leverage long-horizon context
   without an ever-growing prompt.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ======================================================================
# Prompt templates
# ======================================================================

# --- Memoryless prompt (original) ---
DEFAULT_VLM_PROMPT = (
    "Task: {task_description}. Based on the image, describe the immediate "
    "next subtask for the robot arm in one sentence."
)

# --- Memory-enabled prompt ---
# Follows the MEM formulation: pi_HL(l_{t+1}, m_{t+1} | o_t, m_t, g).
# The VLM receives the task goal (g), previous memory (m_t), and current
# observation (o_t via the image), and jointly produces the next subtask
# (l_{t+1}) and updated memory (m_{t+1}).
MEMORY_VLM_PROMPT = """\
You are a robot task planner with a persistent memory.

Goal: {task_description}

Previous memory (summary of what has happened so far):
{memory}

Based on the current image observation and your previous memory:
1. Decide the immediate next subtask for the robot arm (one concise sentence).
2. Update the memory to incorporate what you observe now. Keep only information \
that is relevant for completing the remaining goal. Compress or discard details \
that are no longer useful.

Respond with ONLY a JSON object in this exact format (no other text):
{{"subtask": "<next subtask>", "memory": "<updated memory>"}}"""

# ======================================================================
# Output parsers
# ======================================================================


def parse_subtask_only(text: str) -> tuple[str, None]:
    """Parse VLM output in memoryless mode.

    Returns:
        ``(subtask, None)`` — the full text is the subtask; memory is ``None``.
    """
    return text.strip(), None


def parse_subtask_and_memory(text: str) -> tuple[str, str]:
    """Parse VLM JSON output in memory-enabled mode.

    Expected format::

        {"subtask": "...", "memory": "..."}

    Handles common LLM quirks: markdown code fences, trailing commas, extra
    whitespace.  Falls back gracefully: if parsing fails entirely, the raw
    text is used as the subtask and the memory is left unchanged (returned
    as empty string so the caller can keep the previous memory).

    Returns:
        ``(subtask, memory)`` extracted from the JSON.
    """
    cleaned = text.strip()

    # Strip markdown code fences if present.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Try to extract a JSON object from the text.
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            subtask = str(data.get("subtask", "")).strip()
            memory = str(data.get("memory", "")).strip()
            if subtask:
                return subtask, memory
        except json.JSONDecodeError:
            pass

    # Fallback: try to find subtask/memory with regex patterns.
    subtask_match = re.search(
        r'"subtask"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
    )
    memory_match = re.search(
        r'"memory"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
    )
    if subtask_match:
        subtask = subtask_match.group(1).strip()
        memory = memory_match.group(1).strip() if memory_match else ""
        return subtask, memory

    # Last resort: treat the whole text as subtask, signal empty memory.
    logger.warning(
        "Failed to parse VLM JSON output; using raw text as subtask. "
        "Output: %s",
        text[:200],
    )
    return cleaned if cleaned else "continue current action", ""
