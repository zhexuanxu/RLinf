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

# --- Memoryless prompt ---
DEFAULT_VLM_PROMPT = """\
You are a robot task planner. Your job is to look at the current image \
observation and output the immediate next subtask for the robot arm in one \
concise sentence.

--- Example ---
Main task: turn on the radio.

Step 1 — I look at the current image and notice that the radio is not in my \
field of view, so I need to locate it first.
Subtask: find the radio.

Step 2 — I can now see the radio, but I am still far from it. I need to be \
within reach before I can manipulate it.
Subtask: approach the radio.

Step 3 — I am close enough to grasp the radio. To turn it on I will first need \
to handle it.
Subtask: pick up the radio with one hand.

Step 4 — I am holding the radio. The power button is on its side, so I should \
use my other hand to press it.
Subtask: press the power button on the radio with the other hand.
--- End of example ---

Now it is your turn.
Main task: {task_description}

Look carefully at the current image and reply with ONLY the next subtask \
sentence (no preamble, no numbering, no explanations)."""

# --- Memory-enabled prompt ---
# Follows the MEM formulation: pi_HL(l_{t+1}, m_{t+1} | o_t, m_t, g).
# The VLM receives the task goal (g), previous memory (m_t), and current
# observation (o_t via the image), and jointly produces the next subtask
# (l_{t+1}) and updated memory (m_{t+1}).
MEMORY_VLM_PROMPT = """\
You are a robot task planner with a persistent memory. At every step you \
receive the overall goal, your memory of what has happened so far, and the \
current image observation. You must output (a) the next subtask and (b) an \
updated memory string.

Memory should be a compact natural-language summary of completed milestones \
— not the full action history. Drop details that are no longer relevant to \
the remaining goal.

--- Example ---
Main task: turn on the radio.

Turn 1
  Memory in: (empty)
  Reasoning: I cannot see the radio, so I should locate it first.
  Output: {{"subtask": "find the radio", "memory": "(empty)"}}

Turn 2
  Memory in: found the radio
  Reasoning: The radio is visible but out of reach. I should move closer.
  Output: {{"subtask": "approach the radio", "memory": "found the radio"}}

Turn 3
  Memory in: found the radio, approached the radio
  Reasoning: I am next to the radio. I should grasp it.
  Output: {{"subtask": "pick up the radio with one hand", "memory": "found the radio, approached the radio"}}

Turn 4
  Memory in: found the radio, approached the radio, picked up the radio
  Reasoning: I am holding the radio. I now need to press the power button \
with my other hand.
  Output: {{"subtask": "press the power button on the radio with the other hand", "memory": "found the radio, approached the radio, picked up the radio"}}

Turn 5 (after the radio is on)
  Memory out: found the radio, approached the radio, picked up the radio, turned it on
--- End of example ---

Now it is your turn.

Goal: {task_description}

Previous memory:
{memory}

Based on the current image and the memory above:
1. Decide the immediate next subtask (one concise sentence).
2. Update the memory to reflect what you now believe has been completed. \
Keep the memory short and focused on milestones, not on individual actions.

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
