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

These templates are the **single source of truth** for both SFT training
(``behavior_vlm_data_loader``) and dual-system eval
(``dual_system_agent_loop``).

Two modes are supported:

1. **Memoryless** (``enable_memory=False``): The VLM outputs only a subtask
   string.
2. **Memory-enabled** (``enable_memory=True``): The VLM receives old memory
   (Progress + World state) and outputs reasoning, new memory, and subtask.
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# ======================================================================
# Prompt templates (user message text, paired with an image)
# ======================================================================

# --- Memoryless prompt ---
DEFAULT_VLM_PROMPT = (
    "Task: {task_description}\nWhat skill is being performed in this frame?"
)

# --- Memory-enabled prompt ---
MEMORY_VLM_PROMPT = (
    "Main Task: {task_description}\n"
    "Old Memory:\n"
    "- Progress: {progress}\n"
    "- World State: {world_state}"
)

# ======================================================================
# Answer formatting (for SFT label construction)
# ======================================================================


def format_agentic_answer(
    reasoning: str,
    new_progress: str,
    new_world_state: str,
    subtask: str,
) -> str:
    """Build the expected assistant answer for agentic SFT training.

    Format::

        <think>
        {reasoning}
        </think>
        {
            "Progress": "...",
            "World state": "...",
            "subtask": "..."
        }
    """
    return (
        f"<think>\n{reasoning}\n</think>\n"
        f'{{\n'
        f'    "Progress": "{new_progress}",\n'
        f'    "World state": "{new_world_state}",\n'
        f'    "subtask": "{subtask}"\n'
        f'}}'
    )


# ======================================================================
# Output parsers
# ======================================================================


def parse_subtask_only(text: str) -> tuple[str, None]:
    """Parse VLM output in memoryless mode.

    Returns:
        ``(subtask, None)`` — the full text is the subtask; memory is ``None``.
    """
    return text.strip(), None


def parse_agentic_output(text: str) -> tuple[str, dict]:
    """Parse agentic VLM output: optional ``<think>`` block + JSON.

    Expected format::

        <think>
        ...reasoning...
        </think>
        {
            "Progress": "...",
            "World state": "...",
            "subtask": "..."
        }

    Returns:
        ``(subtask, memory_dict)`` where ``memory_dict`` has keys
        ``"Progress"`` and ``"World state"``.  On parse failure the raw text
        is used as subtask and an empty memory dict is returned.
    """
    cleaned = text.strip()

    # Strip <think>...</think> block if present.
    cleaned = re.sub(
        r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    # Strip markdown code fences.
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    empty_mem = {"Progress": "", "World state": ""}

    # Try to extract a JSON object.
    json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if json_match:
        json_str = json_match.group(0)
        try:
            data = json.loads(json_str)
            subtask = str(data.get("subtask", "")).strip()
            memory = {
                "Progress": str(data.get("Progress", "")).strip(),
                "World state": str(data.get("World state", "")).strip(),
            }
            if subtask:
                return subtask, memory
        except json.JSONDecodeError:
            pass

    # Fallback: regex extraction.
    subtask_match = re.search(
        r'"subtask"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
    )
    if subtask_match:
        subtask = subtask_match.group(1).strip()
        progress_match = re.search(
            r'"Progress"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
        )
        world_match = re.search(
            r'"World state"\s*:\s*"([^"]*)"', cleaned, flags=re.DOTALL
        )
        memory = {
            "Progress": progress_match.group(1).strip() if progress_match else "",
            "World state": world_match.group(1).strip() if world_match else "",
        }
        return subtask, memory

    # Last resort.
    logger.warning(
        "Failed to parse agentic VLM output; using raw text as subtask. "
        "Output: %s",
        text[:200],
    )
    return cleaned if cleaned else "continue current action", empty_mem


def extract_reasoning(text: str) -> str:
    """Extract the content of ``<think>...</think>`` from VLM output."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


# ---- Legacy alias kept for backward compatibility ----
def parse_subtask_and_memory(text: str) -> tuple[str, str]:
    """Legacy parser (returns flat memory string). Prefer ``parse_agentic_output``."""
    subtask, mem_dict = parse_agentic_output(text)
    # Flatten to a single string for old callers.
    parts = []
    if mem_dict.get("Progress"):
        parts.append(mem_dict["Progress"])
    if mem_dict.get("World state"):
        parts.append(mem_dict["World state"])
    return subtask, "; ".join(parts) if parts else ""
