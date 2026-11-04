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

"""Render RPent planner chat requests into prompt token IDs and images."""

from __future__ import annotations

import base64
import binascii
import copy
import io
import json
from dataclasses import dataclass, field
from typing import Any

from rlinf.agents.tokenization import AgenticTokenizer

IMAGE_DATA_PREFIX = "data:"
"""A data URI carries the bytes of one image inside its `image_url` part."""


class TemplateMessageError(ValueError):
    """A planner message the renderer cannot turn into prompt tokens."""


def _image_bytes(part: dict) -> bytes | None:
    """Decode one `image_url` content part, or return None if it is not one."""
    if part.get("type") != "image_url":
        return None
    url = (part.get("image_url") or {}).get("url")
    if not isinstance(url, str) or not url.startswith(IMAGE_DATA_PREFIX):
        raise TemplateMessageError(
            f"image parts must carry a base64 data URI, got {url!r}"
        )
    header, _, payload = url.partition(",")
    if ";base64" not in header:
        raise TemplateMessageError(f"image data URI is not base64: {header!r}")
    try:
        return base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise TemplateMessageError(
            f"image data URI does not decode: {error}"
        ) from error


def message_images(messages: list[dict]) -> list[bytes]:
    """Return the images embedded in a request's messages, in order."""
    images = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            images.extend(
                image for part in content if (image := _image_bytes(part)) is not None
            )
    return images


def template_messages(messages: list[dict]) -> list[dict]:
    """Return messages in the form the chat template accepts.

    Tool-call arguments become mappings; calls whose arguments are not a JSON
    object are kept as text in the message content. Image parts keep their
    content list so the chat template can place a vision marker for each one.
    """
    messages = copy.deepcopy(messages)
    for message in messages:
        if message.get("content") is None:
            message["content"] = ""
        if not isinstance(message["content"], str):
            if not isinstance(message["content"], list):
                raise TemplateMessageError(
                    f"RPent prompts need string or content-list messages, got "
                    f"{type(message['content']).__name__}"
                )
            for part in message["content"]:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind not in ("text", "image_url"):
                    raise TemplateMessageError(
                        f"unsupported content part {part!r}; RPent prompts carry "
                        "text and base64 image_url parts"
                    )
                _image_bytes(part) if kind == "image_url" else None
            continue
        calls, malformed = [], []
        for call in message.get("tool_calls", []):
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            if isinstance(arguments, dict):
                call["function"]["arguments"] = arguments
                calls.append(call)
            else:
                malformed.append(json.dumps(call, ensure_ascii=False))
        if malformed:
            message["content"] += "\n" + "\n".join(malformed)
            message["tool_calls"] = calls
    return messages


@dataclass
class ImagePrompt:
    """Prompt tokens of one image-bearing planner request and its images."""

    input_ids: list[int]
    images: list[bytes] = field(default_factory=list)
    """PNG bytes of each image, in the order the template placed its marker."""
    image_grid_thw: Any = None
    """Processor token grid of the images, kept for the training forward."""


def render_prompt(tokenizer: AgenticTokenizer, request: dict) -> list[int]:
    """Render a whole chat request with the model's chat template."""
    text = tokenizer.apply_chat_template(
        template_messages(request["messages"]),
        tools=request.get("tools"),
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.tokenizer(text, add_special_tokens=False)["input_ids"]


def request_has_images(request: dict) -> bool:
    """Whether a planner request carries image content parts."""
    return any(
        isinstance(message.get("content"), list) for message in request["messages"]
    )


def render_image_prompt(
    tokenizer: AgenticTokenizer,
    request: dict,
    processor: Any,
) -> ImagePrompt:
    """Render an image-bearing request into tokens the image processor expanded.

    The chat template writes one vision marker per image, so the plain render
    holds a single placeholder for each. Expanding it with the model's
    processor turns every placeholder into its grid of pad tokens, which is
    what both the served engine and the training forward score.
    """
    from PIL import Image

    messages = template_messages(request["messages"])
    images = message_images(messages)
    if not images:
        raise TemplateMessageError("render_image_prompt needs at least one image")
    text = tokenizer.apply_chat_template(
        messages, tools=request.get("tools"), tokenize=False, add_generation_prompt=True
    )
    decoded = [Image.open(io.BytesIO(image)).convert("RGB") for image in images]
    inputs = processor(text=[text], images=decoded, return_tensors="pt")
    return ImagePrompt(
        input_ids=inputs["input_ids"][0].tolist(),
        images=images,
        image_grid_thw=inputs["image_grid_thw"],
    )


class AppendOnlyHistory:
    """Build each turn's prompt by appending to the previous turn's sampled tokens.

    A request that continues the recorded conversation (same earlier messages
    and tools, the recorded assistant reply, then only user or tool messages)
    becomes the previous prompt plus the sampled response plus the new
    messages, so every turn is a token prefix of the next. Any other request,
    such as a compacted history, is rendered in full and starts a new chain.
    """

    def __init__(self, tokenizer: AgenticTokenizer) -> None:
        self.tokenizer = tokenizer
        self._messages = None
        self._tools = None
        self._assistant_message = None
        self._ids = None

    def _continues(self, request: dict) -> bool:
        messages = request["messages"]
        count = 0 if self._messages is None else len(self._messages)
        return (
            self._messages is not None
            and request.get("tools") == self._tools
            and len(messages) > count + 1
            and messages[:count] == self._messages
            and template_messages([messages[count]])[0] == self._assistant_message
            and all(m.get("role") in {"user", "tool"} for m in messages[count + 1 :])
        )

    def render(self, request: dict) -> list[int]:
        """Return the prompt token IDs for a chat request."""
        if not self._continues(request):
            return render_prompt(self.tokenizer, request)
        new_messages = template_messages(request["messages"][len(self._messages) + 1 :])
        return self.tokenizer.build_native_tool_response(self._ids, new_messages)

    def record(
        self,
        request: dict,
        prompt_ids: list[int],
        response_ids: list[int],
        assistant_message: dict,
    ) -> None:
        """Make a sampled turn the base of the next request."""
        self._messages = copy.deepcopy(request["messages"])
        self._tools = copy.deepcopy(request.get("tools"))
        self._assistant_message = template_messages([assistant_message])[0]
        self._ids = [*prompt_ids, *response_ids]
