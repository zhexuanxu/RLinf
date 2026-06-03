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

"""Direct-task data parity for the BEHAVIOR SFT transform.

A true parity check against the reference ``create_behavior_data_loader_torch``
needs the reference venv + installed ``openpi``. Instead this asserts parity
against the Phase-1 eval ``BehaviorEvalProcessor``, which was itself verified
byte-for-byte against the old installed-``openpi`` transforms — so matching it is
transitive parity with the reference pipeline. The numerically meaningful,
image-orientation-independent checks are the normalized + padded state, the
tokenized discrete-state prompt, and the action normalize-then-pad order.
"""

from __future__ import annotations

import numpy as np
import pytest


def _stats():
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import NormStats

    # q01=0, q99=1 -> normalize_quantile maps x in [0,1] to [-1,1].
    s = NormStats(
        mean=np.zeros(32, dtype=np.float32),
        std=np.ones(32, dtype=np.float32),
        q01=np.zeros(32, dtype=np.float32),
        q99=np.ones(32, dtype=np.float32),
    )
    return {"state": s, "actions": s}


def test_sft_transform_matches_eval_processor():
    pytest.importorskip("torch")

    from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_transform import (
        BehaviorSftTransform,
        transform_behavior_sft_item,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import normalize_quantile
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.processing import (
        BehaviorEvalProcessor,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer import PaligemmaTokenizer

    norm_stats = _stats()
    tokenizer = PaligemmaTokenizer(max_len=200)
    rng = np.random.default_rng(0)

    head = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    left = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    right = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
    proprio = rng.random(256).astype(np.float32)
    raw_action = rng.random((32, 23)).astype(np.float32)
    task = "turn on radio"

    # --- SFT transform path: one streamed LeRobot frame -> model-input item. ---
    sft = BehaviorSftTransform(
        norm_stats=norm_stats, action_dim=32, tokenizer=tokenizer
    )
    frame = {
        # streamed images are channel-first (C, H, W); BehaviorInputs handles it.
        "observation.images.rgb.head": np.transpose(head, (2, 0, 1)),
        "observation.images.rgb.left_wrist": np.transpose(left, (2, 0, 1)),
        "observation.images.rgb.right_wrist": np.transpose(right, (2, 0, 1)),
        "observation.state": proprio,
        "action": raw_action,
        "task": task,
    }
    item = transform_behavior_sft_item(frame, sft)

    # --- Eval processor path: equivalent env_obs -> Observation. ---
    processor = BehaviorEvalProcessor(
        norm_stats,
        tokenizer,
        action_chunk=32,
        action_env_dim=23,
        model_action_dim=32,
    )
    env_obs = {
        "main_images": np.stack([head]),  # (1, H, W, 3)
        "wrist_images": np.stack([np.stack([left, right])]),  # (1, 2, H, W, 3)
        "states": np.stack([proprio]),  # (1, 256)
        "task_descriptions": [task],
    }
    obs = processor.build_observation(env_obs, device="cpu")

    # State: same 23-dim extraction -> same quantile-normalize -> same pad to 32.
    np.testing.assert_allclose(
        item["state"], obs.state[0].cpu().numpy(), rtol=1e-5, atol=1e-5
    )
    # Tokenized discrete-state prompt is identical.
    np.testing.assert_array_equal(
        np.asarray(item["tokenized_prompt"]),
        obs.tokenized_prompt[0].cpu().numpy(),
    )
    # Images resized to the model resolution.
    assert tuple(np.asarray(item["image"]["base_0_rgb"]).shape) == (224, 224, 3)

    # Actions: normalize THEN pad to 32 (head normalized, tail exactly zero).
    expected = normalize_quantile(raw_action, norm_stats["actions"]).astype(np.float32)
    np.testing.assert_allclose(item["actions"][..., :23], expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(item["actions"][..., 23:], np.zeros((32, 9), np.float32))
