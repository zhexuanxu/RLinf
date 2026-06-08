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

"""Shared, deterministic PINNED input for the dual-repo precision dtype ledger.

Both precision-ledger harnesses (RLinf venv and reference venv) consume the SAME
pinned batch so the dtype ledger compares both repos on identical inputs. The batch
is regenerated deterministically from a committed spec (seed + shapes) using numpy's
legacy ``RandomState`` (MT19937), which is byte-stable across numpy versions / venvs
-- so both harnesses produce identical tensors, proven by the per-field sha256 each
records in its ledger. Pure torch/numpy (no repo imports) so it loads in either venv.

This is a value-INDEPENDENT dtype probe (dtypes do not depend on the values), so a
shared deterministic batch is the correct pinned input here; the real-data pinned
batch is used by the controlled-step numeric parity (a later acceptance criterion).
"""

import hashlib
import json
import os

import numpy as np
import torch

_SPEC_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "docs/evidence/phase5_precision_pinned_spec.json",
)
_IMG_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_DEFAULT_SPEC = {
    "seed": 20260608,
    "batch": 1,
    "image_hw": 224,
    "state_dim": 32,
    "action_horizon": 32,
    "action_dim": 32,
    "max_token_len": 200,
    "image_keys": list(_IMG_KEYS),
}


def build_pinned(spec_path=_SPEC_PATH):
    """Write the committed pinned spec (run once)."""
    os.makedirs(os.path.dirname(spec_path), exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(_DEFAULT_SPEC, f, indent=2)
    return spec_path


def sha_tensor(t) -> str:
    a = np.ascontiguousarray(
        t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
    )
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_pinned(device, spec_path=_SPEC_PATH):
    """Deterministically regenerate the pinned batch + per-field sha256."""
    with open(spec_path) as f:
        spec = json.load(f)
    rs = np.random.RandomState(spec["seed"])
    b, hw = spec["batch"], spec["image_hw"]
    sdim, horizon, adim = spec["state_dim"], spec["action_horizon"], spec["action_dim"]
    keys = tuple(spec["image_keys"])

    def f32(shape):
        return rs.standard_normal(size=shape).astype(np.float32)

    images_np = {k: f32((b, hw, hw, 3)) for k in keys}
    state_np = f32((b, sdim))
    actions_np = f32((b, horizon, adim))
    noise_np = f32((b, horizon, adim))
    time_np = rs.random_sample((b,)).astype(np.float32) * 0.998 + 0.001
    tok_np = np.ones((b, spec["max_token_len"]), dtype=np.int64)

    def t(x):
        return torch.from_numpy(x).to(device)

    images = {k: t(v) for k, v in images_np.items()}
    image_masks = {k: torch.ones(b, dtype=torch.bool, device=device) for k in keys}
    state = t(state_np)
    actions = t(actions_np)
    noise = t(noise_np)
    time = t(time_np)
    tokenized_prompt = t(tok_np)
    tokenized_prompt_mask = torch.ones(
        b, spec["max_token_len"], dtype=torch.bool, device=device
    )

    hashes = {
        **{f"image__{k}": sha_tensor(v) for k, v in images_np.items()},
        "state": sha_tensor(state_np),
        "actions": sha_tensor(actions_np),
        "noise": sha_tensor(noise_np),
        "time": sha_tensor(time_np),
        "tokenized_prompt": sha_tensor(tok_np),
    }
    shapes = {
        "image": [b, hw, hw, 3],
        "state": [b, sdim],
        "actions": [b, horizon, adim],
        "noise": [b, horizon, adim],
        "time": [b],
        "tokenized_prompt": [b, spec["max_token_len"]],
    }
    return {
        "spec_path": os.path.relpath(spec_path, os.getcwd())
        if spec_path.startswith(os.getcwd())
        else spec_path,
        "images": images,
        "image_masks": image_masks,
        "state": state,
        "actions": actions,
        "noise": noise,
        "time": time,
        "tokenized_prompt": tokenized_prompt,
        "tokenized_prompt_mask": tokenized_prompt_mask,
        "hashes": hashes,
        "shapes": shapes,
    }


if __name__ == "__main__":
    print("wrote", build_pinned())
