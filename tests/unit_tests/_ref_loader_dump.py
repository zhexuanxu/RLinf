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

"""Dump reference (openpi-comet-pytorch-mixed) artifacts for parity comparison.

Run with the reference's Python 3.11 venv and the reference src on sys.path:

    .venv/bin/python tests/unit_tests/_ref_loader_dump.py <out_dir>

Writes <out_dir>/ref_tokenizer.npz (exact tokenizer parity inputs/outputs) and,
best-effort, <out_dir>/ref_batch.npz (first transformed batch from the real
``create_behavior_data_loader_torch``). Prints a single ``REF_DUMP_RESULT <json>``
line describing what succeeded, so a skip-gated test can react without parsing
tracebacks.
"""

import dataclasses
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"

# Fixed (prompt, 23-dim normalized state) inputs for exact tokenizer parity.
_TOK_INPUTS = [
    ("turning on radio", np.linspace(-1.0, 1.0, 23, dtype=np.float32)),
    ("turn on the radio", np.zeros(23, dtype=np.float32)),
    ("pick up radio from coffee table", np.full(23, 0.3, dtype=np.float32)),
]


def main(out_dir):
    sys.path.insert(0, _REF_SRC)
    result = {"tokenizer_ok": False, "loader_ok": False}

    # 1) Exact tokenizer parity (frame-independent, reliable).
    try:
        from openpi.models.tokenizer import PaligemmaTokenizer

        tok = PaligemmaTokenizer(max_len=200)
        ids_list, mask_list = [], []
        for prompt, state in _TOK_INPUTS:
            ids, mask = tok.tokenize(prompt, state)
            ids_list.append(np.asarray(ids, dtype=np.int64))
            mask_list.append(np.asarray(mask, dtype=bool))
        np.savez(
            f"{out_dir}/ref_tokenizer.npz",
            ids=np.stack(ids_list),
            mask=np.stack(mask_list),
        )
        result["tokenizer_ok"] = True
    except Exception as e:  # pragma: no cover - environment dependent
        result["tokenizer_err"] = f"{type(e).__name__}: {e}"

    # 2) Best-effort: a real batch from create_behavior_data_loader_torch.
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        cfg = dataclasses.replace(cfg, data=data, batch_size=2, num_workers=0)
        loader = _data_loader.create_behavior_data_loader_torch(cfg, shuffle=False)
        observation, actions = next(iter(loader))
        images = observation.images
        first_img_key = sorted(images)[0]
        np.savez(
            f"{out_dir}/ref_batch.npz",
            state=np.asarray(observation.state),
            actions=np.asarray(actions),
            tokenized_prompt=np.asarray(observation.tokenized_prompt),
            image=np.asarray(images[first_img_key]),
            image_key=np.array(first_img_key),
        )
        result["loader_ok"] = True
    except Exception as e:  # pragma: no cover - environment dependent
        result["loader_err"] = f"{type(e).__name__}: {str(e)[:300]}"

    # 3) Exact fixed-sample parity: one raw frame + its reference transform.
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        cfg = dataclasses.replace(cfg, data=data, num_workers=0)

        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        raw = _data_loader.create_behavior_dataset(
            data_config, action_horizon=cfg.model.action_horizon
        )
        transformed = _data_loader.transform_dataset(
            raw, data_config, skip_norm_stats=False
        )
        raw_frame = transformed._dataset[0]
        ref_item = transformed._transform(raw_frame)

        def _g(frame, *keys):
            for k in keys:
                if k in frame:
                    return np.asarray(frame[k])
            raise KeyError(f"none of {keys} in raw frame keys {list(frame)[:12]}")

        task = raw_frame.get("task", raw_frame.get("prompt", ""))
        np.savez(
            f"{out_dir}/ref_raw_frame.npz",
            head=_g(raw_frame, "observation.images.rgb.head"),
            left=_g(raw_frame, "observation.images.rgb.left_wrist"),
            right=_g(raw_frame, "observation.images.rgb.right_wrist"),
            state=_g(raw_frame, "observation.state"),
            action=_g(raw_frame, "action", "actions"),
            task=np.array(str(task.item() if hasattr(task, "item") else task)),
        )
        imgs = ref_item["image"]
        np.savez(
            f"{out_dir}/ref_item.npz",
            state=np.asarray(ref_item["state"]),
            actions=np.asarray(ref_item["actions"]),
            tokenized_prompt=np.asarray(ref_item["tokenized_prompt"]),
            tokenized_prompt_mask=np.asarray(ref_item["tokenized_prompt_mask"]),
            base=np.asarray(imgs["base_0_rgb"]),
            left=np.asarray(imgs["left_wrist_0_rgb"]),
            right=np.asarray(imgs["right_wrist_0_rgb"]),
        )
        result["sample_ok"] = True
    except Exception as e:  # pragma: no cover - environment dependent
        result["sample_err"] = f"{type(e).__name__}: {str(e)[:400]}"

    print("REF_DUMP_RESULT " + json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp")
