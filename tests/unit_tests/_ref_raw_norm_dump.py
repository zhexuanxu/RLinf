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

"""Dump the reference RAW first batch + its transform + loss (AC-4 root-cause).

Run with the reference (``openpi-comet-pytorch-mixed``) Python 3.11 venv on a GPU,
using the CANONICAL (NEW) norm-stats the reference SFT resolves:

    .venv/bin/python tests/unit_tests/_ref_raw_norm_dump.py <out_dir>

For the first ``N`` frames it records, per frame:
* the RAW LeRobot fields BEFORE any transform (``transformed._dataset[i]``): the
  three raw camera images, the raw state, the raw action chunk, and the task text
  -- plus a sha256 over each raw field (so the RLinf arm proves it consumed the
  IDENTICAL raw batch, not a pre-normalized one);
* the reference per-sample transform output (``transformed._transform(frame)``):
  the quantile-NORMALIZED+padded state and actions, the tokenized prompt+mask, and
  the three resized images.

It then collates those transformed items into the reference Pi0's ``Observation``
and computes ``compute_loss(train=True, rng=None, noise, time)`` (the production
deterministic-crop path) with numpy-generated noise/time, so the RLinf arm
reproduces the same-batch loss exactly. Writes ``ref_raw_norm.npz`` and prints one
``REF_RAW_NORM <json>`` line (hashes + loss + provenance).
"""

import dataclasses
import hashlib
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
# The canonical (NEW) reference-resolved norm-stats (AC-1 sha ff7e1ff0...).
_ASSETS_DIR = (
    "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/"
    "pi05_b1k-task0000_sft_pytorch_mixed"
)
_ASSET_ID = "behavior-1k/2025-challenge-demos"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_N = 8
_SEED = 1234
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _sha(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr))
    return hashlib.sha256(a.tobytes()).hexdigest()


def _raw(frame, *keys):
    for k in keys:
        if k in frame:
            return np.asarray(frame[k])
    raise KeyError(f"none of {keys} in raw frame keys {list(frame)[:12]}")


def main(out_dir):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import openpi.models_pytorch_new.pi0 as pi0_new
        import openpi.models_pytorch_new.pi0_config as pi0_config_new
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader
        import safetensors.torch
        import torch

        device = "cuda"
        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(
            cfg.data.base_config, behavior_dataset_root=_DATA_ROOT
        )
        assets = _config.AssetsConfig(assets_dir=_ASSETS_DIR, asset_id=_ASSET_ID)
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        run_cfg = dataclasses.replace(cfg, data=data, num_workers=0)

        data_config = run_cfg.data.create(run_cfg.assets_dirs, run_cfg.model)
        raw_ds = _data_loader.create_behavior_dataset(
            data_config, action_horizon=run_cfg.model.action_horizon
        )
        transformed = _data_loader.transform_dataset(
            raw_ds, data_config, skip_norm_stats=False
        )

        store, raw_store, tasks = {}, {}, []
        for i in range(_N):
            frame = transformed._dataset[i]
            item = transformed._transform(frame)
            raw_store.setdefault("raw_head", []).append(
                _raw(frame, "observation.images.rgb.head")
            )
            raw_store.setdefault("raw_left", []).append(
                _raw(frame, "observation.images.rgb.left_wrist")
            )
            raw_store.setdefault("raw_right", []).append(
                _raw(frame, "observation.images.rgb.right_wrist")
            )
            raw_store.setdefault("raw_state", []).append(
                _raw(frame, "observation.state")
            )
            raw_store.setdefault("raw_action", []).append(
                _raw(frame, "action", "actions")
            )
            t = frame.get("task", frame.get("prompt", ""))
            tasks.append(str(t.item() if hasattr(t, "item") else t))
            imgs = item["image"]
            store.setdefault("state", []).append(_np(item["state"]))
            store.setdefault("actions", []).append(_np(item["actions"]))
            store.setdefault("tokenized_prompt", []).append(
                _np(item["tokenized_prompt"])
            )
            store.setdefault("tokenized_prompt_mask", []).append(
                _np(item["tokenized_prompt_mask"])
            )
            for k in _IMG:
                store.setdefault(f"image__{k}", []).append(_np(imgs[k]))

        batch = {k: np.stack(v) for k, v in store.items()}
        raw_batch = {k: np.stack(v) for k, v in raw_store.items()}
        raw_hashes = {k: _sha(v) for k, v in raw_batch.items()}

        # Reference forward loss on this exact transformed batch + numpy noise/time.
        model = pi0_new.Pi0(pi0_config_new.Pi0Config(**cfg.model.__dict__)).to(device)
        model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
        model = model.to(torch.bfloat16).eval()
        from openpi.models_pytorch_new import model as omodel

        actions = torch.from_numpy(batch["actions"]).to(device).float()
        rs = np.random.RandomState(_SEED)
        np_noise = rs.randn(*batch["actions"].shape).astype(np.float32)
        np_time = rs.beta(1.5, 1.0, size=(_N,)).astype(np.float32) * 0.999 + 0.001
        noise = torch.from_numpy(np_noise).to(device)
        time = torch.from_numpy(np_time).to(device)
        obs = omodel.Observation.from_dict(
            {
                "image": {
                    k: torch.from_numpy(batch[f"image__{k}"]).to(device) for k in _IMG
                },
                "image_mask": {
                    k: torch.ones(_N, dtype=torch.bool, device=device) for k in _IMG
                },
                "state": torch.from_numpy(batch["state"]).to(device).float(),
                "tokenized_prompt": torch.from_numpy(batch["tokenized_prompt"]).to(
                    device
                ),
                "tokenized_prompt_mask": torch.from_numpy(
                    batch["tokenized_prompt_mask"]
                ).to(device),
            }
        )
        with torch.no_grad():
            ref_loss = float(
                model.compute_loss(
                    obs, actions, train=True, rng=None, noise=noise, time=time
                )
                .float()
                .mean()
            )

        np.savez(
            f"{out_dir}/ref_raw_norm.npz",
            noise=np_noise,
            time=np_time,
            **{f"ref_{k}": v for k, v in batch.items()},
            **raw_batch,
        )
        result.update(
            ok=True,
            ref_loss=ref_loss,
            raw_hashes=raw_hashes,
            n=_N,
            tasks=tasks,
            assets_dir=_ASSETS_DIR,
            asset_id=_ASSET_ID,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        result["err"] = f"{type(e).__name__}: {str(e)[:500]}"

    print("REF_RAW_NORM " + json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
