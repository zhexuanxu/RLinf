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

"""Dump the REFERENCE loader's first-N batch sequence (no training) for the R30 replay.

Run in the reference ``openpi-comet-pytorch-mixed`` py-3.11 venv:

    .venv/bin/python tests/unit_tests/_ref_seq_batch_dump.py <out_dir> <n_steps> <batch_size>

Draws the first ``n_steps`` batches of ``batch_size`` from the real
``create_behavior_data_loader_torch`` (config ``pi05_b1k-turning_on_radio...``,
``shuffle=True``) and writes ``<out_dir>/ref_seq_batches.npz`` (keys ``image__<k>``,
``image_mask__<k>``, ``state``, ``tokenized_prompt``, ``tokenized_prompt_mask``,
``actions``; each shape ``(n_steps, batch_size, ...)``). The R30 orchestrator
``tools/sft_loader_sequence_probe.py`` replays this sequence (and RLinf's own loader's
sequence) through RLinf's training loop and compares the trajectories.
"""

import dataclasses
import json
import sys

import numpy as np

_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"
_ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
_CONFIG = "pi05_b1k-turning_on_radio_cs32_bs32_lr2.5e-5_step30k"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def main(out_dir, n_steps, batch_size):
    sys.path.insert(0, _REF_SRC)
    result = {"ok": False}
    try:
        import openpi.training.config as _config
        import openpi.training.data_loader as _data_loader

        cfg = _config.get_config(_CONFIG)
        base = dataclasses.replace(cfg.data.base_config, behavior_dataset_root=_DATA_ROOT)
        assets = _config.AssetsConfig(
            assets_dir=_ASSETS_DIR, asset_id="behavior-1k/2025-challenge-demos"
        )
        data = dataclasses.replace(cfg.data, base_config=base, assets=assets)
        run_cfg = dataclasses.replace(cfg, data=data, batch_size=batch_size, num_workers=0)
        it = iter(_data_loader.create_behavior_data_loader_torch(run_cfg, shuffle=True))

        store = {}
        for _ in range(n_steps):
            observation, actions = next(it)
            for k in _IMG:
                store.setdefault(f"image__{k}", []).append(_np(observation.images[k]))
                m = getattr(observation, "image_masks", None)
                if m is not None:
                    store.setdefault(f"image_mask__{k}", []).append(_np(m[k]))
            store.setdefault("state", []).append(_np(observation.state))
            store.setdefault("tokenized_prompt", []).append(_np(observation.tokenized_prompt))
            tpm = getattr(observation, "tokenized_prompt_mask", None)
            if tpm is not None:
                store.setdefault("tokenized_prompt_mask", []).append(_np(tpm))
            store.setdefault("actions", []).append(_np(actions))

        np.savez(
            f"{out_dir}/ref_seq_batches.npz",
            **{k: np.stack(v) for k, v in store.items()},
        )
        result.update(ok=True, config=_CONFIG, n_steps=n_steps, batch_size=batch_size)
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        result["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        result["tb"] = traceback.format_exc()[-1000:]

    print("REF_SEQ_BATCH " + json.dumps({k: result[k] for k in ("ok", "err") if k in result}))
    if "tb" in result and not result["ok"]:
        print(result["tb"])


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    bs = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    main(out, n, bs)
