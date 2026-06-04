#!/usr/bin/env python
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

"""Dump the RLinf production BEHAVIOR stream identity (single process).

For the first N emitted samples of the production ``turning_on_radio`` /
``use_skill:false`` loader, prints ``(global_frame_idx, episode_index,
frame_index)`` and summary statistics (unique frames, contiguous-step
transitions, episodes touched). Used to characterise the streaming order/coverage
when investigating the first-50-step training-loss divergence
(``docs/sft-first50-step-evidence.md``): the RLinf loader streams long contiguous
runs of consecutive frames within a keyframe chunk, the same design as the
reference ``BehaviorLeRobotDataset``.

Run (CPU)::

    TMPDIR=/mnt/public/xzxuan/tmp MUJOCO_GL=egl PYTHONPATH=. \\
        python tools/sft_stream_identity_probe.py
"""

from omegaconf import OmegaConf

N = 256
DATA = "/mnt/public/xzxuan/data/2025-challenge-demos"
ASSETS = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"


def main():
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi_pytorch",
                    "num_action_chunks": 32,
                    "openpi": {
                        "assets_dir": ASSETS,
                        "asset_id": "behavior-1k/2025-challenge-demos",
                        "model_action_dim": 32,
                        "max_token_len": 200,
                    },
                },
                "micro_batch_size": 32,
                "eval_batch_size": 1,
                "seed": 42,
            },
            "data": {
                "train_data_paths": DATA,
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": False,
            },
        }
    )
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    loader, _ = build_behavior_sft_dataloader(cfg, 1, 0, DATA)
    raw = loader.torch_loader.dataset._dataset
    fps = raw.fps
    print(
        f"[rlinf] keyframe chunks={len(raw.chunks)} ; fps={fps} ; seed={raw.seed}",
        flush=True,
    )

    rows = []
    for i in range(N):
        gfi = raw.current_streaming_frame_idx
        item = raw[i]
        rows.append(
            (
                gfi,
                int(item["episode_index"].item()),
                round(item["timestamp"].item() * fps),
            )
        )

    eps = [r[1] for r in rows]
    fidx = [r[2] for r in rows]
    contig = sum(
        1 for k in range(1, N) if eps[k] == eps[k - 1] and fidx[k] == fidx[k - 1] + 1
    )
    uniq = len({(e, f) for e, f in zip(eps, fidx)})
    print(
        f"[rlinf] first {N}: unique (ep,frame)={uniq} ; contiguous-step transitions={contig}/{N - 1}",
        flush=True,
    )
    print(
        f"[rlinf] episodes touched (n={len(set(eps))}): {sorted(set(eps))[:20]}",
        flush=True,
    )
    print("[rlinf] first 12 (ep,frame):", [(e, f) for _, e, f in rows[:12]], flush=True)


if __name__ == "__main__":
    main()
