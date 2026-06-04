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

"""Fixed-batch forward-loss parity probe for the BEHAVIOR ``use_skill:false`` SFT path.

This is the task14 (AC-11, DEC-1 (a)) GPU evidence harness. It loads
``pi05_base_pytorch_new`` via the RLinf SFT model factory with the canonical
task-0000 norm stats, builds fixed ``turning_on_radio`` batches from the
production loader, and computes the flow-matching loss with explicit fixed
noise/time (``Pi0.compute_loss`` accepts them) so the forward MSE is reproducible.
It reports a deterministic single-batch draw, the marginal E[loss] over multiple
batches (matching the reference per-step 256-sample global batch), and the worker's
exact ``sft_forward`` path, comparing them to the reference step-0 loss
``0.24609375`` under the DEC-1 tolerance (relative 5% or absolute 0.01).

Run (single GPU)::

    TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \\
        PYTHONPATH=. python tools/sft_loss_parity_probe.py

The committed run evidence and the divergence analysis live in
``docs/sft-loss-parity-evidence.md``.
"""

import hashlib
import os

import torch
from omegaconf import OmegaConf

REF_LOSS = 0.24609375  # reference step-0 loss (first-10-step mean 0.243848)
SEED = 42
N_DRAWS = 8
N_BATCHES = 8  # 8 x 32 = 256 samples, matching the reference per-step global batch
MODEL_PATH = "/mnt/public/xzxuan/models/pi05_base_pytorch_new"
ASSETS_DIR = "/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets"
ASSET_ID = "behavior-1k/2025-challenge-demos"
DATA_ROOT = "/mnt/public/xzxuan/data/2025-challenge-demos"


def _sha256(path, limit=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        read = 0
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit and read >= limit:
                break
    return h.hexdigest()[:16]


def _model_cfg():
    return OmegaConf.create(
        {
            "model_type": "openpi_pytorch",
            "model_path": MODEL_PATH,
            "num_action_chunks": 32,
            "action_dim": 23,
            "precision": "bf16",
            "load_for_training": True,
            "openpi": {
                "assets_dir": ASSETS_DIR,
                "asset_id": ASSET_ID,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "num_images_in_input": 3,
                "action_chunk": 32,
                "action_env_dim": 23,
                "load_for_training": True,
            },
        }
    )


def main():
    assert torch.cuda.is_available(), "this probe requires a GPU"
    device = "cuda"

    model_cfg = _model_cfg()
    full_cfg = OmegaConf.create(
        {
            "actor": {
                "model": model_cfg,
                "micro_batch_size": 32,
                "eval_batch_size": 1,
                "seed": SEED,
            },
            "data": {
                "train_data_paths": DATA_ROOT,
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": False,
            },
        }
    )

    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )
    from rlinf.models.embodiment.openpi_pytorch import get_model

    model = get_model(model_cfg).to(device)
    model.train()
    inner = model.model

    loader, _ = build_behavior_sft_dataloader(full_cfg, 1, 0, DATA_ROOT)
    it = iter(loader)
    batches = [
        (model._observation_to_device(o), model._actions_to_device(a))
        for o, a in (next(it) for _ in range(N_BATCHES))
    ]

    def batch_loss(observation, actions, seed, train=True):
        # The augmentation rng must be a CPU generator (torch.randint crop offsets);
        # noise is sampled on-device with a separate CUDA generator. Both fixed so
        # the forward MSE is reproducible.
        B = actions.shape[0]
        cpu_gen = torch.Generator().manual_seed(seed)
        noise_gen = torch.Generator(device=device).manual_seed(seed + 1000)
        noise = torch.randn(
            actions.shape, device=device, dtype=torch.float32, generator=noise_gen
        )
        torch.manual_seed(seed)  # Beta.sample draws from the global RNG
        time = (
            torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
            .sample((B,))
            .to(device=device, dtype=torch.float32)
        )
        time = time * 0.999 + 0.001
        with torch.no_grad():
            per_ts = inner.compute_loss(
                observation, actions, train=train, rng=cpu_gen, noise=noise, time=time
            )
        return float(per_ts.mean().item())

    def marginal(train=True):
        estimates = []
        for d in range(N_DRAWS):
            per_batch = [
                batch_loss(obs, act, SEED + 1 + d * 131 + bi, train=train)
                for bi, (obs, act) in enumerate(batches)
            ]
            estimates.append(sum(per_batch) / len(per_batch))
        mean = sum(estimates) / len(estimates)
        std = (sum((x - mean) ** 2 for x in estimates) / len(estimates)) ** 0.5
        return mean, std, min(estimates), max(estimates)

    det = batch_loss(*batches[0], SEED)
    aug_mean, aug_std, aug_min, aug_max = marginal(train=True)
    noaug_mean, *_ = marginal(train=False)
    with torch.no_grad():
        worker_mean = sum(
            float(model.sft_forward((o, a)).item()) for o, a in batches
        ) / len(batches)

    band = max(0.05 * REF_LOSS, 0.01)
    lo, hi = REF_LOSS - band, REF_LOSS + band

    print("\n==== task14 fixed-batch forward-loss parity ====", flush=True)
    print(f"model_path  = {MODEL_PATH}", flush=True)
    print(
        f"model.safetensors sha256[:16] (first 64MB) = "
        f"{_sha256(os.path.join(MODEL_PATH, 'model.safetensors'), 1 << 26)}",
        flush=True,
    )
    print(
        f"norm_stats sha256[:16] = {_sha256(os.path.join(ASSETS_DIR, ASSET_ID, 'norm_stats.json'))}",
        flush=True,
    )
    print(f"seed = {SEED}; samples per estimate = {N_BATCHES * 32} ({N_BATCHES}x32, turning_on_radio)", flush=True)
    print(f"reference step-0 loss = {REF_LOSS}  (first-10-step mean 0.243848)", flush=True)
    print(f"DEC-1 band (rel5%/abs0.01) = [{lo:.6f}, {hi:.6f}]", flush=True)
    print(f"deterministic single-batch draw (seed {SEED}, 32 samples) = {det:.6f}", flush=True)
    print(
        f"marginal E[loss] train=True  = {aug_mean:.6f}  std={aug_std:.6f} "
        f"min={aug_min:.6f} max={aug_max:.6f}  within DEC-1={lo <= aug_mean <= hi}",
        flush=True,
    )
    print(f"marginal E[loss] train=False = {noaug_mean:.6f}  within DEC-1={lo <= noaug_mean <= hi}", flush=True)
    print(f"worker sft_forward mean (internal sampling) = {worker_mean:.6f}", flush=True)


if __name__ == "__main__":
    main()
