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

"""torchrun worker for the R29 8-rank-FSDP vs single-GPU same-input grad comparison.

Launched by ``tools/sft_fsdp_consistency_probe.py`` under two world sizes on the SAME
fixed global batch + fixed noise/time + base weights:

* ``--nproc_per_node=1``: plain single-GPU RLinf ``Pi0`` (bf16 compute) over the FULL
  global batch -> the single-GPU global L2 grad norm + loss. (R26/R28 already proved
  this single-GPU path equals the reference.)
* ``--nproc_per_node=8``: RLinf ``Pi0`` wrapped in FSDP1 with the production
  ``MixedPrecision(param_dtype=bf16, reduce_dtype=fp32, buffer_dtype=fp32)`` + FULL_SHARD,
  each rank fed its CONTIGUOUS shard of the same global batch -> the all-reduced
  (FSDP-sharded) global grad norm + AVG-all-reduced loss.

If the 8-rank all-reduced grad norm == the single-GPU grad norm, RLinf's FSDP
sharding/all-reduce/clip is numerically correct (the distributed step is ruled out as
the cause of the production divergence). The grad norm on both sides is the global L2
norm computed by ``clip_grad_norm_`` (FSDP's method on 8 ranks -- the SAME call the
reference trainer uses, train_pytorch_new.py:535; ``torch.nn.utils`` on 1 rank); R27
code reading confirmed RLinf's ``get_grad_norm_for_mixed_precision`` clip path computes
the same global L2 norm.

The batch is the committed R26 reference dump (``ref_grad_batches.npz``: 4 x 8 = 32 real
behavior frames from the reference loader) reshaped to one 32-frame global batch. Writes
``<tmp>/fsdp_grad_w<world>.json`` from rank 0. GPU-only; output under TMPDIR.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_CLIP = 1.0
_BETAS = (0.9, 0.95)
_EPS = 1e-8
_WD = 1e-10
_LR = 2.5e-5 / 1001  # openpi_cosine init = peak/(warmup+1), step 0


def _load_global_batch(tmp):
    """Reshape the R26 dump (n_batches x B x ...) into one contiguous global batch."""
    d = np.load(f"{tmp}/ref_grad_batches.npz")
    nt = np.load(f"{tmp}/ref_grad_noise_time.npz")
    flat = {}
    for k in d.files:
        a = d[k]  # (n_batches, B, ...)
        flat[k] = a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])
    noise = nt["noise"].reshape(-1, *nt["noise"].shape[2:])
    time = nt["time"].reshape(-1, *nt["time"].shape[2:])
    return flat, noise, time


def _slice(flat, noise, time, lo, hi, device):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    obs = Observation.from_dict(
        {
            "image": {
                k: torch.from_numpy(flat[f"image__{k}"][lo:hi]).to(device, torch.float32)
                for k in _IMG
            },
            "image_mask": {
                k: torch.from_numpy(flat[f"image_mask__{k}"][lo:hi]).to(device) for k in _IMG
            },
            "state": torch.from_numpy(flat["state"][lo:hi]).to(device, torch.float32),
            "tokenized_prompt": torch.from_numpy(flat["tokenized_prompt"][lo:hi]).to(device).long(),
            "tokenized_prompt_mask": torch.from_numpy(flat["tokenized_prompt_mask"][lo:hi])
            .to(device)
            .bool(),
        }
    )
    act = torch.from_numpy(flat["actions"][lo:hi]).to(device, torch.float32)
    nz = torch.from_numpy(noise[lo:hi]).to(device)
    tm = torch.from_numpy(time[lo:hi]).to(device)
    return obs, act, nz, tm


def _build_model(device):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    model = Pi0(Pi0Config(pi05=True, action_horizon=32)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model


def main(tmp):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    flat, noise, time = _load_global_batch(tmp)
    n = flat["state"].shape[0]

    if world > 1:
        import torch.distributed as dist
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        dist.init_process_group("nccl")
        per = n // world
        lo, hi = rank * per, (rank + 1) * per
        model = _build_model(device)
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            ),
            device_id=local_rank,
            use_orig_params=True,
        )
    else:
        lo, hi = 0, n
        model = _build_model(device).to(torch.bfloat16)

    model.train()
    obs, act, nz, tm = _slice(flat, noise, time, lo, hi, device)

    opt = torch.optim.AdamW(model.parameters(), lr=_LR, betas=_BETAS, eps=_EPS, weight_decay=_WD)
    model.zero_grad(set_to_none=True)
    # Call forward (model(...)), NOT compute_loss directly: under FSDP the param
    # all-gather runs in the forward hook, so a direct method call would see sharded params.
    loss = model(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
    loss.backward()

    if world > 1:
        import torch.distributed as dist

        gn = model.clip_grad_norm_(max_norm=_CLIP)  # FSDP1 global L2 norm (all-reduced) + clip
        loss_g = loss.detach().clone()
        dist.all_reduce(loss_g, op=dist.ReduceOp.AVG)
        loss_val = float(loss_g)
    else:
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=_CLIP)
        loss_val = float(loss)
    grad_norm = float(gn)

    opt.step()
    # post-step loss on this rank's slice (re-forward), AVG-reduced for world>1
    with torch.no_grad():
        post = model(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
    if world > 1:
        import torch.distributed as dist

        post_g = post.detach().clone()
        dist.all_reduce(post_g, op=dist.ReduceOp.AVG)
        post_val = float(post_g)
    else:
        post_val = float(post)

    if rank == 0:
        out = {
            "world_size": world,
            "n_frames_global": n,
            "per_rank": (hi - lo),
            "global_grad_norm": round(grad_norm, 5),
            "clipped_grad_norm": round(min(grad_norm, _CLIP), 5),
            "was_clipped": grad_norm > _CLIP,
            "loss": round(loss_val, 6),
            "post_step_loss": round(post_val, 6),
            "clip": _CLIP,
            "lr": _LR,
            "fsdp": world > 1,
        }
        with open(f"{tmp}/fsdp_grad_w{world}.json", "w", newline="\n") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print("FSDP_GRAD_WORKER " + json.dumps(out), flush=True)

    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/mnt/public/xzxuan/tmp")
