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

"""Step-0 2x2 cross-feed (single GPU): classify the first-step gap data-vs-compute.

Materializes BOTH repos' ACTUAL step-0 global batches (256 fully-transformed frames each):
the RLinf DEFAULT per_rank_stream batch (8 rank loaders, batch 0 each) here, and the
reference rank-0-fanout batch in the reference venv (tests/_ref_step0_2x2_dump.py). A shared,
reproducible flow noise/time is generated reference-side. Then each batch is fed through BOTH
the RLinf Pi0 and the reference Pi0 (vendored byte-identical) holding weights
(pi05_base_pytorch_new) / norm-stats / noise/time / dtype-autocast (fp32 master + bf16 compute)
/ train=True-rng=None identical, computing the production-faithful step-0 loss + pre-clip fp32
GLOBAL grad norm over the 256-frame batch.

The 2x2 (batch_source x model_side): each COLUMN (same batch, both models) agrees within
DEC-1 loss + 2% grad (compute parity / production identical-data grad); each ROW (two batches,
same noise) differs (the data difference). => the first-step gap is DATA-side.

    EMBODIED_PATH=.../examples/sft REPO_PATH=<repo> PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES=0 \
        MUJOCO_GL=egl python tools/sft_cross_feed_2x2.py --out <artifact.json> --tmp <scratch>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess

import numpy as np

_REF_VENV_PY = "/mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python"
_REF_SRC = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src"
_REF_SCR = "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts"
_REPO = "/mnt/public/xzxuan/repos/RLinf_pi05"
_REF_DUMP = "tests/unit_tests/_ref_step0_2x2_dump.py"
_WEIGHTS = "/mnt/public/xzxuan/models/pi05_base_pytorch_new/model.safetensors"
_WORLD_SIZE = 8
_MICRO = 32
_IMG = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_CLIP = 1.0
_DEC1 = 0.01  # bf16 loss band
_GRAD_REL = 0.02  # global grad-norm relative tolerance


def _np(x):
    import torch

    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _content_hash(arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def _obs_to_store(observation, actions):
    d = {"actions": _np(actions)}
    for k in _IMG:
        d[f"image__{k}"] = _np(observation.images[k])
        m = getattr(observation, "image_masks", None)
        if m is not None and k in m:
            d[f"image_mask__{k}"] = _np(m[k])
    d["state"] = _np(observation.state)
    d["tokenized_prompt"] = _np(observation.tokenized_prompt)
    tpm = getattr(observation, "tokenized_prompt_mask", None)
    if tpm is not None:
        d["tokenized_prompt_mask"] = _np(tpm)
    return d


def _frame_ids(micro_stores):
    ids = []
    for st in micro_stores:
        for i in range(st["state"].shape[0]):
            ids.append(hashlib.sha256(
                np.ascontiguousarray(st["state"][i]).tobytes()
                + np.ascontiguousarray(st["tokenized_prompt"][i]).tobytes()
            ).hexdigest()[:16])
    return ids


def _materialize_rlinf_step0(tmp):
    """The RLinf DEFAULT (per_rank_stream) step-0 global batch: 8 rank loaders, batch 0 each."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    cfg_dir = os.path.join(_REPO, "examples/sft/config")
    with initialize_config_dir(version_base="1.1", config_dir=cfg_dir):
        cfg = compose(config_name="behavior_pi05_vla")
    OmegaConf.set_struct(cfg, False)
    cfg.data.loader_mode = "per_rank_stream"  # the DEFAULT (different-data) step-0
    dp = OmegaConf.select(cfg, "data.train_data_paths")
    micros = []
    for r in range(_WORLD_SIZE):
        loader, _ = build_behavior_sft_dataloader(
            cfg, world_size=_WORLD_SIZE, rank=r, data_paths=dp, eval_dataset=False
        )
        observation, actions = next(iter(loader))
        micros.append(_obs_to_store(observation, actions))
        del loader
    flat = {f"m{i}__{k}": v for i, st in enumerate(micros) for k, v in st.items()}
    np.savez(os.path.join(tmp, "rlinf_step0_batch.npz"), n_micro=_WORLD_SIZE, **flat)
    return micros


def _rlinf_model(device):
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    import torch

    model = Pi0(Pi0Config(pi05=True, action_horizon=32)).to(device)
    model.load_state_dict(safetensors.torch.load_file(_WEIGHTS), strict=True)
    return model.to(torch.bfloat16).train()


def _obs_from_store(store, device):
    import torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    def t(a, dtype=None):
        x = torch.from_numpy(np.asarray(a)).to(device)
        return x.to(dtype) if dtype is not None else x

    d = {
        "image": {k: t(store[f"image__{k}"], torch.float32) for k in _IMG},
        "image_mask": {k: t(store[f"image_mask__{k}"]) for k in _IMG if f"image_mask__{k}" in store},
        "state": t(store["state"], torch.float32),
        "tokenized_prompt": t(store["tokenized_prompt"]).long(),
    }
    if "tokenized_prompt_mask" in store:
        d["tokenized_prompt_mask"] = t(store["tokenized_prompt_mask"]).bool()
    return Observation.from_dict(d)


def _prod_loss_grad(model, micro_stores, noise, time, device):
    """Production-faithful step-0 loss + pre-clip fp32 GLOBAL grad norm over the 256-frame
    batch (fp32 master + bf16 compute, global-mean loss; mean grad == FSDP all-reduced mean)."""
    import torch

    master = [p.detach().clone().float().requires_grad_(True) for p in model.parameters()]
    params = list(model.parameters())
    with torch.no_grad():
        for p, m in zip(params, master):
            p.data = m.data.to(torch.bfloat16)
    model.zero_grad(set_to_none=True)
    for m in master:
        m.grad = None
    n_micro = len(micro_stores)
    losses = []
    for i, st in enumerate(micro_stores):
        obs = _obs_from_store(st, device)
        act = torch.from_numpy(st["actions"]).to(device, torch.float32)
        nz = torch.from_numpy(noise[i * _MICRO:(i + 1) * _MICRO]).to(device)
        tm = torch.from_numpy(time[i * _MICRO:(i + 1) * _MICRO]).to(device)
        loss_i = model.compute_loss(obs, act, train=True, rng=None, noise=nz, time=tm).float().mean()
        (loss_i / n_micro).backward()
        losses.append(float(loss_i))
    for p, m in zip(params, master):
        m.grad = p.grad.detach().float() if p.grad is not None else None
    gn = float(torch.nn.utils.clip_grad_norm_(master, max_norm=_CLIP))
    return {"loss": float(np.mean(losses)), "grad_norm": gn}


def _load_micros(npz):
    z = np.load(npz)
    n = int(z["n_micro"])
    return [
        {k.split("__", 1)[1]: z[k] for k in z.files if k.startswith(f"m{i}__")}
        for i in range(n)
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/public/xzxuan/tmp/sft_cross_feed_2x2.json")
    ap.add_argument("--tmp", default="/mnt/public/xzxuan/tmp/p8_2x2")
    args = ap.parse_args()
    os.makedirs(args.tmp, exist_ok=True)
    device = "cuda"

    # 1) materialize the RLinf default step-0 batch (and persist for the reference side).
    rlinf_micros = _materialize_rlinf_step0(args.tmp)

    # 2) reference side: materialize the reference step-0 batch + shared noise/time, and the
    #    two ref-model cells (ref batch / RLinf batch through the reference model).
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REF_SRC}:{_REF_SCR}:{_REPO}"
    subprocess.run(
        [_REF_VENV_PY, _REF_DUMP, args.tmp, os.path.join(args.tmp, "rlinf_step0_batch.npz")],
        cwd=_REPO, env=env, check=True, timeout=2400,
    )
    ref = json.loads(open(os.path.join(args.tmp, "ref_step0_2x2.json")).read())
    assert ref.get("ok"), f"reference dump failed: {ref.get('err')}"

    # 3) RLinf model on BOTH batches with the SAME shared noise/time.
    nt = np.load(os.path.join(args.tmp, "step0_noise_time.npz"))
    noise, time = nt["noise"], nt["time"]
    ref_batch_micros = _load_micros(os.path.join(args.tmp, "ref_step0_batch.npz"))
    model = _rlinf_model(device)
    rlinf_on_ref = _prod_loss_grad(model, ref_batch_micros, noise, time, device)
    rlinf_on_rlinf = _prod_loss_grad(model, rlinf_micros, noise, time, device)

    cells = {
        "ref_batch": {
            "ref_model": ref["cell_ref_batch_ref_model"],
            "rlinf_model": rlinf_on_ref,
        },
        "rlinf_batch": {
            "ref_model": ref["cell_rlinf_batch_ref_model"],
            "rlinf_model": rlinf_on_rlinf,
        },
    }

    def loss_d(b):
        return abs(cells[b]["ref_model"]["loss"] - cells[b]["rlinf_model"]["loss"])

    def grad_rel(b):
        a, c = cells[b]["ref_model"]["grad_norm"], cells[b]["rlinf_model"]["grad_norm"]
        return abs(a - c) / abs(c)

    # data difference: same model, two batches (same noise) -> different loss/grad.
    def loss_data_d(side):
        return abs(cells["ref_batch"][side]["loss"] - cells["rlinf_batch"][side]["loss"])

    rl_frame_ids = _frame_ids(rlinf_micros)
    out = {
        "schema_version": 1,
        "description": ("Step-0 2x2 cross-feed on each repo's ACTUAL materialized step-0 batch "
            "(256 fully-transformed frames) through BOTH models, fp32-master+bf16-compute "
            "production-faithful loss + pre-clip fp32 global grad norm, SHARED flow noise/time."),
        "weights": _WEIGHTS,
        "total_frames": _WORLD_SIZE * _MICRO,
        "noise_seed": ref["noise_seed"],
        "ref_batch_content_hash": ref["ref_batch_content_hash"],
        "rlinf_batch_content_hash": _content_hash(
            [st["state"] for st in rlinf_micros] + [st["actions"] for st in rlinf_micros]
        ),
        "ref_batch_frame_ids": ref["ref_batch_frame_ids"],
        "rlinf_batch_frame_ids": rl_frame_ids,
        "batches_distinct": ref["ref_batch_content_hash"] != _content_hash(
            [st["state"] for st in rlinf_micros] + [st["actions"] for st in rlinf_micros]),
        "cells": cells,
        # COMPUTE parity per column (same batch, both models):
        "compute_loss_abs_delta": {"ref_batch": loss_d("ref_batch"), "rlinf_batch": loss_d("rlinf_batch")},
        "compute_grad_rel_delta": {"ref_batch": grad_rel("ref_batch"), "rlinf_batch": grad_rel("rlinf_batch")},
        "compute_loss_within_dec1": max(loss_d("ref_batch"), loss_d("rlinf_batch")) <= _DEC1,
        "compute_grad_within_2pct": max(grad_rel("ref_batch"), grad_rel("rlinf_batch")) <= _GRAD_REL,
        # DATA difference per row (same model, two batches):
        "data_loss_abs_delta": {"ref_model": loss_data_d("ref_model"), "rlinf_model": loss_data_d("rlinf_model")},
        "classification": "DATA_SIDE",
    }
    out["classification"] = (
        "DATA_SIDE" if (out["compute_loss_within_dec1"] and out["compute_grad_within_2pct"]
                        and out["batches_distinct"]
                        and min(out["data_loss_abs_delta"].values()) > _DEC1)
        else "INCONCLUSIVE")
    with open(args.out, "w", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("CROSS_FEED_2X2", json.dumps({
        "classification": out["classification"],
        "compute_loss_within_dec1": out["compute_loss_within_dec1"],
        "compute_grad_within_2pct": out["compute_grad_within_2pct"],
        "compute_grad_rel": out["compute_grad_rel_delta"],
        "data_loss_abs_delta": out["data_loss_abs_delta"],
        "cells_loss": {b: {s: round(cells[b][s]["loss"], 5) for s in cells[b]} for b in cells},
        "cells_grad": {b: {s: round(cells[b][s]["grad_norm"], 4) for s in cells[b]} for b in cells},
    }))


if __name__ == "__main__":
    main()
