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

"""Runtime mixed-precision dtype ledger for the REFERENCE pi0.5 FSDP1 training.

Non-invasive: imports and CALLS the reference trainer's own ``init_model`` (which
builds the model, applies ``MixedPrecision(param_dtype=bf16, reduce_dtype=fp32)``
with the constructor default ``buffer_dtype``, casts the master to fp32, FSDP1-wraps,
and builds the fused AdamW) without editing the reference source, then reads the
EFFECTIVE runtime dtype/flag off the built objects and one forward/backward/optimizer
step. The reference's omitted ``buffer_dtype`` default is OBSERVED, not assumed.

Launch under torchrun in the reference venv (world_size >= 2 for a sharded rank)::

    PYTHONPATH=<ref-src>:<ref-scripts> <ref-venv>/torchrun --nproc_per_node=2 \
        tests/unit_tests/_precision_ledger_ref.py --out-dir <scratch>/precision_ledger

Each rank writes ``ref_precision_ledger_rank{r}.json``.
"""

import argparse
import json
import os

import torch

_REF_CONFIG = "pi05_b1k-task0000_sft_pytorch_mixed"
_DIST_METHOD = "fsdp1"
_USE_AUTOCAST = False  # reference run.sh: USE_AUTOCAST=0


def _dt(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    return str(x)


def _sample(named, limit=4, want_grad=False):
    out = {}
    for name, p in named:
        if not torch.is_tensor(p):
            continue
        if want_grad:
            if p.grad is None:
                continue
            out[name] = _dt(p.grad)
        else:
            out[name] = _dt(p)
        if len(out) >= limit:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    ledger = {"repo": "reference", "rank": rank, "world_size": world_size, "ok": False}
    try:
        import openpi.training.config as _config
        from train_pytorch_new import init_model

        config = _config.get_config(_REF_CONFIG)
        ledger["pytorch_training_precision"] = str(config.pytorch_training_precision)

        model, optim = init_model(
            config,
            dist_method=_DIST_METHOD,
            device=device,
            world_size=world_size,
            is_main=(rank == 0),
            compile_mode="default",
            use_autocast=_USE_AUTOCAST,
        )

        mp = getattr(model, "mixed_precision", None)
        ledger["fsdp_mixed_precision"] = {
            "param_dtype": _dt(getattr(mp, "param_dtype", None)),
            "reduce_dtype": _dt(getattr(mp, "reduce_dtype", None)),
            "buffer_dtype": _dt(getattr(mp, "buffer_dtype", None)),
            "cast_forward_inputs": getattr(mp, "cast_forward_inputs", None),
            "cast_root_forward_inputs": getattr(mp, "cast_root_forward_inputs", None),
            "keep_low_precision_grads": getattr(mp, "keep_low_precision_grads", None),
            "provenance": "init_model(...).mixed_precision",
        }
        # The omitted-buffer_dtype effective default, observed (not assumed): sample a
        # real buffer's dtype off the wrapped model.
        buf = {}
        for n, b in model.named_buffers():
            if torch.is_tensor(b) and b.numel() > 0:
                buf[n] = _dt(b)
                if len(buf) >= 4:
                    break
        ledger["buffer_dtype_observed"] = buf

        ledger["param_dtype_outside_forward"] = _sample(model.named_parameters())
        ledger["autocast_enabled"] = _USE_AUTOCAST
        ledger["grad_scaler_enabled"] = (
            False  # reference uses no GradScaler (fp32 reduce)
        )

        # One forward/backward/step on a shape-correct batch (dtype-only ledger).
        obs, actions, noise, time = _shape_correct_ref_batch(config, device)
        out = model(obs, actions, train=True, noise=noise, time=time)
        loss = out.mean() if torch.is_tensor(out) else out
        ledger["loss_dtype"] = _dt(loss)
        loss.backward()
        ledger["grad_dtype"] = _sample(model.named_parameters(), want_grad=True)
        optim.step()

        # Read the EFFECTIVE per-param-group hyperparameters (consistent with the
        # RLinf ledger), not optimizer.defaults which can be a misleading proxy.
        defaults = optim.defaults
        pg = optim.param_groups[0]
        ledger["optimizer"] = {
            "type": type(optim).__name__,
            "fused": defaults.get("fused"),
            "foreach": defaults.get("foreach"),
            "capturable": defaults.get("capturable"),
            "betas": list(pg.get("betas", defaults.get("betas", ()))),
            "eps": pg.get("eps", defaults.get("eps")),
            "weight_decay": pg.get("weight_decay", defaults.get("weight_decay")),
            "lr": pg.get("lr"),
        }
        st_dtypes = {}
        for _p, st in list(optim.state.items())[:2]:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    st_dtypes[k] = _dt(st[k])
        ledger["optimizer_state_dtype"] = st_dtypes
        ledger["ok"] = True
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        ledger["err"] = f"{type(e).__name__}: {str(e)[:400]}"
        ledger["trace"] = traceback.format_exc()[-1800:]

    with open(
        os.path.join(args.out_dir, f"ref_precision_ledger_rank{rank}.json"), "w"
    ) as f:
        json.dump(ledger, f, indent=2)
    print(
        "REF_LEDGER rank=%d %s"
        % (rank, json.dumps({k: ledger.get(k) for k in ("ok", "err")}))
    )


def _shape_correct_ref_batch(config, device):
    """Build a shape-correct reference (Observation, actions, noise, time)."""
    import numpy as np
    from openpi.models_pytorch_new import model as omodel

    horizon = int(config.model.action_horizon)
    adim = int(config.model.action_dim)
    img_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    b = 1
    obs = omodel.Observation.from_dict(
        {
            "image": {k: torch.rand(b, 224, 224, 3, device=device) for k in img_keys},
            "image_mask": {
                k: torch.ones(b, dtype=torch.bool, device=device) for k in img_keys
            },
            "state": torch.rand(b, adim, device=device),
            "tokenized_prompt": torch.ones(b, 200, dtype=torch.long, device=device),
            "tokenized_prompt_mask": torch.ones(
                b, 200, dtype=torch.bool, device=device
            ),
        }
    )
    actions = torch.rand(b, horizon, adim, device=device)
    noise = torch.from_numpy(
        np.random.RandomState(0).randn(b, horizon, adim).astype("float32")
    ).to(device)
    time = torch.full((b,), 0.5, device=device)
    return obs, actions, noise, time


if __name__ == "__main__":
    main()
