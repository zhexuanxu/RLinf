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

"""Runtime mixed-precision dtype ledger for the RLinf BEHAVIOR pi0.5 FSDP SFT path.

Non-invasive: this harness does NOT edit the trainer. It drives the REAL build the
SFT path uses -- the production ``FSDPModelManager.setup_model_and_optimizer`` (the
real ``get_model`` with ``load_for_training=True`` -> fp32 master, the real
``FSDPStrategy.wrap_model`` -> ``MixedPrecision`` policy, the real AdamW) -- and then
reads the EFFECTIVE runtime dtype/flag off the built objects and one forward/backward/
optimizer step. Every recorded value carries a runtime-observed provenance (the object
it was read from), never a YAML string.

Launch under torchrun with world_size >= 2 so a sharded (non-zero) rank is observed::

    torchrun --nproc_per_node=2 tests/unit_tests/_precision_ledger_rlinf.py \
        --config-name behavior_pi05_vla --out-dir <scratch>/precision_ledger

Each rank writes ``rlinf_precision_ledger_rank{r}.json``.
"""

import argparse
import json
import os

import torch


def _dt(x) -> str:
    """Render a torch dtype / tensor dtype as a stable string, or None."""
    if x is None:
        return "None"
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    return str(x)


def _sample_param_dtypes(named_params, limit=4):
    """A small, deterministic sample of (name, dtype) for provenance."""
    out = {}
    for name, p in named_params:
        if p is None or not torch.is_tensor(p):
            continue
        out[name] = _dt(p)
        if len(out) >= limit:
            break
    return out


def _build_cfg(config_name):
    """Compose the SFT config exactly as the SFT entry does (Hydra)."""
    from hydra import compose, initialize_config_dir

    cfg_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "examples/sft/config",
    )
    with initialize_config_dir(version_base="1.1", config_dir=cfg_dir):
        cfg = compose(config_name=config_name)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-name", default="behavior_pi05_vla")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    ledger = {"repo": "rlinf", "rank": rank, "world_size": world_size, "ok": False}
    try:
        from rlinf.config import build_config
        from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
        from rlinf.models import get_model

        cfg = _build_cfg(args.config_name)
        cfg = build_config(cfg) if hasattr(build_config, "__call__") else cfg
        actor_cfg = cfg.actor

        # The production manager build, with the openpi_pytorch model provider the
        # SFT worker uses (get_model -> load_for_training=True -> fp32 master).
        class _LedgerManager(FSDPModelManager):
            def model_provider_func(self):
                return get_model(actor_cfg.model)

        mgr = _LedgerManager(actor_cfg, world_size, rank)
        mgr.setup_model_and_optimizer()
        fsdp_model = mgr.model
        optimizer = mgr.optimizer

        # 1) FSDP MixedPrecision policy object (read off the wrapped module).
        mp = getattr(fsdp_model, "mixed_precision", None)
        ledger["fsdp_mixed_precision"] = {
            "param_dtype": _dt(getattr(mp, "param_dtype", None)),
            "reduce_dtype": _dt(getattr(mp, "reduce_dtype", None)),
            "buffer_dtype": _dt(getattr(mp, "buffer_dtype", None)),
            "cast_forward_inputs": getattr(mp, "cast_forward_inputs", None),
            "cast_root_forward_inputs": getattr(mp, "cast_root_forward_inputs", None),
            "keep_low_precision_grads": getattr(mp, "keep_low_precision_grads", None),
            "provenance": "fsdp_model.mixed_precision",
        }

        # 2) Master/exposed param dtype OUTSIDE forward (use_orig_params exposes the
        #    sharded flat-param storage dtype).
        ledger["param_dtype_outside_forward"] = _sample_param_dtypes(
            fsdp_model.named_parameters()
        )

        # 3) Exposed param dtype DURING forward (a pre-hook captures the cast view).
        during = {}

        def _capture(_module, _inp):
            if not during:
                for n, p in _module.named_parameters(recurse=True):
                    if torch.is_tensor(p) and p.numel() > 0:
                        during[n] = _dt(p)
                        if len(during) >= 4:
                            break

        h = fsdp_model.register_forward_pre_hook(_capture)

        # autocast / grad-scaler state (both expected no-op for this recipe).
        amp_enabled = bool(actor_cfg.fsdp_config.amp_autocast.get("enabled", False))
        gs_enabled = bool(actor_cfg.fsdp_config.grad_scaler.get("enabled", False))
        ledger["autocast_enabled"] = amp_enabled
        ledger["grad_scaler_enabled"] = gs_enabled

        # 4) One forward/backward/optimizer step on a shape-correct batch (dtype-only;
        #    values are irrelevant to the dtype ledger). Built lazily below.
        batch = _shape_correct_sft_batch(actor_cfg, device=f"cuda:{local_rank}")
        from rlinf.models.embodiment.base_policy import ForwardType

        loss = fsdp_model(forward_type=ForwardType.SFT, data=batch)
        ledger["loss_dtype"] = _dt(loss)
        loss.backward()
        h.remove()
        ledger["param_dtype_during_forward"] = during

        # 5) Gradient dtype after backward.
        ledger["grad_dtype"] = {
            n: _dt(p.grad)
            for n, p in list(fsdp_model.named_parameters())[:4]
            if p.grad is not None
        }

        optimizer.step()

        # 6) Optimizer (AdamW) construction + state dtype after step 1.
        defaults = optimizer.defaults
        ledger["optimizer"] = {
            "type": type(optimizer).__name__,
            "fused": defaults.get("fused"),
            "foreach": defaults.get("foreach"),
            "capturable": defaults.get("capturable"),
            "betas": list(defaults.get("betas", ())),
            "eps": defaults.get("eps"),
            "weight_decay": defaults.get("weight_decay"),
        }
        state_dtypes = {}
        for p, st in list(optimizer.state.items())[:2]:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st:
                    state_dtypes[k] = _dt(st[k])
        ledger["optimizer_state_dtype"] = state_dtypes

        ledger["ok"] = True
    except Exception as e:  # pragma: no cover - environment dependent
        import traceback

        ledger["err"] = f"{type(e).__name__}: {str(e)[:400]}"
        ledger["trace"] = traceback.format_exc()[-1800:]

    with open(
        os.path.join(args.out_dir, f"rlinf_precision_ledger_rank{rank}.json"), "w"
    ) as f:
        json.dump(ledger, f, indent=2)
    print(
        f"RLINF_LEDGER rank={rank} "
        + json.dumps({k: ledger.get(k) for k in ("ok", "err")})
    )


def _shape_correct_sft_batch(actor_cfg, device):
    """Build a shape-correct SFT batch (dtype-only ledger; values irrelevant)."""
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

    b = 1
    horizon = int(actor_cfg.model.num_action_chunks)
    model_dim = int(actor_cfg.model.openpi.model_action_dim)
    img_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    obs = Observation(
        images={k: torch.rand(b, 224, 224, 3, device=device) for k in img_keys},
        image_masks={
            k: torch.ones(b, dtype=torch.bool, device=device) for k in img_keys
        },
        state=torch.rand(b, model_dim, device=device),
        tokenized_prompt=torch.ones(b, 200, dtype=torch.long, device=device),
        tokenized_prompt_mask=torch.ones(b, 200, dtype=torch.bool, device=device),
        token_ar_mask=None,
        token_loss_mask=None,
        pcd_xyz=None,
    )
    actions = torch.rand(b, horizon, model_dim, device=device)
    return {"observation": obs, "actions": actions}


if __name__ == "__main__":
    main()
