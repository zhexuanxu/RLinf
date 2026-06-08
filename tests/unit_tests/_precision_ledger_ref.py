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

Non-invasive: imports + CALLS the reference trainer's own ``init_model`` (model +
``MixedPrecision(param_dtype=bf16, reduce_dtype=fp32)`` with default ``buffer_dtype``
+ fp32 master + FSDP1 wrap + fused AdamW), feeds the SAME shared pinned batch as the
RLinf ledger, and reads the EFFECTIVE runtime dtype/flag off the built objects + one
forward / backward / real-grad-norm / optimizer step. Same per-surface
``{value, provenance, stage}`` schema as the RLinf ledger.

Launch under torchrun in the reference venv (world_size >= 2)::

    PYTHONPATH=<ref-src>:<ref-scripts>:<rlinf-repo> <ref-venv>/torchrun \
        --nproc_per_node=2 tests/unit_tests/_precision_ledger_ref.py --out-dir <dir>

Writes ``ref_precision_ledger_rank{r}.json``.
"""

import argparse
import datetime
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _precision_pinned import load_pinned, sha_tensor  # noqa: E402

_REF_CONFIG = "pi05_b1k-task0000_sft_pytorch_mixed"
_DIST_METHOD = "fsdp1"
_USE_AUTOCAST = False


def _dt(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    return str(x)


def _S(value, provenance, stage=None):
    rec = {"value": value, "provenance": provenance}
    if stage is not None:
        rec["stage"] = stage
    return rec


def _weight_fingerprint(state_dict):
    """Key-independent runtime fingerprint of the loaded weights.

    Hashes every floating-point tensor's fp32 value bytes and folds the SET of
    per-tensor hashes into one digest. Two models that hold the SAME weights hash
    identically regardless of key names or tied-weight duplication, so this proves
    cross-repo identical-loaded-weights from runtime tensors (not config text).
    """
    import hashlib

    per = {
        sha_tensor(v.detach().to(torch.float32))
        for v in state_dict.values()
        if torch.is_tensor(v) and v.is_floating_point()
    }
    digest = hashlib.sha256("".join(sorted(per)).encode()).hexdigest()
    return {"set_hash": digest, "n_distinct_float_tensors": len(per)}


def _first_leaf_linear(module):
    for m in module.modules():
        if isinstance(m, torch.nn.Linear) and m.weight is not None:
            return m
    return None


def _buffer_default_probe(device):
    """Observe the installed FSDP's behavior for an OMITTED buffer_dtype (the reference
    omits it): wrap a tiny module with an fp32 buffer using MixedPrecision(buffer_dtype=
    None) and read the buffer dtype after wrap."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import MixedPrecision

    m = torch.nn.Linear(8, 8).to(device)
    m.register_buffer("probe_buf", torch.ones(8, dtype=torch.float32, device=device))
    wrapped = FSDP(
        m,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=None
        ),
        device_id=device.index,
    )
    after = {n: _dt(b) for n, b in wrapped.named_buffers() if torch.is_tensor(b)}
    return {
        "omitted_buffer_dtype": "None",
        "buffer_dtype_after_wrap": after,
        "note": "with buffer_dtype omitted, FSDP leaves the buffer at its original dtype",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")

    ledger = {"repo": "reference", "rank": rank, "world_size": world_size, "ok": False}
    surf = {}
    ledger["surfaces"] = surf
    try:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc
        import openpi.training.config as _config
        from openpi.models_pytorch_new import model as omodel
        from train_pytorch_new import init_model

        config = _config.get_config(_REF_CONFIG)
        surf["pytorch_training_precision"] = _S(
            str(config.pytorch_training_precision), "config.pytorch_training_precision"
        )

        model, optim = init_model(
            config,
            dist_method=_DIST_METHOD,
            device=device,
            world_size=world_size,
            is_main=(rank == 0),
            compile_mode="default",
            use_autocast=_USE_AUTOCAST,
        )

        # --- shared pinned input ---
        pin = load_pinned(device)
        ledger["pinned_input"] = {
            "spec_path": pin["spec_path"],
            "artifact": pin["artifact"],
            "field_sha256": pin["hashes"],
            "shapes": pin["shapes"],
            "provenance": (
                "_precision_pinned.load_pinned (committed ref_pinned_npz artifact; "
                "derived from _ref_pinned_run.py pinned SFT input/noise-time path)"
            ),
        }
        obs = omodel.Observation.from_dict(
            {
                "image": pin["images"],
                "image_mask": pin["image_masks"],
                "state": pin["state"],
                "tokenized_prompt": pin["tokenized_prompt"],
                "tokenized_prompt_mask": pin["tokenized_prompt_mask"],
            }
        )
        actions, noise, time = pin["actions"], pin["noise"], pin["time"]
        surf["action_input_dtype"] = _S(_dt(actions), "pinned batch actions", "input")
        surf["noise_input_dtype"] = _S(_dt(noise), "pinned batch noise", "input")
        surf["time_input_dtype"] = _S(_dt(time), "pinned batch time", "input")

        mp = getattr(model, "mixed_precision", None)
        for k in ("param_dtype", "reduce_dtype", "buffer_dtype"):
            surf[f"fsdp_{k}"] = _S(
                _dt(getattr(mp, k, None)), f"init_model(...).mixed_precision.{k}"
            )
        surf["gradient_reduction_dtype"] = _S(
            _dt(getattr(mp, "reduce_dtype", None)),
            "init_model(...).mixed_precision.reduce_dtype",
        )
        for k in (
            "cast_forward_inputs",
            "cast_root_forward_inputs",
            "keep_low_precision_grads",
        ):
            surf[f"mp_{k}"] = _S(getattr(mp, k, None), f"mixed_precision.{k}")

        out_p = next(iter(model.named_parameters()))
        surf["master_param_dtype"] = _S(
            _dt(out_p[1]),
            f"model.named_parameters()[{out_p[0]}] outside forward",
            "pre_forward",
        )

        bufs = [(n, b) for n, b in model.named_buffers() if torch.is_tensor(b)]
        surf["buffer_count"] = _S(len(bufs), "model.named_buffers()")
        surf["buffer_dtypes"] = _S(
            sorted({_dt(b) for _, b in bufs}), "model.named_buffers() dtypes"
        )

        surf["ema"] = _S(
            "none"
            if getattr(config, "ema_decay", None) is None
            else str(config.ema_decay),
            "config.ema_decay",
        )
        surf["grad_scaler"] = _S(
            {"type": "None", "enabled": False},
            "reference training uses no GradScaler (fp32 reduce)",
        )

        leaf = _first_leaf_linear(model)
        during = {}

        def _hook(_m, _i):
            if "param_dtype" not in during:
                during["param_dtype"] = _dt(_m.weight)
                during["autocast_enabled"] = bool(torch.is_autocast_enabled())

        h = leaf.register_forward_pre_hook(_hook) if leaf is not None else None

        out = model(obs, actions, train=True, noise=noise, time=time)
        loss = out.mean() if torch.is_tensor(out) else out
        surf["loss_dtype"] = _S(_dt(loss), "model(...) return mean", "after_forward")
        surf["output_dtype"] = _S(_dt(out), "model(...) raw return", "after_forward")
        if h is not None:
            h.remove()
        surf["param_dtype_during_forward"] = _S(
            during.get("param_dtype"),
            "inner Linear forward pre-hook (compute view)",
            "forward",
        )
        surf["compute_autocast_enabled"] = _S(
            during.get("autocast_enabled"),
            "torch.is_autocast_enabled() inside forward",
            "forward",
        )

        loss.backward()
        gp = [(n, p) for n, p in model.named_parameters() if p.grad is not None]
        surf["grad_dtype"] = _S(
            sorted({_dt(p.grad) for _, p in gp[:8]}) or None,
            f"param.grad after backward ({len(gp)} params with grad)",
            "after_backward",
        )

        max_norm = float(getattr(config.optimizer, "clip_gradient_norm", 1.0))
        grad_norm = model.clip_grad_norm_(max_norm=max_norm)
        gn = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        surf["grad_norm"] = _S(
            {
                "value": gn,
                "dtype": _dt(grad_norm) if torch.is_tensor(grad_norm) else "float",
            },
            "model.clip_grad_norm_ (FSDP1 real path)",
            "pre_step",
        )

        # --- runtime fingerprint of the LOADED weights, gathered as an unsharded fp32
        #     FULL_STATE_DICT BEFORE optim.step() mutates them. Both repos load the SAME
        #     base checkpoint (pi05_base_pytorch_new); this set-hash proves that from
        #     runtime tensors, so the grad-norm above is a controlled-step numeric parity
        #     on IDENTICAL weights, not a comparison of independent inits. ---
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            _pre_sd = model.state_dict()
        surf["loaded_weight_fingerprint"] = _S(
            _weight_fingerprint(_pre_sd),
            "FSDP FULL_STATE_DICT gathered BEFORE optim.step(): key-independent set-hash "
            "of fp32 param values (cross-repo identical-loaded-weights proof)",
            "pre_step",
        )
        optim.step()

        defaults, pg = optim.defaults, optim.param_groups[0]
        surf["optimizer"] = _S(
            {
                "type": type(optim).__name__,
                "fused": defaults.get("fused"),
                "foreach": defaults.get("foreach"),
                "capturable": defaults.get("capturable"),
                "betas": list(pg.get("betas", ())),
                "eps": pg.get("eps"),
                "weight_decay": pg.get("weight_decay"),
                "lr": pg.get("lr"),
            },
            "optim.param_groups[0] + optim.defaults",
        )
        st = {}
        for _p, s in list(optim.state.items())[:2]:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in s:
                    st[k] = _dt(s[k])
        surf["optimizer_state_dtype"] = _S(
            st or None, "optim.state[*] after step 1", "after_step"
        )

        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            sd = model.state_dict()
            sd_tensors = [v for v in sd.values() if torch.is_tensor(v)]
            surf["saved_checkpoint_dtype"] = _S(
                {
                    "dtypes": sorted({_dt(v) for v in sd_tensors}),
                    "count": len(sd_tensors),
                },
                "FSDP FULL_STATE_DICT: ALL tensor dtypes",
                "saved",
            )
            # Real load-after-save: load the saved dict BACK into the model (under the
            # full-state-dict context) and inspect ALL post-load param dtypes.
            model.load_state_dict(sd, strict=False)
        post = [p for _, p in model.named_parameters()]
        surf["load_after_save_dtype"] = _S(
            {"dtypes": sorted({_dt(p) for p in post}), "count": len(post)},
            "model.load_state_dict(saved): ALL post-load param dtypes",
            "reloaded",
        )

        # --- observed omitted-buffer_dtype FSDP default (a tiny labeled probe) ---
        try:
            _probe = _buffer_default_probe(device)
        except Exception as _pe:  # pragma: no cover - environment dependent
            _probe = {"error": f"{type(_pe).__name__}: {str(_pe)[:200]}"}
        surf["buffer_default_probe"] = _S(
            _probe, "FSDP MixedPrecision(buffer_dtype=None) probe"
        )

        # --- per-record identity (surface name + repo + rank on every record) ---
        for _key, _rec in surf.items():
            if isinstance(_rec, dict):
                _rec.setdefault("surface", _key)
                _rec["repo"] = "reference"
                _rec["rank"] = rank

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


if __name__ == "__main__":
    main()
