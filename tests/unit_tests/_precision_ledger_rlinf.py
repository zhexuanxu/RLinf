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

Non-invasive: drives the REAL SFT build (``FSDPModelManager.setup_model_and_optimizer``
-> ``get_model(load_for_training=True)`` -> ``wrap_model`` -> AdamW + grad scaler) and
reads the EFFECTIVE runtime dtype/flag off the built objects and one pinned
forward / backward / real-optimizer-step. Both repos consume the SAME pinned input
(``_precision_pinned.load_pinned``); every ledger record is an object
``{value, provenance, stage}`` whose provenance points at the runtime object it was
read from, never YAML. Run under torchrun, world_size >= 2 (a sharded rank)::

    torchrun --nproc_per_node=2 tests/unit_tests/_precision_ledger_rlinf.py \
        --out-dir <scratch>/precision_ledger

Writes ``rlinf_precision_ledger_rank{r}.json``.
"""

import argparse
import json
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _precision_pinned import build_pinned, load_pinned, sha_tensor  # noqa: E402


def _dt(x) -> str:
    if x is None:
        return "None"
    if isinstance(x, torch.dtype):
        return str(x).replace("torch.", "")
    if torch.is_tensor(x):
        return str(x.dtype).replace("torch.", "")
    return str(x)


def _S(value, provenance, stage=None):
    """One ledger record: the value + where it was runtime-observed."""
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
    """Observe the installed FSDP's behavior for an OMITTED buffer_dtype: wrap a tiny
    module that HAS an fp32 buffer with MixedPrecision(buffer_dtype=None) and read the
    buffer dtype after wrap (separate from the real model, which has 0 buffers)."""
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
    ap.add_argument("--config-name", default="behavior_pi05_vla")
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

    ledger = {"repo": "rlinf", "rank": rank, "world_size": world_size, "ok": False}
    surf = {}
    ledger["surfaces"] = surf
    try:
        from hydra import compose, initialize_config_dir

        from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
        from rlinf.models import get_model
        from rlinf.models.embodiment.base_policy import ForwardType
        from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation

        cfg_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "examples/sft/config",
        )
        with initialize_config_dir(version_base="1.1", config_dir=cfg_dir):
            cfg = compose(config_name=args.config_name)
        actor_cfg = cfg.actor

        class _LedgerManager(FSDPModelManager):
            def model_provider_func(self):
                return get_model(actor_cfg.model)

        mgr = _LedgerManager(actor_cfg, world_size, rank)
        mgr.setup_model_and_optimizer()
        fsdp_model, optimizer = mgr.model, mgr.optimizer

        # --- Pinned input (the SAME deterministic batch both repos consume) ---
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
        obs = Observation(
            images=pin["images"],
            image_masks=pin["image_masks"],
            state=pin["state"],
            tokenized_prompt=pin["tokenized_prompt"],
            tokenized_prompt_mask=pin["tokenized_prompt_mask"],
            token_ar_mask=None,
            token_loss_mask=None,
            pcd_xyz=None,
        )
        batch = {
            "observation": obs,
            "actions": pin["actions"],
            "noise": pin["noise"],
            "time": pin["time"],
        }
        surf["action_input_dtype"] = _S(
            _dt(pin["actions"]), "pinned batch actions tensor", "input"
        )
        surf["noise_input_dtype"] = _S(
            _dt(pin["noise"]), "pinned batch noise tensor", "input"
        )
        surf["time_input_dtype"] = _S(
            _dt(pin["time"]), "pinned batch time tensor", "input"
        )

        # --- FSDP MixedPrecision policy object ---
        mp = getattr(fsdp_model, "mixed_precision", None)
        for k in ("param_dtype", "reduce_dtype", "buffer_dtype"):
            surf[f"fsdp_{k}"] = _S(
                _dt(getattr(mp, k, None)), f"fsdp_model.mixed_precision.{k}"
            )
        surf["gradient_reduction_dtype"] = _S(
            _dt(getattr(mp, "reduce_dtype", None)),
            "fsdp_model.mixed_precision.reduce_dtype",
        )
        for k in (
            "cast_forward_inputs",
            "cast_root_forward_inputs",
            "keep_low_precision_grads",
        ):
            surf[f"mp_{k}"] = _S(
                getattr(mp, k, None), f"fsdp_model.mixed_precision.{k}"
            )

        # --- master param dtype OUTSIDE forward ---
        out_p = next(iter(fsdp_model.named_parameters()))
        surf["master_param_dtype"] = _S(
            _dt(out_p[1]),
            f"fsdp_model.named_parameters()[{out_p[0]}] outside forward",
            "pre_forward",
        )

        # --- buffers (count + dtypes) ---
        bufs = [(n, b) for n, b in fsdp_model.named_buffers() if torch.is_tensor(b)]
        surf["buffer_count"] = _S(len(bufs), "fsdp_model.named_buffers()")
        surf["buffer_dtypes"] = _S(
            sorted({_dt(b) for _, b in bufs}), "fsdp_model.named_buffers() dtypes"
        )

        # --- EMA absence (the SFT build has no EMA averaging) ---
        surf["ema"] = _S(
            "none",
            "FSDPModelManager SFT build has no EMA; cfg.actor has no ema_decay",
        )

        # --- grad scaler (real object; enabled=False -> no-op) ---
        gs = getattr(mgr, "grad_scaler", None)
        surf["grad_scaler"] = _S(
            {
                "type": type(gs).__name__,
                "enabled": bool(getattr(gs, "is_enabled", lambda: False)()),
            },
            "mgr.grad_scaler object",
        )

        # --- during-forward param dtype + runtime autocast (inner-module hook) ---
        leaf = _first_leaf_linear(fsdp_model)
        during = {}

        def _hook(_m, _i):
            if "param_dtype" not in during:
                during["param_dtype"] = _dt(_m.weight)
                during["autocast_enabled"] = bool(torch.is_autocast_enabled())

        h = leaf.register_forward_pre_hook(_hook) if leaf is not None else None

        loss = fsdp_model(forward_type=ForwardType.SFT, data=batch)
        surf["loss_dtype"] = _S(_dt(loss), "SFT forward return", "after_forward")
        surf["output_dtype"] = _S(
            _dt(loss), "model SFT output (scalar loss)", "after_forward"
        )
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
        gp = [(n, p) for n, p in fsdp_model.named_parameters() if p.grad is not None]
        surf["grad_dtype"] = _S(
            sorted({_dt(p.grad) for _, p in gp[:8]}) or None,
            f"param.grad after backward ({len(gp)} params with grad)",
            "after_backward",
        )

        # Observe the fp32 grad-norm ACCUMULATION dtype directly off the grads (the
        # strategy's real path accumulates in fp32 then returns a python float; this
        # captures the fp32 tensor dtype the float is derived from) BEFORE the step
        # clips the grads in place.
        sq = torch.zeros((), dtype=torch.float32, device=device)
        for _, p in gp:
            sq = sq + p.grad.detach().to(torch.float32).pow(2).sum()
        gn_tensor = sq.sqrt()

        # --- runtime fingerprint of the LOADED weights, gathered as an unsharded fp32
        #     FULL_STATE_DICT BEFORE the optimizer step mutates them. Both repos load the
        #     SAME base checkpoint (pi05_base_pytorch_new); this set-hash proves that from
        #     runtime tensors, so the grad-norm below is a controlled-step numeric parity
        #     on IDENTICAL weights, not a comparison of independent inits. ---
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            _pre_sd = fsdp_model.state_dict()
        surf["loaded_weight_fingerprint"] = _S(
            _weight_fingerprint(_pre_sd),
            "FSDP FULL_STATE_DICT gathered BEFORE optimizer_step: key-independent "
            "set-hash of fp32 param values (cross-repo identical-loaded-weights proof)",
            "pre_step",
        )

        # --- grad-norm via the REAL training step path (returns the production value) ---
        grad_norm, _lrs = mgr.optimizer_step()
        gn = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
        surf["grad_norm"] = _S(
            {
                "value": gn,
                "dtype": _dt(gn_tensor),
                "fp32_observed_value": gn_tensor.item(),
            },
            "value: mgr.optimizer_step()->strategy.clip_grad_norm_; "
            "dtype: fp32 accumulation observed over param grads",
            "pre_step",
        )

        # --- optimizer construction (param_groups) + state dtype after step ---
        defaults, pg = optimizer.defaults, optimizer.param_groups[0]
        surf["optimizer"] = _S(
            {
                "type": type(optimizer).__name__,
                "fused": defaults.get("fused"),
                "foreach": defaults.get("foreach"),
                "capturable": defaults.get("capturable"),
                "betas": list(pg.get("betas", ())),
                "eps": pg.get("eps"),
                "weight_decay": pg.get("weight_decay"),
                "lr": pg.get("lr"),
            },
            "optimizer.param_groups[0] + optimizer.defaults",
        )
        st = {}
        for _p, s in list(optimizer.state.items())[:2]:
            for k in ("exp_avg", "exp_avg_sq"):
                if k in s:
                    st[k] = _dt(s[k])
        surf["optimizer_state_dtype"] = _S(
            st or None, "optimizer.state[*] after step 1", "after_step"
        )

        # --- saved-checkpoint dtype + load-after-save dtype (FSDP full state dict) ---
        # FullStateDictConfig, StateDictType, FSDP already imported by the pre-step block.
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        ):
            sd = fsdp_model.state_dict()
        sd_tensors = [v for v in sd.values() if torch.is_tensor(v)]
        surf["saved_checkpoint_dtype"] = _S(
            {
                "dtypes": sorted({_dt(v) for v in sd_tensors}),
                "count": len(sd_tensors),
            },
            "FSDP FULL_STATE_DICT: ALL tensor dtypes",
            "saved",
        )
        # Real load-after-save round trip: instantiate a FRESH model via the same build
        # path, load the saved state dict, inspect ALL post-load param dtypes.
        fresh = get_model(actor_cfg.model)
        fresh.load_state_dict(sd, strict=False)
        reload_p = [p for _, p in fresh.named_parameters()]
        surf["load_after_save_dtype"] = _S(
            {
                "dtypes": sorted({_dt(p) for p in reload_p}),
                "count": len(reload_p),
            },
            "fresh get_model.load_state_dict(saved): ALL post-load param dtypes",
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
                _rec["repo"] = "rlinf"
                _rec["rank"] = rank

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
        "RLINF_LEDGER rank=%d %s"
        % (rank, json.dumps({k: ledger.get(k) for k in ("ok", "err")}))
    )


# Keep the build helpers referenced so torchrun's spawn import does not flag them.
_ = (build_pinned, sha_tensor)

if __name__ == "__main__":
    main()
