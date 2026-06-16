# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gradient-norm audit for the vlm_vla SFT loss (tests H1a / H1b / H2).

Loads a bare new-format Pi0 checkpoint and measures, on the same fixed batch
with fixed noise/time, the gradient L2 norm for three loss isolations:
CE-only, flow-only, and combined. It reports global, VLM, action-expert, and
other bucket norms before and after the configured global clip for
``stop_gradient_to_vlm`` both False and True.

By default the batch is synthetic but uses the real PaliGemma tokenizer masks.
For the mandatory fixed-real-batch gate, pass ``--batch-path`` pointing at a
``torch.save`` payload containing either ``(observation, actions)`` or a dict
with ``observation``/``actions`` keys. Run:

    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/vlm_vla_diagnostics/grad_norm_audit.py \
        --ckpt <dir with model.safetensors> \
        --tokenizer /mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import (  # noqa: E402
    Observation,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import (  # noqa: E402
    Pi0Config,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (  # noqa: E402
    PaligemmaTokenizer,
)

SUBTASKS = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]
TASK = "Turn on the radio receiver that's on the table in the living room."


def _set_detach_prefix_kv(model: torch.nn.Module, value: bool) -> int:
    """Mirror the Pi0 init: toggle detach_prefix_kv on every attention layer."""
    n = 0
    for module in model.modules():
        if hasattr(module, "detach_prefix_kv"):
            module.detach_prefix_kv = value
            n += 1
    return n


def _build_batch(tokenizer, cfg, batch_size, device):
    """One synthetic vlm_vla batch: real tokenizer masks, random pixels/actions."""
    state_text = np.linspace(-0.9, 0.9, 8)
    toks, valids, ars, losses, kvs = [], [], [], [], []
    for i in range(batch_size):
        t, v, ar, loss, kv = tokenizer.tokenize_with_subtask(
            TASK, state_text, SUBTASKS[i % len(SUBTASKS)]
        )
        toks.append(t)
        valids.append(v)
        ars.append(ar)
        losses.append(loss)
        kvs.append(kv)

    def _stack(arrs, dtype):
        return torch.as_tensor(np.stack(arrs), dtype=dtype, device=device)

    obs = cfg.fake_obs(batch_size)
    gen = torch.Generator().manual_seed(0)
    observation = Observation(
        images={k: v.to(device) for k, v in obs.images.items()},
        image_masks={k: v.to(device) for k, v in obs.image_masks.items()},
        state=torch.rand(batch_size, cfg.action_dim, generator=gen).to(device) * 2 - 1,
        tokenized_prompt=_stack(toks, torch.long),
        tokenized_prompt_mask=_stack(valids, torch.bool),
        token_ar_mask=_stack(ars, torch.bool),
        token_loss_mask=_stack(losses, torch.bool),
        token_kv_cache_mask=_stack(kvs, torch.bool),
    )
    actions = (
        torch.rand(batch_size, cfg.action_horizon, cfg.action_dim, generator=gen).to(
            device
        )
        * 2
        - 1
    )
    return observation, actions


def _move_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def _move_observation(observation: Observation, device: torch.device) -> Observation:
    """Move a serialized or dataloader observation to the audit device."""
    return Observation(
        images={k: v.to(device) for k, v in observation.images.items()},
        image_masks={k: v.to(device) for k, v in observation.image_masks.items()},
        state=observation.state.to(device),
        tokenized_prompt=_move_tensor(observation.tokenized_prompt, device),
        tokenized_prompt_mask=_move_tensor(observation.tokenized_prompt_mask, device),
        token_ar_mask=_move_tensor(observation.token_ar_mask, device),
        token_loss_mask=_move_tensor(observation.token_loss_mask, device),
        token_kv_cache_mask=_move_tensor(observation.token_kv_cache_mask, device),
        pcd_xyz=_move_tensor(observation.pcd_xyz, device),
    )


def _load_batch(batch_path: pathlib.Path, device: torch.device):
    """Load a fixed batch saved by a dataloader/control script."""
    payload = torch.load(batch_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        observation = payload.get("observation")
        actions = payload.get("actions")
    elif isinstance(payload, (tuple, list)) and len(payload) == 2:
        observation, actions = payload
    else:
        raise ValueError(
            "--batch-path must contain (observation, actions) or "
            "{'observation': ..., 'actions': ...}."
        )
    if isinstance(observation, dict):
        observation = Observation.from_dict(observation)
    if not isinstance(observation, Observation):
        raise TypeError(f"Unsupported observation payload: {type(observation)!r}")
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    return _move_observation(observation, device), actions.to(device)


_ACTION_EXPERT_RE = re.compile(
    r"^llm\.layers\.\d+\.(attn\.(q_proj|k_proj|v_proj|o_proj)\.1|"
    r"pre_attention_norms\.1|pre_ffw_norms\.1|mlps\.1)\."
)
_VLM_EXPERT_RE = re.compile(
    r"^llm\.layers\.\d+\.(attn\.(q_proj|k_proj|v_proj|o_proj)\.0|"
    r"pre_attention_norms\.0|pre_ffw_norms\.0|mlps\.0)\."
)


def _bucket_for_name(name: str) -> str:
    """Classify Pi0 parameters into the buckets used by the audit."""
    action_prefixes = (
        "action_in_proj.",
        "action_out_proj.",
        "state_proj.",
        "time_mlp_in.",
        "time_mlp_out.",
        "action_time_mlp_in.",
        "action_time_mlp_out.",
        "llm.final_norms.1.",
    )
    vlm_prefixes = ("img.", "llm.embedder.", "llm.final_norms.0.")
    if name.startswith(action_prefixes) or _ACTION_EXPERT_RE.match(name):
        return "action_expert"
    if name.startswith(vlm_prefixes) or _VLM_EXPERT_RE.match(name):
        return "vlm"
    return "other"


def _collect_grad_stats(model: torch.nn.Module, clip_grad: float) -> dict:
    """Collect global and bucketed pre/post clip norms without mutating grads."""
    bucket_squares = {"vlm": 0.0, "action_expert": 0.0, "other": 0.0}
    bucket_params = {"vlm": 0, "action_expert": 0, "other": 0}
    total_sq = 0.0
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        sq = float(param.grad.detach().float().pow(2).sum())
        bucket = _bucket_for_name(name)
        bucket_squares[bucket] += sq
        bucket_params[bucket] += param.numel()
        total_sq += sq
    pre_global = total_sq**0.5
    clip_coef = min(1.0, clip_grad / pre_global) if pre_global > 0 else 1.0
    buckets = {}
    for bucket, sq in bucket_squares.items():
        pre = sq**0.5
        buckets[bucket] = {
            "pre": pre,
            "post": pre * clip_coef,
            "num_params_with_grad": bucket_params[bucket],
        }
    return {
        "global_pre": pre_global,
        "global_post": pre_global * clip_coef,
        "clip_coef": clip_coef,
        "buckets": buckets,
    }


def _run_once(model, observation, actions, noise, time, lw, aw, clip_grad):
    model.zero_grad(set_to_none=True)
    model.language_loss_weight = lw
    model.action_loss_weight = aw
    # Seed the augmentation RNG so the three loss isolations are comparable.
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    out = model.compute_loss(
        observation,
        actions,
        train=True,
        noise=noise,
        time=time,
    )
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
    comps = {}
    if isinstance(out, dict):
        for k in ("language_loss", "action_loss", "language_acc"):
            if k in out and out[k] is not None:
                comps[k] = float(out[k])
    return _collect_grad_stats(model, clip_grad), comps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="dir containing model.safetensors")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument(
        "--batch-path",
        type=pathlib.Path,
        default=None,
        help=(
            "Optional torch.save payload with a fixed real batch: either "
            "(observation, actions) or {'observation': ..., 'actions': ...}."
        ),
    )
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument(
        "--preview-lw",
        type=float,
        default=None,
        help=(
            "If set, also run a fix-preview combined step at this "
            "language_loss_weight (action weight 1.0) and report its action-expert "
            "update attenuation — verifies a CE down-weight relieves the coupling "
            "with no training."
        ),
    )
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import safetensors.torch

    device = torch.device(args.device)
    ckpt_dir = pathlib.Path(args.ckpt)
    shape = json.load(open(ckpt_dir / "config.json"))
    cfg = Pi0Config(
        pi05=bool(shape.get("pi05", True)),
        action_horizon=int(shape["action_horizon"]),
        action_dim=int(shape["action_dim"]),
        max_token_len=int(shape.get("max_token_len", 200)),
        paligemma_variant=str(shape["paligemma_variant"]),
        action_expert_variant=str(shape["action_expert_variant"]),
        dtype="float32",
        pcd=bool(shape.get("pcd", False)),
        mode="vlm_vla",
    )
    model = cfg.create()
    state_dict = safetensors.torch.load_file(str(ckpt_dir / "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    model = model.float().to(device)

    tokenizer = PaligemmaTokenizer(args.tokenizer, max_len=cfg.max_token_len)
    if args.batch_path is not None:
        observation, actions = _load_batch(args.batch_path, device)
        batch_source = str(args.batch_path)
    else:
        observation, actions = _build_batch(tokenizer, cfg, args.batch_size, device)
        batch_source = "synthetic-tokenized"

    gen = torch.Generator(device=device).manual_seed(1234)
    noise = torch.randn(actions.shape, device=device, dtype=actions.dtype, generator=gen)
    time = (
        torch.distributions.Beta(torch.tensor(1.5), torch.tensor(1.0))
        .sample((actions.shape[0],))
        .to(device=device, dtype=actions.dtype)
    )
    time = time * 0.999 + 0.001

    print(f"=== vlm_vla gradient-norm audit ({ckpt_dir.name}) ===")
    print(
        f"batch_source={batch_source} batch_size={actions.shape[0]} "
        f"clip_grad={args.clip_grad} device={device}\n"
    )

    def _print_stats(label: str, stats: dict) -> None:
        print(
            f"  {label:<9} global pre={stats['global_pre']:.3f} "
            f"post={stats['global_post']:.3f} coef={stats['clip_coef']:.4f}"
        )
        for bucket in ("vlm", "action_expert", "other"):
            bucket_stats = stats["buckets"][bucket]
            print(
                f"    {bucket:<13} pre={bucket_stats['pre']:.3f} "
                f"post={bucket_stats['post']:.3f} "
                f"params_with_grad={bucket_stats['num_params_with_grad']}"
            )

    for sg in (False, True):
        model.stop_gradient_to_vlm = sg
        n_layers = _set_detach_prefix_kv(model, sg)
        s_ce, _ = _run_once(
            model, observation, actions, noise, time, 1.0, 0.0, args.clip_grad
        )
        s_flow, _ = _run_once(
            model, observation, actions, noise, time, 0.0, 1.0, args.clip_grad
        )
        s_comb, c_comb = _run_once(
            model, observation, actions, noise, time, 1.0, 1.0, args.clip_grad
        )
        # The action-expert update comes only from the flow term; under the single
        # global clip it is scaled by clip_comb instead of clip_flow.
        atten = (
            s_comb["clip_coef"] / s_flow["clip_coef"]
            if s_flow["clip_coef"] > 0
            else float("nan")
        )
        print(f"--- stop_gradient_to_vlm={sg} (detach layers set: {n_layers}) ---")
        print(f"  losses: language={c_comb.get('language_loss'):.4f} "
              f"action={c_comb.get('action_loss'):.4f} "
              f"language_acc={c_comb.get('language_acc', float('nan')):.4f}")
        _print_stats("CE-only", s_ce)
        _print_stats("flow-only", s_flow)
        _print_stats("combined", s_comb)
        print(f"  clip coef @ {args.clip_grad}: combined={s_comb['clip_coef']:.4f} "
              f"flow-only={s_flow['clip_coef']:.4f}")
        print(f"  => action-expert update attenuation vs flow-only step: {atten:.4f} "
              f"(1.0 = no coupling; <1 = CE shrinks the action update)")
        if args.preview_lw is not None:
            s_fix, _ = _run_once(
                model, observation, actions, noise, time,
                args.preview_lw, 1.0, args.clip_grad,
            )
            atten_fix = (
                s_fix["clip_coef"] / s_flow["clip_coef"]
                if s_flow["clip_coef"] > 0
                else float("nan")
            )
            print(f"  FIX-PREVIEW language_loss_weight={args.preview_lw}: "
                  f"combined global={s_fix['global_pre']:.3f} "
                  f"clip_coef={s_fix['clip_coef']:.4f} "
                  f"=> attenuation={atten_fix:.4f}")
        print()


if __name__ == "__main__":
    main()
