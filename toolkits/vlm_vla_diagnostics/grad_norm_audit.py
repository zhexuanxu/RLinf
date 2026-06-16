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

Loads a bare new-format Pi0 checkpoint, builds one synthetic vlm_vla batch
(real PaliGemma tokenizer + the three token masks; random images/state/actions),
and measures, on the SAME batch with fixed noise/time, the global pre-clip
gradient L2 norm for three loss isolations — CE-only, flow-only, combined — and
for ``stop_gradient_to_vlm`` both False and True. It then reports the clip
coefficient each combined norm would get under ``clip_grad=1.0`` and the
resulting attenuation of the action-expert (flow) update relative to a
flow-only step.

Decisive read (no bucketing needed): if CE-only norm >> flow-only norm, then the
single global clip is dominated by CE and shrinks the action-expert gradient by
~flow/combined — the coupling H1a predicts. Run:

    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/vlm_vla_diagnostics/grad_norm_audit.py \
        --ckpt <dir with model.safetensors> \
        --tokenizer /mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np
import torch

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import PaligemmaTokenizer

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


def _grad_norm(model: torch.nn.Module) -> float:
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().float().pow(2).sum())
    return sq**0.5


def _run_once(model, observation, actions, lw, aw, device):
    model.zero_grad(set_to_none=True)
    model.language_loss_weight = lw
    model.action_loss_weight = aw
    # Seed the global RNG (not a passed generator) so the model's CPU image-crop
    # randint and its CUDA flow noise both reproduce identically across the three
    # loss isolations — making CE/flow/combined norms directly comparable.
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    out = model.compute_loss(observation, actions, train=True)
    loss = out["loss"] if isinstance(out, dict) else out
    loss.backward()
    comps = {}
    if isinstance(out, dict):
        for k in ("language_loss", "action_loss", "language_acc"):
            if k in out and out[k] is not None:
                comps[k] = float(out[k])
    return _grad_norm(model), comps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="dir containing model.safetensors")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--clip-grad", type=float, default=1.0)
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
    observation, actions = _build_batch(tokenizer, cfg, args.batch_size, device)

    print(f"=== vlm_vla gradient-norm audit ({ckpt_dir.name}) ===")
    print(f"batch_size={args.batch_size} clip_grad={args.clip_grad} device={device}\n")

    for sg in (False, True):
        model.stop_gradient_to_vlm = sg
        n_layers = _set_detach_prefix_kv(model, sg)
        n_ce, c_ce = _run_once(model, observation, actions, 1.0, 0.0, device)
        n_flow, c_flow = _run_once(model, observation, actions, 0.0, 1.0, device)
        n_comb, c_comb = _run_once(model, observation, actions, 1.0, 1.0, device)
        clip_comb = min(1.0, args.clip_grad / n_comb) if n_comb > 0 else 1.0
        clip_flow = min(1.0, args.clip_grad / n_flow) if n_flow > 0 else 1.0
        # The action-expert update comes only from the flow term; under the single
        # global clip it is scaled by clip_comb instead of clip_flow.
        atten = clip_comb / clip_flow if clip_flow > 0 else float("nan")
        print(f"--- stop_gradient_to_vlm={sg} (detach layers set: {n_layers}) ---")
        print(f"  losses: language={c_comb.get('language_loss'):.4f} "
              f"action={c_comb.get('action_loss'):.4f} "
              f"language_acc={c_comb.get('language_acc', float('nan')):.4f}")
        print(f"  pre-clip grad norm  CE-only   : {n_ce:.3f}")
        print(f"  pre-clip grad norm  flow-only : {n_flow:.3f}")
        print(f"  pre-clip grad norm  combined  : {n_comb:.3f}")
        print(f"  clip coef @ {args.clip_grad}: combined={clip_comb:.4f} "
              f"flow-only={clip_flow:.4f}")
        print(f"  => action-expert update attenuation vs flow-only step: {atten:.4f} "
              f"(1.0 = no coupling; <1 = CE shrinks the action update)\n")


if __name__ == "__main__":
    main()
