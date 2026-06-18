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

"""Real-model eval-path parity: does the vlm_vla eval denoise match the joint
training forward for the SAME subtask + SAME noise?

The eval path is generate_language -> _denoise_actions (suffix attends a
StaticKVCache). The training/loss path is one joint forward (prefix+response+
suffix together). If the two produce DIFFERENT action velocities for identical
tokens, the action expert "trains fine but evals ~0" because eval feeds it a
different context than training — a real-model bug the tiny-model unit tests miss.

Procedure (one batch, fixed noise):
1. eval path: reason_and_sample_actions(obs_gen) -> A_eval + generated tokens.
2. joint path: rebuild the Observation with those generated subtask tokens as the
   response (training layout) and run an Euler denoise using the EXACT joint
   forward from Pi0.compute_loss, same noise/steps -> A_joint.
3. report max|A_eval - A_joint|. ~0 => eval path is faithful (action expert is
   trained-bad); large => eval-path divergence is the root cause.

Run:
    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/vlm_vla_diagnostics/eval_parity_check.py \
        --ckpt <dir with model.safetensors> \
        --tokenizer /mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as model_mod  # noqa: E402
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation  # noqa: E402
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import (  # noqa: E402
    block_suffix_from_excluded_prefix,
    make_attn_mask,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config  # noqa: E402
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (  # noqa: E402
    PaligemmaTokenizer,
)

TASK = "Turn on the radio receiver that's on the table in the living room."
SUBTASKS = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]


def _joint_velocity(model, observation, x_t, time):
    """Action-expert velocity from the JOINT forward (mirrors compute_loss)."""
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )
    B = x_t.shape[0]
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    if prefix_ar_mask.dim() == 2:
        suffix_ar_mask = suffix_ar_mask.unsqueeze(0).expand(B, -1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=1)
    else:
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    prefix_len = prefix_mask.shape[1]
    if model.vlm_vla:
        attn_mask = block_suffix_from_excluded_prefix(
            attn_mask, observation.token_kv_cache_mask, prefix_len=prefix_len
        )
        text_len = observation.token_kv_cache_mask.shape[1]
        position_mask = input_mask.clone()
        position_mask[:, prefix_len - text_len : prefix_len] &= (
            observation.token_kv_cache_mask
        )
        positions = torch.cumsum(position_mask.int(), dim=1) - 1
    else:
        positions = torch.cumsum(input_mask.int(), dim=1) - 1
    _, suffix_out = model.llm(
        [prefix_tokens, suffix_tokens],
        positions=positions,
        mask=attn_mask,
        adarms_cond=[None, adarms_cond],
    )[0]
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def _joint_denoise(model, observation, noise, num_steps):
    """Euler denoise using the joint forward (same scheme as _denoise_actions)."""
    observation = model_mod.preprocess_observation(observation, train=False)
    observation = model_mod._observation_to_dtype(observation, model.embed_dtype)
    dt = -1.0 / num_steps
    x_t = noise.to(model.embed_dtype)
    t = 1.0
    while t >= -dt / 2:
        t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
        v_t = _joint_velocity(model, observation, x_t, t_tensor)
        x_t = x_t + dt * v_t
        t = t + dt
    return x_t


def _tok(tokenizer, subtask, device):
    t, v, ar, loss, kv = tokenizer.tokenize_with_subtask(
        TASK, np.linspace(-0.9, 0.9, 8), subtask
    )
    to = lambda a, d: torch.as_tensor(a, dtype=d, device=device)  # noqa: E731
    return (to(t, torch.long), to(v, torch.bool), to(ar, torch.bool),
            to(loss, torch.bool), to(kv, torch.bool))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import safetensors.torch

    device = torch.device(args.device)
    shape = json.load(open(pathlib.Path(args.ckpt) / "config.json"))
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
    sd = safetensors.torch.load_file(str(pathlib.Path(args.ckpt) / "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model = model.float().to(device).eval()

    tok = PaligemmaTokenizer(args.tokenizer, max_len=cfg.max_token_len)
    B = args.batch_size
    base = cfg.fake_obs(B)
    gen = torch.Generator().manual_seed(0)

    # Eval observation: prefix-only (generation form) for each row.
    pg = [_tok(tok, None, device) for _ in range(B)]  # response=None -> prefix only
    obs_gen = Observation(
        images={k: v.to(device) for k, v in base.images.items()},
        image_masks={k: v.to(device) for k, v in base.image_masks.items()},
        state=(torch.rand(B, cfg.action_dim, generator=gen) * 2 - 1).to(device),
        tokenized_prompt=torch.stack([p[0] for p in pg]),
        tokenized_prompt_mask=torch.stack([p[1] for p in pg]),
        token_ar_mask=torch.stack([p[2] for p in pg]),
        token_loss_mask=torch.stack([p[3] for p in pg]),
        token_kv_cache_mask=torch.stack([p[4] for p in pg]),
    )
    noise = torch.randn(B, cfg.action_horizon, cfg.action_dim, generator=gen).to(device)

    with torch.no_grad():
        a_eval, generation = model.reason_and_sample_actions(
            obs_gen, eos_token_id=tok.eos_token_id, num_steps=args.num_steps,
            noise=noise.clone(),
        )
        # Decode each row's generated subtask, rebuild the training-layout obs.
        eos_steps = generation["eos_steps"]
        gtokens = generation["tokens"]
        subtasks = [
            tok.decode(gtokens[r, : int(eos_steps[r].item())].tolist()).strip().rstrip(".")
            for r in range(B)
        ]
        print("generated subtasks:", subtasks)
        pj = [_tok(tok, s if s else "move to radio", device) for s in subtasks]
        obs_joint = Observation(
            images={k: v.to(device) for k, v in base.images.items()},
            image_masks={k: v.to(device) for k, v in base.image_masks.items()},
            state=obs_gen.state.clone(),
            tokenized_prompt=torch.stack([p[0] for p in pj]),
            tokenized_prompt_mask=torch.stack([p[1] for p in pj]),
            token_ar_mask=torch.stack([p[2] for p in pj]),
            token_loss_mask=torch.stack([p[3] for p in pj]),
            token_kv_cache_mask=torch.stack([p[4] for p in pj]),
        )
        a_joint = _joint_denoise(model, obs_joint, noise.clone(), args.num_steps)

    diff = (a_eval.float() - a_joint.float()).abs()
    print(f"=== eval-path parity ({pathlib.Path(args.ckpt).name}) ===")
    print(f"A_eval  norm/mean: {a_eval.float().norm():.4f} / {a_eval.float().mean():.4f}")
    print(f"A_joint norm/mean: {a_joint.float().norm():.4f} / {a_joint.float().mean():.4f}")
    print(f"max|A_eval - A_joint| = {diff.max():.5f}   mean = {diff.mean():.5f}")
    print(
        "VERDICT:",
        "PARITY OK (eval path faithful -> action expert is trained-bad)"
        if diff.max() < 1e-2
        else "DIVERGENCE (eval denoise != joint forward -> EVAL-PATH BUG is the cause)",
    )


if __name__ == "__main__":
    main()
