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

"""Teacher-forced train/eval velocity parity for the real ``vlm_vla`` model.

This diagnostic compares the action velocity predicted by two paths on the same
fixed real batch:

* the joint SFT forward layout: image + full text response + action suffix;
* the eval layout: image + generation prefix + teacher-forced response tokens in
  a frozen :class:`StaticKVCache` + the same action suffix.

Only non-EOS response tokens are written to the eval cache, matching production
generation where EOS never becomes visible to the action expert. The comparison
uses one fixed noise/time sample and reports the max absolute velocity
difference against a bf16-oriented tolerance.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import pathlib
import sys
from typing import Any

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
from rlinf.models.embodiment.openpi_pytorch.pi0_model.static_kv_cache import (  # noqa: E402
    StaticKVCache,
    left_to_right_align,
)


def _update_hash_from_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy()
    h.update(str(array.shape).encode("utf-8"))
    h.update(str(array.dtype).encode("utf-8"))
    h.update(array.tobytes())


def _tensor_fingerprint(tensor: torch.Tensor) -> str:
    h = hashlib.sha256()
    _update_hash_from_tensor(h, tensor)
    return h.hexdigest()


def _sample_flow_inputs(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(shape, device=device, dtype=dtype, generator=gen)
    u = torch.rand(shape[0], device=device, dtype=dtype, generator=gen)
    time = u.pow(1.0 / 1.5) * 0.999 + 0.001
    return noise, time


def _move_tensor(value: Any, device: torch.device) -> Any:
    return value.to(device) if isinstance(value, torch.Tensor) else value


def _move_observation(observation: Observation, device: torch.device) -> Observation:
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


def _load_vlm_vla_batch(
    batch_path: pathlib.Path,
    device: torch.device,
) -> tuple[Observation, torch.Tensor, dict[str, Any], str | None]:
    payload = torch.load(batch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Batch payload must be a dict saved by capture_fixed_batch.py.")
    variants = payload.get("variants")
    if isinstance(variants, dict) and "vlm_vla" in variants:
        item = variants["vlm_vla"]
    elif "observation" in payload and "actions" in payload:
        item = payload
    else:
        raise KeyError(f"No vlm_vla variant found in {batch_path}.")

    observation = item["observation"]
    if isinstance(observation, dict):
        observation = Observation.from_dict(observation)
    if not isinstance(observation, Observation):
        raise TypeError(f"Unsupported observation payload: {type(observation)!r}")

    actions = item["actions"]
    if not isinstance(actions, torch.Tensor):
        actions = torch.as_tensor(actions)
    return (
        _move_observation(observation, device),
        actions.to(device),
        payload.get("metadata", {}),
        item.get("fingerprint"),
    )


def _load_pi0(
    ckpt: pathlib.Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
    dtype_name: str,
):
    import safetensors.torch

    with open(ckpt / "config.json", encoding="utf-8") as f:
        shape = json.load(f)
    cfg = Pi0Config(
        pi05=bool(shape.get("pi05", True)),
        action_horizon=int(shape["action_horizon"]),
        action_dim=int(shape["action_dim"]),
        max_token_len=int(shape.get("max_token_len", 200)),
        paligemma_variant=str(shape["paligemma_variant"]),
        action_expert_variant=str(shape["action_expert_variant"]),
        dtype=dtype_name,
        pcd=bool(shape.get("pcd", False)),
        mode="vlm_vla",
    )
    model = cfg.create()
    state_dict = safetensors.torch.load_file(str(ckpt / "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype=dtype, device=device).eval()


def _joint_velocity(model, observation: Observation, x_t: torch.Tensor, time: torch.Tensor):
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )

    batch_size = x_t.shape[0]
    input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    if prefix_ar_mask.dim() == 2:
        suffix_ar_mask = suffix_ar_mask.unsqueeze(0).expand(batch_size, -1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=1)
    else:
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    prefix_len = prefix_mask.shape[1]
    attn_mask = block_suffix_from_excluded_prefix(
        attn_mask,
        observation.token_kv_cache_mask,
        prefix_len=prefix_len,
    )

    text_len = observation.token_kv_cache_mask.shape[1]
    position_mask = input_mask.clone()
    position_mask[:, prefix_len - text_len : prefix_len] &= (
        observation.token_kv_cache_mask
    )
    positions = torch.cumsum(position_mask.int(), dim=1) - 1

    _, suffix_out = model.llm(
        [prefix_tokens, suffix_tokens],
        positions=positions,
        mask=attn_mask,
        adarms_cond=[None, adarms_cond],
    )[0]
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def _split_prefix_and_response(observation: Observation) -> tuple[Observation, torch.Tensor, torch.Tensor]:
    tokens = observation.tokenized_prompt
    valid = observation.tokenized_prompt_mask
    loss = observation.token_loss_mask
    kv = observation.token_kv_cache_mask
    if tokens is None or valid is None or loss is None or kv is None:
        raise ValueError("vlm_vla observation must include token masks.")

    batch_size, text_len = tokens.shape
    prefix_tokens = torch.zeros_like(tokens)
    prefix_valid = torch.zeros_like(valid)
    response_rows: list[torch.Tensor] = []
    max_response_len = 0
    for row in range(batch_size):
        response_mask = loss[row] & kv[row]
        response_idx = torch.nonzero(response_mask, as_tuple=False).flatten()
        if response_idx.numel() == 0:
            prefix_end = int(valid[row].sum().item())
            response = tokens.new_empty((0,))
        else:
            prefix_end = int(response_idx[0].item())
            response = tokens[row, response_idx]
        prefix_tokens[row, :prefix_end] = tokens[row, :prefix_end]
        prefix_valid[row, :prefix_end] = True
        response_rows.append(response)
        max_response_len = max(max_response_len, int(response.numel()))

    if max_response_len == 0:
        raise ValueError("No teacher-forced non-EOS response tokens found.")

    response_tokens = torch.zeros(
        batch_size,
        max_response_len,
        dtype=tokens.dtype,
        device=tokens.device,
    )
    response_valid = torch.zeros(
        batch_size,
        max_response_len,
        dtype=torch.bool,
        device=tokens.device,
    )
    for row, response in enumerate(response_rows):
        length = int(response.numel())
        if length:
            response_tokens[row, :length] = response
            response_valid[row, :length] = True

    prefix_observation = Observation(
        images=observation.images,
        image_masks=observation.image_masks,
        state=observation.state,
        tokenized_prompt=prefix_tokens,
        tokenized_prompt_mask=prefix_valid,
        token_ar_mask=torch.zeros(text_len, dtype=torch.bool, device=tokens.device)
        .unsqueeze(0)
        .expand(batch_size, -1),
        token_loss_mask=torch.zeros_like(loss),
        token_kv_cache_mask=prefix_valid.clone(),
        pcd_xyz=observation.pcd_xyz,
    )
    return prefix_observation, response_tokens, response_valid


def _teacher_forced_cache(
    model,
    prefix_observation: Observation,
    response_tokens: torch.Tensor,
    response_valid: torch.Tensor,
) -> tuple[StaticKVCache, torch.Tensor]:
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(prefix_observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_tokens, prefix_mask, prefix_attn_mask = left_to_right_align(
        prefix_tokens,
        prefix_mask,
        prefix_attn_mask,
    )

    batch_size, prefill_size = prefix_mask.shape
    response_len = response_tokens.shape[1]
    device = prefix_tokens.device
    paligemma_config = model.llm.configs[0]
    cache = StaticKVCache(
        batch_size=batch_size,
        max_len=prefill_size + response_len,
        num_layers=paligemma_config.depth,
        num_kv_heads=paligemma_config.num_kv_heads,
        head_dim=paligemma_config.head_dim,
        dtype=model.embed_dtype,
        device=device,
    )

    cache.set_write_col(0)
    prefill_attn_mask = torch.nn.functional.pad(prefix_attn_mask, (0, response_len))
    prefix_positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
    model.llm(
        [prefix_tokens, None],
        positions=prefix_positions,
        mask=prefill_attn_mask,
        kv_cache=cache,
    )

    prefill_len = prefix_mask.sum(dim=-1)
    generated_valid = torch.zeros_like(response_valid)
    for step in range(response_len):
        generated_valid[:, step] = response_valid[:, step]
        token_embedding = model.llm.embed(response_tokens[:, step : step + 1])
        step_positions = (prefill_len + step).long()[:, None]
        keys_valid = torch.cat([prefix_mask, generated_valid], dim=1)
        cache.set_write_col(prefill_size + step)
        model.llm(
            [token_embedding, None],
            positions=step_positions,
            mask=keys_valid[:, None, :],
            kv_cache=cache,
        )
    cache.freeze()
    return cache, torch.cat([prefix_mask, generated_valid], dim=1)


def _cached_velocity(
    model,
    observation: Observation,
    kv_cache: StaticKVCache,
    kv_valid_mask: torch.Tensor,
    x_t: torch.Tensor,
    time: torch.Tensor,
):
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        observation, x_t, time
    )
    suffix_len = suffix_tokens.shape[1]
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    cache_to_suffix_mask = kv_valid_mask[:, None, :].expand(
        x_t.shape[0],
        suffix_len,
        kv_valid_mask.shape[1],
    )
    full_attn_mask = torch.cat([cache_to_suffix_mask, suffix_attn_mask], dim=-1)
    suffix_positions = (
        kv_valid_mask.sum(dim=-1)[:, None] + torch.cumsum(suffix_mask.int(), dim=-1) - 1
    )
    _, suffix_out = model.llm(
        [None, suffix_tokens],
        positions=suffix_positions,
        mask=full_attn_mask,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )[0]
    return model.action_out_proj(suffix_out[:, -model.action_horizon :])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--batch-path", required=True, type=pathlib.Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=("float32", "bfloat16"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    observation, actions, metadata, batch_fp = _load_vlm_vla_batch(
        args.batch_path,
        device,
    )
    model = _load_pi0(
        args.ckpt,
        device=device,
        dtype=dtype,
        dtype_name=args.dtype,
    )

    noise, time = _sample_flow_inputs(
        actions.shape,
        device=device,
        dtype=actions.dtype,
        seed=args.seed,
    )
    time_expanded = time[:, None, None]
    x_t = (time_expanded * noise + (1 - time_expanded) * actions).to(dtype)
    time = time.to(dtype)

    with torch.no_grad():
        full_observation = model_mod.preprocess_observation(observation, train=False)
        full_observation = model_mod._observation_to_dtype(full_observation, dtype)
        train_velocity = _joint_velocity(model, full_observation, x_t, time)

        prefix_observation, response_tokens, response_valid = _split_prefix_and_response(
            observation
        )
        prefix_observation = model_mod.preprocess_observation(
            prefix_observation,
            train=False,
        )
        prefix_observation = model_mod._observation_to_dtype(prefix_observation, dtype)
        cache, cache_valid_mask = _teacher_forced_cache(
            model,
            prefix_observation,
            response_tokens,
            response_valid,
        )
        eval_velocity = _cached_velocity(
            model,
            prefix_observation,
            cache,
            cache_valid_mask,
            x_t,
            time,
        )

    diff = (train_velocity.float() - eval_velocity.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    result = {
        "ckpt": str(args.ckpt),
        "batch_path": str(args.batch_path),
        "batch_fingerprint": batch_fp,
        "batch_metadata": metadata,
        "seed": args.seed,
        "dtype": args.dtype,
        "tolerance": args.tolerance,
        "noise_fingerprint": _tensor_fingerprint(noise),
        "time_fingerprint": _tensor_fingerprint(time.float()),
        "response_lengths": response_valid.sum(dim=-1).detach().cpu().tolist(),
        "max_abs_velocity_diff": max_abs,
        "mean_abs_velocity_diff": mean_abs,
        "passes_tolerance": max_abs <= args.tolerance,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
