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

"""Teacher-forced normalized action-flow MSE proxy for VLA vs VLM-VLA.

This diagnostic loads two new-format Pi0 checkpoints and evaluates their
flow-matching action MSE on matched saved batch variants from
``capture_fixed_batch.py``. The VLM-VLA side uses teacher-forced subtask tokens;
the VLA side uses the action-only prompt variant from the same raw frames.
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

from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import (  # noqa: E402
    Observation,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import (  # noqa: E402
    Pi0Config,
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
    # The training loss samples Beta(1.5, 1.0). For beta=1 the inverse CDF is
    # u ** (1 / alpha), so this is generator-controlled and distribution-matched.
    u = torch.rand(shape[0], device=device, dtype=dtype, generator=gen)
    time = u.pow(1.0 / 1.5) * 0.999 + 0.001
    return noise, time


def _move_tensor(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


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


def _load_variant(
    batch_path: pathlib.Path,
    variant: str,
    device: torch.device,
) -> tuple[Observation, torch.Tensor, dict[str, Any], str | None]:
    payload = torch.load(batch_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("Batch payload must be a dict saved by capture_fixed_batch.py.")
    variants = payload.get("variants")
    if isinstance(variants, dict) and variant in variants:
        item = variants[variant]
    elif variant == "vlm_vla" and "observation" in payload and "actions" in payload:
        item = payload
    else:
        raise KeyError(f"Variant {variant!r} not found in {batch_path}.")

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
    mode: str,
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
        mode=mode,
    )
    model = cfg.create()
    state_dict = safetensors.torch.load_file(str(ckpt / "model.safetensors"))
    model.load_state_dict(state_dict, strict=True)
    return model.to(dtype=dtype, device=device).eval()


@torch.no_grad()
def _action_mse(
    model,
    observation: Observation,
    actions: torch.Tensor,
    *,
    noise: torch.Tensor,
    time: torch.Tensor,
) -> float:
    output = model.compute_loss(
        observation,
        actions,
        train=False,
        noise=noise,
        time=time,
    )
    if isinstance(output, dict):
        return float(output["action_loss"])
    return float(output.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--broken-ckpt", required=True, type=pathlib.Path)
    parser.add_argument("--batch-path", required=True, type=pathlib.Path)
    parser.add_argument("--baseline-mode", default="vla", choices=("vla", "vlm_vla"))
    parser.add_argument("--broken-mode", default="vlm_vla", choices=("vla", "vlm_vla"))
    parser.add_argument("--baseline-variant", default="vla")
    parser.add_argument("--broken-variant", default="vlm_vla")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="Weight/compute dtype for the diagnostic forward.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threshold-ratio", type=float, default=1.5)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    base_obs, base_actions, metadata, base_fp = _load_variant(
        args.batch_path, args.baseline_variant, device
    )
    broken_obs, broken_actions, _, broken_fp = _load_variant(
        args.batch_path, args.broken_variant, device
    )
    if base_fp is not None and broken_fp is not None and base_fp != broken_fp:
        raise ValueError(
            "Baseline and broken batch variants do not share the same "
            f"fingerprint: {base_fp} != {broken_fp}."
        )
    if base_actions.shape != broken_actions.shape:
        raise ValueError(
            f"Action shape mismatch: {base_actions.shape} vs {broken_actions.shape}."
        )

    noise, time = _sample_flow_inputs(
        base_actions.shape,
        device=device,
        dtype=base_actions.dtype,
        seed=args.seed,
    )

    baseline = _load_pi0(
        args.baseline_ckpt,
        args.baseline_mode,
        device,
        dtype,
        args.dtype,
    )
    baseline_mse = _action_mse(
        baseline, base_obs, base_actions, noise=noise, time=time
    )
    del baseline
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    broken = _load_pi0(
        args.broken_ckpt,
        args.broken_mode,
        device,
        dtype,
        args.dtype,
    )
    broken_mse = _action_mse(
        broken, broken_obs, broken_actions, noise=noise, time=time
    )
    del broken
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ratio = broken_mse / baseline_mse if baseline_mse > 0 else float("inf")
    passes_gate = ratio <= args.threshold_ratio

    result = {
        "batch_path": str(args.batch_path),
        "batch_fingerprint": base_fp or broken_fp,
        "batch_metadata": metadata,
        "baseline_ckpt": str(args.baseline_ckpt),
        "broken_ckpt": str(args.broken_ckpt),
        "baseline_mode": args.baseline_mode,
        "broken_mode": args.broken_mode,
        "baseline_variant": args.baseline_variant,
        "broken_variant": args.broken_variant,
        "seed": args.seed,
        "dtype": args.dtype,
        "noise_fingerprint": _tensor_fingerprint(noise),
        "time_fingerprint": _tensor_fingerprint(time),
        "threshold_ratio": args.threshold_ratio,
        "M_base": baseline_mse,
        "M_broken": broken_mse,
        "ratio": ratio,
        "passes_gate": passes_gate,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
