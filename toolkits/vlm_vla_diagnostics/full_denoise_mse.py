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

"""FULL-denoise action MSE vs ground truth: baseline-VLA vs broken-VLM-VLA.

The one-step flow-MSE proxy only checks the velocity field at a single (noise,
time). This runs the COMPLETE Euler denoise (eval `num_steps`) from the same
noise and compares the produced actions to the dataset ground-truth actions, in
normalized action space. It answers whether the eval-style actions are actually
good (=> closed-loop dynamics is the issue) or bad (=> the action expert is
trained-bad in the full-denoise sense). Teacher-forced subtask for the VLM-VLA
side (uses the saved vlm_vla batch variant's prompt).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi_pytorch.pi0_model import model as model_mod  # noqa: E402
from toolkits.vlm_vla_diagnostics.action_mse_proxy import (  # noqa: E402
    _load_pi0,
    _load_variant,
)
from toolkits.vlm_vla_diagnostics.eval_parity_check import _joint_velocity  # noqa: E402


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).pow(2).mean())


def _joint_denoise(model, observation, noise, num_steps):
    """Full Euler denoise via the joint forward (teacher-forced subtask)."""
    observation = model_mod.preprocess_observation(observation, train=False)
    observation = model_mod._observation_to_dtype(observation, model.embed_dtype)
    dt = -1.0 / num_steps
    x_t = noise.to(model.embed_dtype)
    t = 1.0
    while t >= -dt / 2:
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
        x_t = x_t + dt * _joint_velocity(model, observation, x_t, tt)
        t = t + dt
    return x_t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--broken-ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--batch-path", required=True, type=pathlib.Path)
    ap.add_argument("--num-steps", type=int, default=5)  # eval default
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32  # measure in fp32 (eval-path bf16 already ruled out)

    obs_vla, gt, _, _ = _load_variant(args.batch_path, "vla", device)
    obs_vlm, gt2, _, _ = _load_variant(args.batch_path, "vlm_vla", device)
    assert _mse(gt, gt2) < 1e-8, "vla/vlm_vla variants must share GT actions"
    B, H, D = gt.shape
    gen = torch.Generator(device=device).manual_seed(args.seed)
    noise = torch.randn(B, H, D, device=device, dtype=dtype, generator=gen)

    base = _load_pi0(args.baseline_ckpt, "vla", device, dtype, "float32")
    with torch.no_grad():
        base_actions = base.sample_actions(obs_vla, num_steps=args.num_steps, noise=noise.clone())
    base_mse = _mse(base_actions, gt)
    del base
    torch.cuda.empty_cache()

    broken = _load_pi0(args.broken_ckpt, "vlm_vla", device, dtype, "float32")
    with torch.no_grad():
        broken_actions = _joint_denoise(broken, obs_vlm, noise.clone(), args.num_steps)
    broken_mse = _mse(broken_actions, gt)

    # Also a noise->GT reference (how far raw noise is) for scale.
    noise_mse = _mse(noise, gt)
    print("=== FULL-denoise action MSE vs ground truth (normalized) ===")
    print(f"batch={B}x{H}x{D}  num_steps={args.num_steps}  device={device}")
    print(f"raw-noise MSE (no denoise):        {noise_mse:.6f}")
    print(f"baseline-VLA  full-denoise MSE:    {base_mse:.6f}")
    print(f"broken-VLMVLA full-denoise MSE:    {broken_mse:.6f}")
    print(f"ratio broken/baseline:             {broken_mse / max(base_mse, 1e-9):.3f}")
    print(
        "VERDICT:",
        "actions COMPARABLE -> closed-loop dynamics is the issue, not action quality"
        if broken_mse <= 2.0 * base_mse
        else "broken actions MUCH WORSE -> action expert IS bad in full denoise",
    )


if __name__ == "__main__":
    main()
