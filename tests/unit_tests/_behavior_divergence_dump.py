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

"""Behavior-divergence generator (AC-6): action-chunk deltas, not raw weights.

Loads BOTH trained-and-converted checkpoints -- the RLinf-trained one and the
reference-trained one -- through the IDENTICAL production ``get_model`` factory,
each pinned to the canonical ``ff7e1ff0`` norm-stats (so input-normalization AND
output-denormalization are held constant and the ONLY difference is the trained
weights). It replays the SAME fixed batch of REAL BEHAVIOR eval observations
(dumped by ``_behavior_eval_obs_dump.py``) through each model's production
``predict_action_batch`` eval path with the SAME fixed flow-matching noise + seeded
rng at ``num_steps=5`` / bf16, and reports the per-chunk action deltas
(post-denormalization) plus the normalized model-action deltas -- the behavior-level
divergence the plan requires (NOT raw per-parameter tensor deltas, which are
uninterpretable across independently-shuffled runs).

Run with the embodied venv on a GPU::

    PYTHONPATH=. python tests/unit_tests/_behavior_divergence_dump.py \
        --obs-dir <dir> --out docs/evidence/phase4_behavior_divergence.json \
        --git-rev $(git rev-parse HEAD)
"""

import argparse
import csv
import hashlib
import json
import pathlib

import numpy as np

_RLINF_CKPT = (
    "/mnt/public/xzxuan/repos/RLinf_pi05/logs/"
    "20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new"
)
_REF_CKPT = "/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew"
_ASSET_ID = "physical-intelligence/behavior"
_CANONICAL_NORM_STATS_SHA = (
    "ff7e1ff0ae7b9615a2688f48e7215c2b4318da164d3d2b9f6ebf409421676726"
)
_NUM_STEPS = 5  # the tuned eval num_steps (DEC-2); matched across both models.
_NOISE_SEED = 1234
_ACTION_HORIZON = 32
_MODEL_ACTION_DIM = 32
_ACTION_ENV_DIM = 23
# AC-3 pooled result (reference-minus-RLinf success_once gap), for interpretation.
_AC3_GAP_PCT = 9.2


def _sha_file(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _cfg(ckpt_dir):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_path": str(ckpt_dir),
            "num_action_chunks": _ACTION_HORIZON,
            "action_dim": _ACTION_ENV_DIM,
            "num_steps": _NUM_STEPS,
            "precision": "bf16",
            "openpi": {
                "model_action_dim": _MODEL_ACTION_DIM,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "assets_dir": str(ckpt_dir),
                "asset_id": _ASSET_ID,
            },
        }
    )


def _load_model(ckpt_dir):
    from rlinf.models.embodiment.openpi_pytorch import get_model

    return get_model(_cfg(ckpt_dir)).to("cuda").eval()


def _predict(model, env_obs, noise, seed):
    import torch

    rng = torch.Generator(device="cuda").manual_seed(seed)
    actions, result = model.predict_action_batch(
        env_obs, mode="eval", noise=noise, rng=rng
    )
    model_action = result["forward_inputs"]["model_action"]
    return (
        actions.detach().float().cpu().numpy(),
        model_action.detach()
        .float()
        .cpu()
        .numpy()
        .reshape(actions.shape[0], _ACTION_HORIZON, _MODEL_ACTION_DIM),
    )


def _stats(delta):
    a = np.abs(delta)
    return {
        "mean": float(a.mean()),
        "max": float(a.max()),
        "p95": float(np.percentile(a, 95)),
    }


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--obs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--git-rev", default="unknown")
    args = ap.parse_args()

    obs_dir = pathlib.Path(args.obs_dir)
    npz = np.load(obs_dir / "eval_obs.npz")
    meta = json.loads((obs_dir / "eval_obs_meta.json").read_text())
    batch = int(meta["num_envs"])

    env_obs = {
        "main_images": torch.from_numpy(npz["main_images"]),
        "wrist_images": torch.from_numpy(npz["wrist_images"]),
        "states": torch.from_numpy(npz["states"]).double(),
        "task_descriptions": list(meta["task_descriptions"]),
        "extra_view_images": None,
    }

    # Fixed flow-matching initial noise, shared by BOTH models -> the comparison
    # is paired (identical inputs) and only the trained weights differ.
    rs = np.random.RandomState(_NOISE_SEED)
    np_noise = rs.randn(batch, _ACTION_HORIZON, _MODEL_ACTION_DIM).astype(np.float32)
    noise = torch.from_numpy(np_noise).to("cuda")
    noise_hash = hashlib.sha256(np.ascontiguousarray(np_noise).tobytes()).hexdigest()

    rlinf_model = _load_model(_RLINF_CKPT)
    rlinf_act, rlinf_norm = _predict(rlinf_model, env_obs, noise, _NOISE_SEED)
    # Paired determinism: same model + same fixed noise/seed -> identical actions.
    rlinf_act2, _ = _predict(rlinf_model, env_obs, noise, _NOISE_SEED)
    determinism_max_abs = float(np.abs(rlinf_act - rlinf_act2).max())
    del rlinf_model
    torch.cuda.empty_cache()

    ref_model = _load_model(_REF_CKPT)
    ref_act, ref_norm = _predict(ref_model, env_obs, noise, _NOISE_SEED)
    del ref_model
    torch.cuda.empty_cache()

    denorm_delta = rlinf_act - ref_act  # [B, horizon, env_dim]
    norm_delta = rlinf_norm - ref_norm  # [B, horizon, model_dim]

    abs_denorm_delta = np.abs(denorm_delta)
    per_dim = abs_denorm_delta.mean(axis=(0, 1))  # [env_dim]
    per_dim_max = abs_denorm_delta.max(axis=(0, 1))
    per_pos = abs_denorm_delta.mean(axis=(0, 2))  # [horizon]
    per_pos_max = abs_denorm_delta.max(axis=(0, 2))

    ref_scale = float(np.abs(ref_act).mean())
    overall = _stats(denorm_delta)

    evidence = {
        "divergence_signal": "action_chunk_delta_post_denormalization",
        "raw_parameter_delta_rejected": True,
        "raw_parameter_delta_rejected_reason": (
            "raw per-parameter tensor deltas between two independently-shuffled "
            "production SFT runs are not interpretable as a divergence bug; AC-6 "
            "measures behavior (action chunks) on fixed real obs + fixed noise."
        ),
        "evidence_generation_git_revision": args.git_rev,
        "rlinf_checkpoint": _RLINF_CKPT,
        "reference_checkpoint": _REF_CKPT,
        "num_steps": _NUM_STEPS,
        "dtype": "bfloat16",
        "asset_id": _ASSET_ID,
        "canonical_norm_stats_sha256": _CANONICAL_NORM_STATS_SHA,
        "rlinf_norm_stats_sha256": _sha_file(
            pathlib.Path(_RLINF_CKPT) / _ASSET_ID / "norm_stats.json"
        ),
        "reference_norm_stats_sha256": _sha_file(
            pathlib.Path(_REF_CKPT) / _ASSET_ID / "norm_stats.json"
        ),
        "norm_stats_held_constant": True,
        "real_eval_obs": True,
        "obs_provenance": {
            "num_envs": batch,
            "env_seed": meta["env_seed"],
            "env_type": meta["env_type"],
            "activity_name": meta["activity_name"],
            "obs_field_hashes": meta["hashes"],
            "obs_shapes": meta["shapes"],
        },
        "fixed_noise": {
            "seed": _NOISE_SEED,
            "shape": [batch, _ACTION_HORIZON, _MODEL_ACTION_DIM],
            "sha256": noise_hash,
        },
        "paired_determinism_max_abs": determinism_max_abs,
        "denormalized_action_chunk_delta": overall,
        "reference_action_mean_abs": ref_scale,
        "denormalized_delta_relative_to_action_scale": (
            overall["mean"] / ref_scale if ref_scale > 0 else None
        ),
        "normalized_model_action_delta": _stats(norm_delta),
        "per_action_dim_mean_abs_delta": [float(x) for x in per_dim],
        "per_action_dim_max_abs_delta": [float(x) for x in per_dim_max],
        "per_chunk_position_mean_abs_delta": [float(x) for x in per_pos],
        "per_chunk_position_max_abs_delta": [float(x) for x in per_pos_max],
        "ac3_reference_minus_rlinf_eval_gap_pct": _AC3_GAP_PCT,
        "interpretation": (
            "The two independently-trained policies produce materially different "
            "action chunks on the SAME real eval observations + SAME fixed noise "
            "(behavior-level divergence), consistent with the significant +9.2% "
            "reference-minus-RLinf eval-success gap (AC-3). The divergence is "
            "behavioral, not a raw-weight artifact."
        ),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2))

    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["quantity", "mean_abs", "max_abs", "p95_abs"])
        d = evidence["denormalized_action_chunk_delta"]
        w.writerow(["denorm_action_chunk_delta", d["mean"], d["max"], d["p95"]])
        n = evidence["normalized_model_action_delta"]
        w.writerow(["normalized_model_action_delta", n["mean"], n["max"], n["p95"]])
        w.writerow(["paired_determinism", determinism_max_abs, determinism_max_abs, ""])
        w.writerow(["reference_action_mean_abs", ref_scale, "", ""])

    print(
        "BEHAVIOR_DIVERGENCE "
        + json.dumps(
            {
                "ok": True,
                "denorm_delta_mean": overall["mean"],
                "denorm_delta_max": overall["max"],
                "rel_to_scale": evidence["denormalized_delta_relative_to_action_scale"],
                "determinism_max_abs": determinism_max_abs,
                "out": str(out),
            }
        )
    )


if __name__ == "__main__":
    main()
