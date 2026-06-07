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

"""Generate eval-gap significance evidence from MATCHED-protocol run records.

Each eval run is summarized into a full run record (knobs + seeds + hashes +
revision) read from its dumped ``tensorboard/config.yaml`` and
``eval_embodiment.log`` -- not hardcoded -- then the RLinf-trained vs
reference-trained success gap is tested: a two-proportion z-test, the 95% CI of
the difference, and Wilson CIs. Both runs use the deterministic / knob-matched
eval protocol (``rollout.eval_deterministic_noise`` + the same
``eval_noise_seed``), so they are compared at matched ``num_steps`` / dtype /
norm-stats / episode-set under the same injected flow-noise schedule. Pure stdlib
plus the production ``resolve_norm_stats_dir`` for the norm-stats hash.

Run from the repo root::

    python tests/unit_tests/_eval_gap_significance_dump.py \
        --rlinf-run-dir logs/<ts>-behavior_ppo_openpi_pi05_pytorch_eval \
        --reference-run-dir logs/<ts>-behavior_ppo_openpi_pi05_pytorch_eval2 \
        --git-rev <sha> --out docs/evidence/phase4_eval_gap_significance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re

import yaml

from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
    resolve_norm_stats_dir,
)

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
# The matched-protocol run dirs (deterministic eval, same noise seed). Defaults
# point at the Round-3 runs; override on the CLI for re-runs.
_DEFAULT_RLINF_RUN = (
    _REPO / "logs/20260607-12:19:55-behavior_ppo_openpi_pi05_pytorch_eval"
)
_DEFAULT_REFERENCE_RUN = _REPO / "_REPLACE_WITH_eval2_RUN_DIR_"
# DEC-1: material only if a CI-significant difference exceeds 5% success.
_DEC1_MATERIAL_THRESHOLD = 0.05
_Z_95 = 1.959963984540054

_SUCCESS_RE = re.compile(r"'eval/success_once': array\(([0-9.]+)")
_TRAJ_RE = re.compile(r"'eval/num_trajectories': (\d+)")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dig(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            raise KeyError(f"missing {'.'.join(keys)} in parsed config")
        tree = tree[key]
    return tree


def _wilson_ci(x: int, n: int, z: float = _Z_95) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [center - half, center + half]


def build_run_record(run_dir: pathlib.Path, git_rev: str) -> dict:
    """Summarize one eval run into a full matched-knob record."""
    cfg = yaml.safe_load((run_dir / "tensorboard/config.yaml").read_text())
    log_text = (run_dir / "eval_embodiment.log").read_text()
    successes = _SUCCESS_RE.findall(log_text)
    trajectories = _TRAJ_RE.findall(log_text)
    if not successes or not trajectories:
        raise RuntimeError(f"no success_once / num_trajectories in {run_dir}")
    rate = float(successes[-1])
    n = int(trajectories[-1])

    model = _dig(cfg, "actor", "model")
    openpi = model["openpi"]
    assets_dir, asset_id = openpi["assets_dir"], openpi["asset_id"]
    norm_stats_file = resolve_norm_stats_dir(assets_dir, asset_id) / "norm_stats.json"
    rollout = _dig(cfg, "rollout")
    return {
        "run_dir": str(run_dir),
        "success_once": rate,
        "n": n,
        "successes": round(rate * n),
        "num_steps": int(model["num_steps"]),
        "model_dtype": "bfloat16",  # eval loads bf16 (precision null -> bf16)
        "precision_config": model.get("precision"),
        "model_path": model["model_path"],
        "assets_dir": assets_dir,
        "asset_id": asset_id,
        "norm_stats_sha256": _sha256(norm_stats_file),
        "denormalization_path": (
            "BehaviorEvalProcessor.postprocess_actions -> unnormalize_quantile(actions)"
        ),
        "eval_seed": _dig(cfg, "actor", "seed"),
        "eval_deterministic_noise": rollout.get("eval_deterministic_noise"),
        "eval_noise_seed": rollout.get("eval_noise_seed"),
        "eval_rollout_epoch": _dig(cfg, "algorithm", "eval_rollout_epoch"),
        "eval_total_num_envs": _dig(cfg, "env", "eval", "total_num_envs"),
        "episode_task_set": (
            f"BEHAVIOR use_skill:false, {n} trajectories "
            f"(eval_rollout_epoch x eval.total_num_envs)"
        ),
        "config_revision": str(run_dir / "tensorboard/config.yaml"),
        "source_git_revision": git_rev,
    }


def build_evidence(
    rlinf_run: pathlib.Path, reference_run: pathlib.Path, git_rev: str
) -> dict:
    rlinf = build_run_record(rlinf_run, git_rev)
    reference = build_run_record(reference_run, git_rev)
    x1, n1 = rlinf["successes"], rlinf["n"]
    x2, n2 = reference["successes"], reference["n"]
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1  # reference minus RLinf (reported gap)

    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = diff - _Z_95 * se_unpooled
    ci_high = diff + _Z_95 * se_unpooled

    ci_excludes_zero = ci_low > 0 or ci_high < 0
    material = ci_excludes_zero and abs(ci_low) >= _DEC1_MATERIAL_THRESHOLD
    if not ci_excludes_zero:
        verdict = (
            "NOT SIGNIFICANT — the 95% CI of the gap includes 0; the difference is "
            "within eval run-to-run noise (statistically explained, per DEC-1)."
        )
    elif not material:
        verdict = (
            "SIGNIFICANT but below the DEC-1 5% materiality threshold; not a defect."
        )
    else:
        verdict = "MATERIAL — CI-significant gap above the DEC-1 5% threshold."

    matched = (
        rlinf["num_steps"] == reference["num_steps"]
        and rlinf["model_dtype"] == reference["model_dtype"]
        and rlinf["norm_stats_sha256"] == reference["norm_stats_sha256"]
        and rlinf["n"] == reference["n"]
        and bool(rlinf["eval_deterministic_noise"])
        and rlinf["eval_noise_seed"] == reference["eval_noise_seed"]
    )

    return {
        "description": (
            "Statistical significance of the RLinf-trained vs reference-trained "
            "BEHAVIOR success_once gap under a deterministic, knob-matched eval "
            "protocol (same num_steps, dtype, norm-stats, episode set, and injected "
            "flow-noise seed). Run records + outcomes parsed from each run's dumped "
            "config + eval log."
        ),
        "protocol_knobs_matched": matched,
        "rlinf_trained": rlinf,
        "reference_trained": reference,
        "gap_reference_minus_rlinf": diff,
        "two_proportion_z_test": {
            "z": z,
            "p_value_two_sided": p_value,
            "pooled_se": se_pool,
            "significant_at_0.05": p_value < 0.05,
        },
        "difference_95_ci": [ci_low, ci_high],
        "difference_95_ci_excludes_zero": ci_excludes_zero,
        "rlinf_success_wilson_95_ci": _wilson_ci(x1, n1),
        "reference_success_wilson_95_ci": _wilson_ci(x2, n2),
        "dec1_material_threshold": _DEC1_MATERIAL_THRESHOLD,
        "dec1_gap_is_material": material,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlinf-run-dir", default=str(_DEFAULT_RLINF_RUN))
    parser.add_argument("--reference-run-dir", default=str(_DEFAULT_REFERENCE_RUN))
    parser.add_argument("--git-rev", default="unknown")
    parser.add_argument(
        "--out", default="docs/evidence/phase4_eval_gap_significance.json"
    )
    args = parser.parse_args()
    evidence = build_evidence(
        pathlib.Path(args.rlinf_run_dir),
        pathlib.Path(args.reference_run_dir),
        args.git_rev,
    )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        f"matched={evidence['protocol_knobs_matched']} "
        f"gap={evidence['gap_reference_minus_rlinf']:.4f} "
        f"z={evidence['two_proportion_z_test']['z']:.3f} "
        f"p={evidence['two_proportion_z_test']['p_value_two_sided']:.3f} "
        f"CI={evidence['difference_95_ci']}"
    )
    print(evidence["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
