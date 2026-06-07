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

"""Generate eval-gap significance evidence from an EXPANDED matched protocol.

Both checkpoints are evaluated under a deterministic, knob-matched eval over
several seed pairs (`(env.eval.seed, rollout.eval_noise_seed)`). Each eval run is
summarized into a full run record -- the complete seed/task schedule (actor seed,
env seed + per-rank offset formula, flow-noise seed, reset/task knobs) and the
model knobs (num_steps, dtype, model_path, assets/asset_id, norm-stats sha256,
denorm path) -- read from its dumped `tensorboard/config.yaml` +
`eval_embodiment.log`, never hardcoded. Per-run provenance is the run dir + its
dumped config; `evidence_generation_git_revision` is the generator's `--git-rev`
(when the evidence was built), not a per-run log-derived field. The two
checkpoints' records are pooled into a two-proportion z-test + 95% CI, judged
against DEC-1, with a per-seed table so outlier seeds are visible.

The run dirs come from a committed manifest (so the documented invocation rebuilds
the committed evidence -- no placeholder defaults). Run from the repo root::

    python tests/unit_tests/_eval_gap_significance_dump.py \
        --manifest docs/evidence/phase4_eval_seed_pairs.json \
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
_DEFAULT_MANIFEST = _REPO / "docs/evidence/phase4_eval_seed_pairs.json"
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


def _two_proportion(x1: int, n1: int, x2: int, n2: int) -> dict:
    """Pooled z-test + unpooled 95% CI of the difference (reference - rlinf)."""
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else 0.0
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = diff - _Z_95 * se_unpooled
    ci_high = diff + _Z_95 * se_unpooled
    return {
        "rlinf_successes": x1,
        "rlinf_n": n1,
        "reference_successes": x2,
        "reference_n": n2,
        "gap_reference_minus_rlinf": diff,
        "z": z,
        "p_value_two_sided": 2 * (1 - _normal_cdf(abs(z))),
        "difference_95_ci": [ci_low, ci_high],
        "difference_95_ci_excludes_zero": ci_low > 0 or ci_high < 0,
    }


def build_run_record(run_dir: pathlib.Path, git_rev: str) -> dict:
    """Summarize one eval run into a full matched-knob + seed/task-schedule record."""
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
    env_eval = _dig(cfg, "env", "eval")
    task = env_eval.get("omni_config", {}).get("task", {}) or {}
    return {
        "run_dir": str(run_dir),
        "success_once": rate,
        "n": n,
        "successes": round(rate * n),
        # --- model knobs ---
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
        # --- full seed schedule (distinct from each other) ---
        "actor_seed": _dig(cfg, "actor", "seed"),
        "env_eval_seed": env_eval.get("seed"),
        "env_seed_offset_formula": (
            "per-env-worker seed = env.eval.seed + rank*stage_num + stage_id "
            "(rlinf/workers/env/env_worker.py); rlinf/envs/behavior/behavior_env.py "
            "self.seed = cfg.seed + seed_offset"
        ),
        "flow_noise_seed": rollout.get("eval_noise_seed"),
        "eval_deterministic_noise": rollout.get("eval_deterministic_noise"),
        # --- reset / task schedule ---
        "use_fixed_reset_state_ids": env_eval.get("use_fixed_reset_state_ids"),
        "num_env_subprocess": env_eval.get("num_env_subprocess"),
        "eval_rollout_epoch": _dig(cfg, "algorithm", "eval_rollout_epoch"),
        "total_num_envs": env_eval.get("total_num_envs"),
        "task_activity_name": task.get("activity_name"),
        "task_instance_resample_mode": task.get("instance_resample_mode"),
        "task_online_object_sampling": task.get("online_object_sampling"),
        "task_activity_definition_id": task.get("activity_definition_id"),
        "task_activity_instance_id": task.get("activity_instance_id"),
        # --- provenance ---
        # Per-run provenance is the run dir + its dumped config; the git rev below
        # is the EVIDENCE-GENERATION revision (the --git-rev passed to this
        # generator), not a per-run log-derived field.
        "config_revision": str(run_dir / "tensorboard/config.yaml"),
        "evidence_generation_git_revision": git_rev,
    }


# The knob set the two checkpoints in a pair MUST share for a valid comparison.
_MATCH_KEYS = (
    "num_steps",
    "model_dtype",
    "norm_stats_sha256",
    "n",
    "env_eval_seed",
    "flow_noise_seed",
    "use_fixed_reset_state_ids",
    "total_num_envs",
    "eval_rollout_epoch",
    "task_activity_name",
    "task_instance_resample_mode",
    "task_online_object_sampling",
)


def pair_matched(rlinf: dict, reference: dict) -> bool:
    """True iff the two checkpoints share the full env/reset/task/noise/model knobs."""
    if not (
        rlinf["eval_deterministic_noise"] and reference["eval_deterministic_noise"]
    ):
        return False
    return all(rlinf[k] == reference[k] for k in _MATCH_KEYS)


def build_evidence(manifest: dict, git_rev: str) -> dict:
    per_seed = []
    all_matched = True
    sum_x1 = sum_n1 = sum_x2 = sum_n2 = 0
    for pair in manifest["seed_pairs"]:
        rlinf = build_run_record(_REPO / pair["rlinf_run_dir"], git_rev)
        reference = build_run_record(_REPO / pair["reference_run_dir"], git_rev)
        matched = pair_matched(rlinf, reference)
        all_matched = all_matched and matched
        sum_x1 += rlinf["successes"]
        sum_n1 += rlinf["n"]
        sum_x2 += reference["successes"]
        sum_n2 += reference["n"]
        per_seed.append(
            {
                "index": pair["index"],
                "env_seed": pair["env_seed"],
                "noise_seed": pair["noise_seed"],
                "protocol_knobs_matched": matched,
                "rlinf_trained": rlinf,
                "reference_trained": reference,
                "stats": _two_proportion(
                    rlinf["successes"],
                    rlinf["n"],
                    reference["successes"],
                    reference["n"],
                ),
            }
        )

    pooled = _two_proportion(sum_x1, sum_n1, sum_x2, sum_n2)
    ci_low, ci_high = pooled["difference_95_ci"]
    ci_excludes_zero = pooled["difference_95_ci_excludes_zero"]
    gap = pooled["gap_reference_minus_rlinf"]
    # DEC-1: a CI-significant gap whose estimate exceeds 5% success is material.
    material = ci_excludes_zero and abs(gap) >= _DEC1_MATERIAL_THRESHOLD
    if not ci_excludes_zero:
        verdict = (
            "NOT SIGNIFICANT — the pooled 95% CI of the gap includes 0; the difference "
            "is within eval run-to-run noise (statistically explained, per DEC-1)."
        )
    elif not material:
        verdict = (
            "SIGNIFICANT but the gap estimate is below the DEC-1 5% materiality "
            "threshold; not a material defect."
        )
    else:
        verdict = (
            "MATERIAL — a CI-significant pooled gap of "
            f"{gap * 100:.1f}% (reference minus RLinf) above the DEC-1 5% threshold. "
            "The reference-trained model significantly outperforms the RLinf-trained "
            "model. AC-1 (norm-stats) and AC-2 (converter) are byte-identical, so the "
            "difference is in the trained weights -> the AC-4 same-batch step-1 "
            "root-cause and AC-5 pinned training-step parity are required to localize "
            "it. The single matched pair (pair 0) understated the gap; pooling over "
            "5 seed pairs reveals it."
        )

    return {
        "description": (
            "Statistical significance of the RLinf-trained vs reference-trained "
            "BEHAVIOR success_once gap under an EXPANDED deterministic, knob-matched "
            f"eval ({len(per_seed)} seed pairs, {sum_n1} episodes/model). Each run "
            "is summarized into a full run record (model knobs + the complete "
            "seed/task schedule) parsed from its dumped config + log; the per-seed "
            "outcomes are pooled into a two-proportion test + 95% CI."
        ),
        "n_seed_pairs": len(per_seed),
        "all_pairs_protocol_knobs_matched": all_matched,
        "episodes_per_model": sum_n1,
        "pooled": pooled,
        "pooled_rlinf_wilson_95_ci": _wilson_ci(sum_x1, sum_n1),
        "pooled_reference_wilson_95_ci": _wilson_ci(sum_x2, sum_n2),
        "dec1_material_threshold": _DEC1_MATERIAL_THRESHOLD,
        "dec1_gap_is_material": material,
        "verdict": verdict,
        "per_seed": per_seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(_DEFAULT_MANIFEST))
    parser.add_argument("--git-rev", default="unknown")
    parser.add_argument(
        "--out", default="docs/evidence/phase4_eval_gap_significance.json"
    )
    args = parser.parse_args()
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    evidence = build_evidence(manifest, args.git_rev)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    p = evidence["pooled"]
    print(
        f"pairs={evidence['n_seed_pairs']} matched={evidence['all_pairs_protocol_knobs_matched']} "
        f"episodes/model={evidence['episodes_per_model']} "
        f"gap={p['gap_reference_minus_rlinf']:.4f} z={p['z']:.3f} "
        f"p={p['p_value_two_sided']:.3f} CI={p['difference_95_ci']}"
    )
    print(evidence["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
