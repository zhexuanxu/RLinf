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

"""Generate the eval-gap significance evidence.

The reported success gap between the RLinf-trained and the reference-trained
BEHAVIOR checkpoints (`success_once`) comes from a stochastic, unpaired eval. This
parses each eval run's logged `success_once` + `num_trajectories` (so the numbers
are read from the runs, not hardcoded), then computes whether the gap is
statistically distinguishable from zero: a two-proportion z-test, the 95% CI of
the difference, and Wilson 95% CIs for each rate. Pure stdlib (no scipy).

Run from the repo root::

    python tests/unit_tests/_eval_gap_significance_dump.py \
        --out docs/evidence/phase4_eval_gap_significance.json
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re

_REPO = pathlib.Path("/mnt/public/xzxuan/repos/RLinf_pi05")
# The two eval runs: RLinf-trained converted checkpoint, and the reference-trained
# checkpoint through the SAME RLinf eval pipeline (the control direction).
_RLINF_EVAL_LOG = (
    _REPO
    / "logs/20260607-08:13:16-behavior_ppo_openpi_pi05_pytorch_eval/eval_embodiment.log"
)
_REFERENCE_EVAL_LOG = (
    _REPO
    / "logs/20260607-08:50:42-behavior_ppo_openpi_pi05_pytorch_eval2/eval_embodiment.log"
)
# DEC-1: the gap is material only if a CI-significant difference exceeds 5% success.
_DEC1_MATERIAL_THRESHOLD = 0.05
_Z_95 = 1.959963984540054

_SUCCESS_RE = re.compile(r"'eval/success_once': array\(([0-9.]+)")
_TRAJ_RE = re.compile(r"'eval/num_trajectories': (\d+)")


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def parse_eval_log(path: pathlib.Path) -> dict:
    """Read the final logged success_once + num_trajectories from an eval run."""
    text = path.read_text()
    successes = _SUCCESS_RE.findall(text)
    trajectories = _TRAJ_RE.findall(text)
    if not successes or not trajectories:
        raise RuntimeError(f"no success_once / num_trajectories in {path}")
    rate = float(successes[-1])
    n = int(trajectories[-1])
    x = round(rate * n)
    return {"log": str(path), "success_once": rate, "n": n, "successes": x}


def _wilson_ci(x: int, n: int, z: float = _Z_95) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = x / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return [center - half, center + half]


def build_evidence() -> dict:
    rlinf = parse_eval_log(_RLINF_EVAL_LOG)
    reference = parse_eval_log(_REFERENCE_EVAL_LOG)
    x1, n1 = rlinf["successes"], rlinf["n"]
    x2, n2 = reference["successes"], reference["n"]
    p1, p2 = x1 / n1, x2 / n2
    diff = p2 - p1  # reference minus RLinf (reported gap)

    # Two-proportion z-test (pooled) for H0: p1 == p2.
    p_pool = (x1 + x2) / (n1 + n2)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    # 95% CI of the difference (unpooled SE).
    se_unpooled = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    ci_low = diff - _Z_95 * se_unpooled
    ci_high = diff + _Z_95 * se_unpooled

    # DEC-1: material only if the difference is CI-significant (CI excludes 0) AND
    # the CI lower bound clears the 5% materiality threshold.
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

    return {
        "description": (
            "Statistical significance of the RLinf-trained vs reference-trained "
            "BEHAVIOR success_once gap. Unpaired, stochastic eval; values parsed "
            "from each run's logged success_once + num_trajectories."
        ),
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
    parser.add_argument(
        "--out", default="docs/evidence/phase4_eval_gap_significance.json"
    )
    args = parser.parse_args()
    evidence = build_evidence()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    print(
        f"gap={evidence['gap_reference_minus_rlinf']:.4f} "
        f"z={evidence['two_proportion_z_test']['z']:.3f} "
        f"p={evidence['two_proportion_z_test']['p_value_two_sided']:.3f} "
        f"CI={evidence['difference_95_ci']}"
    )
    print(evidence["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
