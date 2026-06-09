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

"""Compare the reference rank-0-fanout loader and RLinf 8-rank streaming loader on the
per-global-step frame-id MULTISET.

Both dumps carry ``per_rank_per_step_frame_hashes`` (per step, a list of per-rank ordered
frame-id lists; same sha256-over-state+actions+tokenized_prompt identity). For each global
step this computes the global frame SET on each side, compares the sorted-set hashes,
records the distinct count and the Jaccard overlap, and reports the FIRST step whose global
sets differ (with a sample of the symmetric-difference ids). The verdict is IDENTICAL only
if every audited step matches.
"""

from __future__ import annotations

import hashlib
import json


def _step_global_set(step_rank_hashes):
    return {f for rh in step_rank_hashes for f in rh}


def _set_hash(s):
    return hashlib.sha256("".join(sorted(s)).encode()).hexdigest()


def compare_audits(rlinf: dict, ref: dict, *, max_diff_ids: int = 12) -> dict:
    """Compare two loader-audit dumps step-by-step. Returns a verdict dict; the global
    sets must be identical at EVERY audited step for an IDENTICAL verdict."""
    rl_steps = rlinf["per_rank_per_step_frame_hashes"]
    rf_steps = ref["per_rank_per_step_frame_hashes"]
    n = min(len(rl_steps), len(rf_steps))
    per_step = []
    first_mismatch = None
    for s in range(n):
        rl_set = _step_global_set(rl_steps[s])
        rf_set = _step_global_set(rf_steps[s])
        match = _set_hash(rl_set) == _set_hash(rf_set)
        inter = rl_set & rf_set
        union = rl_set | rf_set
        jacc = (len(inter) / len(union)) if union else 1.0
        per_step.append(
            {
                "step": s,
                "rlinf_distinct": len(rl_set),
                "ref_distinct": len(rf_set),
                "match": match,
                "jaccard": jacc,
            }
        )
        if not match and first_mismatch is None:
            diff = sorted(rl_set ^ rf_set)[:max_diff_ids]
            first_mismatch = {
                "step": s,
                "rlinf_only_sample": sorted(rl_set - rf_set)[: max_diff_ids // 2],
                "ref_only_sample": sorted(rf_set - rl_set)[: max_diff_ids // 2],
                "symmetric_difference_sample": diff,
            }
    matches = sum(1 for p in per_step if p["match"])
    mean_jacc = sum(p["jaccard"] for p in per_step) / len(per_step) if per_step else 0.0

    # Cross-dump comparability guard: if the frame IDENTITY were byte-comparable across
    # the two loaders, the union of all frames each dumped (over many steps, from a
    # finite dataset pool) would overlap by chance. EXACTLY ZERO cross-dump overlap means
    # the same underlying frame hashes DIFFERENTLY between repos (float-level
    # normalization differences) -> the per-step multiset cannot be judged via this
    # identity, and a low per-step overlap is a hash artifact, not proof of divergence.
    rl_all = {f for step in rl_steps for rh in step for f in rh}
    rf_all = {f for step in rf_steps for rh in step for f in rh}
    cross = len(rl_all & rf_all)
    comparable = cross > 0
    if not comparable:
        verdict = "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE"
    elif matches == n and n > 0:
        verdict = "IDENTICAL_MULTISET"
    else:
        verdict = "DIVERGENT_MULTISET"
    return {
        "schema_version": 1,
        "audited_steps": n,
        "steps_matched": matches,
        "all_steps_identical": matches == n and n > 0,
        "frame_identity_comparable": comparable,
        "cross_dump_distinct_overlap": cross,
        "rlinf_total_distinct": len(rl_all),
        "ref_total_distinct": len(rf_all),
        "verdict": verdict,
        "first_mismatch": first_mismatch,
        "mean_jaccard_overlap": mean_jacc,
        "rlinf_distinct_per_step": [p["rlinf_distinct"] for p in per_step],
        "ref_distinct_per_step": [p["ref_distinct"] for p in per_step],
        "per_step": per_step,
        "sources": {
            "rlinf": {k: rlinf.get(k) for k in ("seed", "num_workers", "world_size")},
            "ref": {k: ref.get(k) for k in ("config", "world_size", "num_workers")},
        },
    }


def main(rlinf_path, ref_path, out_path):
    rlinf = json.loads(open(rlinf_path).read())
    ref = json.loads(open(ref_path).read())
    assert rlinf.get("ok") and ref.get("ok"), "one of the audit dumps is not ok"
    result = compare_audits(rlinf, ref)
    with open(out_path, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("LOADER_AUDIT_COMPARE", json.dumps({
        "verdict": result["verdict"],
        "audited_steps": result["audited_steps"],
        "steps_matched": result["steps_matched"],
        "mean_jaccard": round(result["mean_jaccard_overlap"], 4),
        "first_mismatch_step": (result["first_mismatch"] or {}).get("step"),
    }))
    return result


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2], sys.argv[3])
