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

"""Conclusive comparison of the reference vs RLinf loader per-global-step frame MULTISET
using the VALUE-INDEPENDENT ``episode:frame`` id (byte-comparable across repos).

Compares the compact per-step global-set hash streams step-by-step (CONCLUSIVE
``IDENTICAL_MULTISET`` only if every step matches; otherwise ``DIVERGENT_MULTISET`` with
the first-mismatch step + concrete differing ids from the detail files). A comparability
guard (cross-dump distinct overlap) confirms the ids really are comparable; for the
value-independent id it must be > 0 (unlike the old exact-byte tensor hash).
"""

from __future__ import annotations

import json


def compare_id_audits(rlinf, ref, rlinf_detail=None, ref_detail=None, *, max_ids=12):
    rl_h = rlinf["per_step_global_set_hash"]
    rf_h = ref["per_step_global_set_hash"]
    n = min(len(rl_h), len(rf_h))
    matches = sum(1 for s in range(n) if rl_h[s] == rf_h[s])

    first_mismatch_step = next((s for s in range(n) if rl_h[s] != rf_h[s]), None)

    # comparability + first-mismatch ids from the (uncommitted) detail full sets
    comparable = None
    cross = None
    first_mismatch = None
    if rlinf_detail is not None and ref_detail is not None:
        rl_sets = rlinf_detail["per_step_global_set"]
        rf_sets = ref_detail["per_step_global_set"]
        rl_all = {f for s in rl_sets for f in s}
        rf_all = {f for s in rf_sets for f in s}
        cross = len(rl_all & rf_all)
        comparable = cross > 0
        if first_mismatch_step is not None:
            rlS = set(rl_sets[first_mismatch_step])
            rfS = set(rf_sets[first_mismatch_step])
            first_mismatch = {
                "step": first_mismatch_step,
                "rlinf_only_sample": sorted(rlS - rfS)[: max_ids // 2],
                "ref_only_sample": sorted(rfS - rlS)[: max_ids // 2],
            }
    elif first_mismatch_step is not None:
        first_mismatch = {"step": first_mismatch_step}

    if comparable is False:
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
        "verdict": verdict,
        "first_mismatch": first_mismatch,
        "rlinf": {
            "audited_steps": rlinf.get("audited_steps"),
            "num_workers": rlinf.get("num_workers"),
            "rolling_hash": rlinf.get("rolling_hash"),
            "min_distinct": min(rlinf["per_step_unique"]) if rlinf.get("per_step_unique") else None,
            "max_distinct": max(rlinf["per_step_unique"]) if rlinf.get("per_step_unique") else None,
        },
        "ref": {
            "audited_steps": ref.get("audited_steps"),
            "num_workers": ref.get("num_workers"),
            "rolling_hash": ref.get("rolling_hash"),
            "min_distinct": min(ref["per_step_unique"]) if ref.get("per_step_unique") else None,
            "max_distinct": max(ref["per_step_unique"]) if ref.get("per_step_unique") else None,
        },
    }


def _load(p):
    return json.loads(open(p).read())


def main(rlinf_path, ref_path, out_path, rlinf_detail=None, ref_detail=None):
    rlinf, ref = _load(rlinf_path), _load(ref_path)
    assert rlinf.get("ok") and ref.get("ok"), "one of the id audit dumps is not ok"
    rd = _load(rlinf_detail) if rlinf_detail else None
    fd = _load(ref_detail) if ref_detail else None
    result = compare_id_audits(rlinf, ref, rd, fd)
    with open(out_path, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("LOADER_AUDIT_COMPARE_IDS", json.dumps({
        "verdict": result["verdict"],
        "audited_steps": result["audited_steps"],
        "steps_matched": result["steps_matched"],
        "cross_overlap": result["cross_dump_distinct_overlap"],
        "first_mismatch_step": (result["first_mismatch"] or {}).get("step"),
    }))
    return result


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
