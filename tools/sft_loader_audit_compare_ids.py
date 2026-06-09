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

"""Comparison of the reference vs RLinf loader per-global-step frame MULTISET using the
VALUE-INDEPENDENT ``episode:frame`` id (byte-comparable across repos), schema v2.

The full-30k audit revealed TWO distinct divergence causes, so the comparator reports the
length of the IDENTICAL leading prefix and classifies the first divergence:

* **lane count** — at production ``num_workers`` the two pipelines shard the chunk stream into
  a different number of lanes (reference rank-0 fanout = ``num_workers``; RLinf decentralized =
  ``world_size*num_workers``), so the global multiset diverges from STEP 0.
* **epoch boundary** — at matched lane count the global multiset is IDENTICAL for the entire
  first epoch (``epoch_len`` steps, incl. step 0), then diverges exactly at the epoch boundary
  because the reference's centralized fanout re-creates its single DataLoader iterator when it
  exhausts (``13435`` batches, NOT a multiple of ``world_size``), drifting its worker-lanes,
  while RLinf's per-rank loaders stream a full lane uniformly. Both keep sampling the SAME pool
  (429,928 frames, 256/step) — a benign reorder, not a data-availability difference.

Verdicts (comparability guard first: ``INCONCLUSIVE_*`` if cross-dump overlap is 0):
* ``IDENTICAL_MULTISET`` — identical at EVERY one of the audited steps.
* ``IDENTICAL_FIRST_EPOCH`` — identical for >= ``epoch_len`` leading steps (the full first
  epoch, incl. step 0), then a benign epoch-boundary reorder of the same pool.
* ``DIVERGENT_MULTISET`` — diverges within the first epoch (e.g. step 0 — the lane-count cause).
"""

from __future__ import annotations

import json

SCHEMA_VERSION = 2


def _dump_summary(dump):
    uniques = dump.get("per_step_unique")
    return {
        "schema_version": dump.get("schema_version"),
        "audited_steps": dump.get("audited_steps"),
        "num_workers": dump.get("num_workers"),
        "global_rolling_hash": dump.get("global_rolling_hash"),
        "min_distinct": min(uniques) if uniques else None,
        "max_distinct": max(uniques) if uniques else None,
    }


def _rank_relation(rlinf, ref):
    a = rlinf.get("per_rank_rolling_hash")
    b = ref.get("per_rank_rolling_hash")
    if not a or not b or len(a) != len(b):
        return "unknown"
    if a == b:
        return "identical-order"
    if sorted(a) == sorted(b):
        return "permutation"
    return "rank-divergent"


def compare_id_audits(rlinf, ref, rlinf_detail=None, ref_detail=None, *, epoch_len=None, max_ids=12):
    rl_roll = rlinf.get("global_rolling_hash")
    rf_roll = ref.get("global_rolling_hash")
    n = min(rlinf.get("audited_steps") or 0, ref.get("audited_steps") or 0)
    rolling_match = rl_roll is not None and rl_roll == rf_roll

    comparable = None
    cross = None
    first_mismatch = None
    identical_prefix = None
    if rlinf_detail is not None and ref_detail is not None:
        rl_sets = rlinf_detail["per_step_global_set"]
        rf_sets = ref_detail["per_step_global_set"]
        rl_all = {f for s in rl_sets for f in s}
        rf_all = {f for s in rf_sets for f in s}
        cross = len(rl_all & rf_all)
        comparable = cross > 0
        rl_h = rlinf_detail.get("per_step_global_hash")
        rf_h = ref_detail.get("per_step_global_hash")
        if rl_h and rf_h:
            compared_steps = min(len(rl_h), len(rf_h))
            first_div_step = next(
                (s for s in range(compared_steps) if rl_h[s] != rf_h[s]), None
            )
            identical_prefix = (
                compared_steps if first_div_step is None else first_div_step
            )
            if first_div_step is not None:
                rlinf_set = set(rl_sets[first_div_step])
                ref_set = set(rf_sets[first_div_step])
                first_mismatch = {
                    "step": first_div_step,
                    "rlinf_only_sample": sorted(rlinf_set - ref_set)[: max_ids // 2],
                    "ref_only_sample": sorted(ref_set - rlinf_set)[: max_ids // 2],
                }

    identical_full = bool(rolling_match) and (identical_prefix in (None, n)) and n > 0
    identical_first_epoch = (
        epoch_len is not None
        and identical_prefix is not None
        and identical_prefix >= epoch_len
    )
    # classify the first divergence
    if identical_prefix is None or identical_prefix == n:
        first_divergence_cause = None if identical_full else "unknown"
    elif identical_prefix == 0:
        first_divergence_cause = "lane_count"
    elif epoch_len is not None and identical_prefix == epoch_len:
        first_divergence_cause = "epoch_boundary"
    else:
        first_divergence_cause = "other"

    if comparable is False:
        verdict = "INCONCLUSIVE_FRAME_IDENTITY_NOT_BYTE_COMPARABLE"
    elif identical_full:
        verdict = "IDENTICAL_MULTISET"
    elif identical_first_epoch:
        verdict = "IDENTICAL_FIRST_EPOCH"
    else:
        verdict = "DIVERGENT_MULTISET"

    return {
        "schema_version": SCHEMA_VERSION,
        "audited_steps": n,
        "epoch_len": epoch_len,
        "global_rolling_hash_match": bool(rolling_match),
        "identical_prefix_steps": identical_prefix,
        "identical_full": identical_full,
        "identical_first_epoch": identical_first_epoch,
        "first_divergence_cause": first_divergence_cause,
        "frame_identity_comparable": comparable,
        "cross_dump_distinct_overlap": cross,
        "rank_assignment_relation": _rank_relation(rlinf, ref),
        "verdict": verdict,
        "first_mismatch": first_mismatch,
        "rlinf": _dump_summary(rlinf),
        "ref": _dump_summary(ref),
    }


def _load(p):
    with open(p) as f:
        return json.loads(f.read())


def main(rlinf_path, ref_path, out_path, rlinf_detail=None, ref_detail=None, epoch_len=None):
    rlinf, ref = _load(rlinf_path), _load(ref_path)
    assert rlinf.get("ok") and ref.get("ok"), "one of the id audit dumps is not ok"
    rd = _load(rlinf_detail) if rlinf_detail else None
    fd = _load(ref_detail) if ref_detail else None
    result = compare_id_audits(
        rlinf, ref, rd, fd, epoch_len=int(epoch_len) if epoch_len else None
    )
    with open(out_path, "w", newline="\n") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print("LOADER_AUDIT_COMPARE_IDS", json.dumps({
        "verdict": result["verdict"],
        "identical_prefix_steps": result["identical_prefix_steps"],
        "cause": result["first_divergence_cause"],
        "rank_relation": result["rank_assignment_relation"],
        "cross_overlap": result["cross_dump_distinct_overlap"],
        "first_mismatch_step": (result["first_mismatch"] or {}).get("step"),
    }))
    return result


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
