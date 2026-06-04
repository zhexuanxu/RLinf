# Exact per-sample stream-identity comparison (first-50-step, action windows + prompt)

Per-worker sha256 over the full first-50-step per-sample identity stream
(`global_frame_idx, episode_index, frame_index, chunk tuple, action_query_start/end,
action_is_pad, prompt, skip_count`), driven through the REAL dataset helpers
(`_select_streaming_chunk`, `_get_query_indices`, `_get_fine_grained_task`).

## RLinf made rank-independent  vs  reference (per worker)

| worker | RLinf-rank-independent sha256[:16] | reference sha256[:16] | match |
|--------|------------------------------------|-----------------------|-------|
| 0 | 270aec08ec6565bc | 270aec08ec6565bc | YES |
| 1 | a7d1b9f27b52677a | a7d1b9f27b52677a | YES |
| 2 | a7c63712b14d0321 | a7c63712b14d0321 | YES |
| 3 | c7220aa3fb86b772 | c7220aa3fb86b772 | YES |
| 4 | c5eb2ceaaee2db68 | c5eb2ceaaee2db68 | YES |
| 5 | fce5f27defe85d3c | fce5f27defe85d3c | YES |
| 6 | c63426db4c8088c5 | c63426db4c8088c5 | YES |
| 7 | ba1586e06fc97a42 | ba1586e06fc97a42 | YES |

**All 8 workers match exactly: True.** So RLinf with the rank-independent
partition (the temporary override used in the auditable 8-GPU rerun,
`r17_rank_independent_rerun.md`) emits the EXACT same first-50-step stream identity —
frames, keyframe chunks, action query windows, and prompts — as the reference loader.

## Production RLinf (rank-folding)  vs  reference

- Per-step unique-frame coverage: RLinf rank-folding = [256]
  vs reference (= RLinf rank-independent) = [32].
- Per-(rank0,worker0) stream sha256: rank-folding `dc2aafd4db2bd890` != reference `270aec08ec6565bc`
  — production RLinf folds the distributed rank into the partition, so its per-rank
  streams are disjoint (256 unique frames/step) and differ from the reference.

## Conclusion — data stream RULED OUT (exact identity)

Production RLinf and the reference stream DIFFER. But making RLinf's stream EXACTLY
equal to the reference's (rank-independent; per-worker sha256 match above, including
action windows + prompts) and re-running on 8 GPUs left the first-50 loss FLAT (12/50
within band; `r17_rank_independent_first50_losses.csv`). Therefore the data stream —
frame selection, ordering, coverage, action windows, and prompts — is RULED OUT as the
cause of the flat-vs-dropping divergence. The remaining cause is the FSDP
optimizer/gradient/weight-update path (next round).
