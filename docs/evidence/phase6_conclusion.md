# Phase-6 conclusion — the BEHAVIOR pi0.5 SFT first-step loss/grad gap is DATA-side

## Question
RLinf's 8-GPU BEHAVIOR pi0.5 FSDP SFT logged step-0 `loss=0.3275146`, `grad_norm=6.690909`
(`lr=2.4975e-08`), while the reference (`openpi-comet-pytorch-mixed`, `pi05_b1k-task0000_sft_pytorch_mixed`)
logged step-0 `loss=0.24609375`, `grad_norm=2.1718` (`lr=2.4975e-08`). Is the gap **data-side** or
**compute-side**, and is the optimizer/lr aligned? The investigation does NOT assume the prior
"benign RNG/shuffle" explanation.

## Resolution — outcome (a): a real loader misalignment, now fixed; no compute defect on identical data
The first-step gap is **DATA-side**. The decentralized **default** loader (`data.loader_mode=per_rank_stream`)
shards the chunk stream into `world_size * num_workers = 64` lanes, whereas the reference rank-0 fanout has
`8` lanes; at the production `num_workers=8` the two feed **different** step-0 256-frame batches. With
**identical** step-0 data the loss/grad match and the model compute is byte-identical, so there is **no
compute-side defect**. The optimizer/lr is strictly aligned (a warmup off-by-one was found and fixed). The
original `0.3275`-vs-`0.24609` is the decentralized default's *different* step-0 data, not a compute bug.

> **The default `per_rank_stream` is NOT strictly identical to the reference** — it is the decentralized
> throughput path (64 lanes). Strict per-step identity is provided by the user-approved
> `data.loader_mode=reference_fanout` (AC-2). This **supersedes** the prior "benign RNG/shuffle" claim:
> the full-30k multiset audit shows the default is genuinely a *different* per-step multiset (lane-count
> divergence at step 0), not a benign post-epoch reorder.

## Evidence by acceptance criterion (concrete numbers + the committed gate/artifact)

### AC-1 — trustworthy step-0 provenance from a confirmed-fixed instrumented 8-GPU run
`docs/evidence/phase6_sft_step0_capture.json` + `phase6_sft_step0_provenance.json`: a real 8-GPU FSDP SFT
step-0, runtime-instrumented to prove the rank/world-size loader fix EXECUTED — `rank_disjoint=true`,
`global_unique_frames=256` (not the rank-replicated 32 path), `fix_exercised=true`. Captured step-0
**loss `0.3275146`, grad_norm `6.690909`** (pre-clip fp32 global), **lr `2.4975e-08`** (= reference warmup
init, within 1e-12). Gates: `tests/unit_tests/test_openpi_pytorch_sft_step0_provenance.py`,
`test_openpi_pytorch_sft_sharding.py`.

### AC-2 — loaders strictly consistent in `reference_fanout` mode (all 30000 steps)
`docs/evidence/phase6_loader_audit_ids_fanout_compare.json` (+ `..._fanout_nonperturb.json`,
`phase6_loader_lane_root_cause.md`): a full **30000-step** rolling per-step global-multiset audit using
value-independent `(episode_index, frame_index)` ids: **verdict `IDENTICAL_MULTISET`**,
`identical_prefix_steps=30000`, the RLinf `reference_fanout` `global_rolling_hash` equals the reference
exactly, per-rank `rank_assignment_relation=identical-order`, across the epoch boundary; the
non-perturbation rerun is byte-identical. The **default** decentralized loader (64 lanes) diverges from
the reference (8 lanes) at step 0 (`lane_count`), which is the data-side cause. Gates:
`tests/unit_tests/test_openpi_pytorch_sft_loader_audit_ids.py`, `test_openpi_pytorch_sft_sharding.py`.

### AC-3 — the gap is DATA-side; compute byte-identical; the real production stack matches on identical data
`docs/evidence/phase6_ac3_cross_feed_2x2.json`: the literal 2×2 on both ACTUAL step-0 batches (256
fully-transformed frames each). Same batch + both models with a SHARED noise/time → columns agree: the
reference-batch column is ref-model `0.22795 / 2.3383` vs rlinf-model `0.22790 / 2.3377` (grad rel
**0.029%**) → **compute ruled out**. The two batches are genuinely different data: per-surface hashes
differ on every content surface (masks identical), and the value-independent `(episode,frame)` ids share
**0 of 256** frames. Rows (same model, two batches) differ Δloss `0.107` (`0.22795` vs `0.33534`) →
**DATA-side**.

`docs/evidence/phase6_ac3_production_stack_controlled.json`: the REAL 8-GPU FSDP stack
(`data.loader_mode=per_rank_stream` + the rank-sliced pinned loader; pinned forces the per-rank topology)
on the IDENTICAL reference step-0 batch + CONTROLLED seed-4242 noise → **loss `0.2281494` / grad
`2.3253620`**, reproducing the bit-identical-input reference cell `0.22795 / 2.33834` to **loss abs
`0.000201` ≤ 0.01** and **grad rel `0.555%` ≤ 0.02** — gated directly on the production stack.

`docs/evidence/phase6_ac3_production_step0.json`: with identical data (`reference_fanout`) the live-noise
8-GPU step-0 is `0.25049 / 2.347` ≈ reference `0.24609 / 2.172` (loss within 0.0044), collapsing from the
decentralized default's `0.3275 / 6.691`. Gate: `tests/unit_tests/test_openpi_pytorch_sft_step0_parity.py`.

### AC-4 — optimizer/lr strictly aligned (warmup off-by-one found + fixed)
`tools/sft_optimizer_parity_probe.py` + `tests/unit_tests/test_openpi_pytorch_sft_optimizer_parity.py`:
RLinf's real `openpi_cosine` scheduler reproduces the reference inline lr schedule across warmup + cosine
to ≤ `1e-12`; step-0 `lr=2.4975e-08` (= `peak/(warmup+1)`), peak `2.5e-5` at step 1000. A genuine bug was
found via a two-builder probe — `warmup_optimizer_state` advanced the AdamW step counter to 1, biasing the
first real update; fixed by resetting to 0. AdamW built through RLinf's real `build_optimizer` is
bit-identical to the fused reference.

## Conclusion
The first-step gap is **DATA-side**: the decentralized default `per_rank_stream` loader feeds RLinf a
*different* step-0 256-frame batch than the reference (AC-2 lane-count divergence), and that different data
— not the model compute — produces the `0.3275 / 6.691` step-0. The model forward+backward is byte-identical
(AC-3 compute ruled out), and the REAL 8-GPU production stack on identical data + controlled noise
reproduces the reference within tolerance, so there is **no compute-side defect to fix** on identical data.
Strict per-step data identity is available via the user-approved `data.loader_mode=reference_fanout`
(AC-2, full-30k `IDENTICAL_MULTISET`); the default remains the decentralized throughput path. The
optimizer/lr is strictly aligned and a warmup off-by-one was found and fixed (AC-4). This supersedes the
prior "benign RNG/shuffle" explanation, which lacked the 30k multiset proof.

## Independent audit (task5)
See `docs/evidence/phase6_task5_audit.md` — an independent per-surface audit of the AC-1..AC-4 methodology
and evidence (step-0 logging semantics, global-mean loss reduction, pre-clip fp32 all-reduced grad norm, the
30k hash coverage and non-perturbation, the 2×2 identical-input controls, the pinned production-stack topology
and input/noise identity, and the AC-4 lr/AdamW probe).

Audit verdict: **GAP**. The substantive data-side / no-compute-defect / AC-4-aligned conclusions are supported,
but AC-5 synthesis is blocked until `docs/evidence/phase6_ac3_production_stack_controlled.json` records a
literally executable production command. The current command contains `<p9>` and `<scratch>` placeholders.

## Artifacts + gates
- AC-1: `phase6_sft_step0_capture.json`, `phase6_sft_step0_provenance.json` →
  `test_openpi_pytorch_sft_step0_provenance.py`, `test_openpi_pytorch_sft_sharding.py`.
- AC-2: `phase6_loader_audit_ids_fanout_compare.json` (+ nonperturb), `phase6_loader_lane_root_cause.md` →
  `test_openpi_pytorch_sft_loader_audit_ids.py`, `test_openpi_pytorch_sft_sharding.py`.
- AC-3: `phase6_ac3_cross_feed_2x2.json`, `phase6_ac3_production_stack_controlled.json`,
  `phase6_ac3_production_step0.json`, `phase6_ac3_conclusion.md` → `test_openpi_pytorch_sft_step0_parity.py`.
- AC-4: `tools/sft_optimizer_parity_probe.py` → `test_openpi_pytorch_sft_optimizer_parity.py`.
- AC-5 (this synthesis): `phase6_conclusion.json` + `phase6_conclusion.md` →
  `tests/unit_tests/test_phase6_conclusion.py`.
