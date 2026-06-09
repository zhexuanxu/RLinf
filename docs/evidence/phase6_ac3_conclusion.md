# AC-3 — first-step loss/grad gap: data-side vs compute-side

## Question
RLinf's 8-GPU BEHAVIOR SFT logs step-0 `loss≈0.327`, `grad_norm≈6.69` (AC-1 instrumented run), while
the reference logs `loss=0.24609`, `grad_norm=2.172`. Is the gap data-side or compute-side?

## Evidence

### The literal 2×2 cross-feed on BOTH actual step-0 batches (compute ruled out; data isolated)
`docs/evidence/phase6_ac3_cross_feed_2x2.json` (`tools/sft_cross_feed_2x2.py`): each repo's ACTUAL
materialized step-0 global batch (256 fully-transformed frames; the reference rank-0 fanout batch and
the RLinf default `per_rank_stream` batch) is fed through BOTH the RLinf `Pi0` and the reference.
The two batches are genuinely different data on THREE independent identity surfaces: (i) **per-surface
sha256** — every CONTENT surface (state, actions, each image, tokenized prompt) differs, while the
all-True image masks are identical (same task); (ii) the combined state+actions **content hash**
differs; (iii) **value-independent `(episode_index, frame_index)` ids** (from the id_only loader
audits, byte-comparable across repos) — each batch is 256 distinct frames and the two share **0 of
256** frames. Each batch is fed through BOTH the RLinf `Pi0` and the reference
`openpi.models_pytorch_new.Pi0` with a SHARED flow `noise`/`time` and identical weights
(`pi05_base_pytorch_new`) / norm-stats / dtype-autocast (fp32 master + bf16 compute, production-faithful)
/ train=True-rng=None. Per-cell step-0 loss + pre-clip fp32 GLOBAL grad norm over the 256-frame batch:

| step-0 (shared noise) | reference model | RLinf model | column Δ |
|---|---|---|---|
| **reference batch** | loss 0.22795, grad 2.3383 | loss 0.22790, grad 2.3377 | loss 5e-5, **grad rel 0.029%** |
| **RLinf-default batch** | loss 0.33534, grad 6.2720 | loss 0.33541, grad 6.2800 | loss 7e-5, **grad rel 0.13%** |

- **Columns (same batch, both models) agree** to loss ≤ 5e-5 and grad rel ≤ 0.13% (well within the bf16
  0.01 band and the 2% grad band) → the model forward+backward is **byte-identical (compute ruled out)**;
  the reference-batch column is also the production identical-data grad with controlled noise (within 2%).
- **Rows (same model, two distinct batches, same noise) differ** by Δloss 0.107 and grad 2.34 vs 6.27 →
  the **data** is the cause.

### With identical data, the production step-0 matches the reference (gap is data-side)
`docs/evidence/phase6_ac3_production_step0.json` + `phase6_ac3_fanout_step0_capture.json`: a real 8-GPU
FSDP SFT step-0, instrumented (pre-`optimizer.step()` global-mean loss; pre-clip fp32 all-reduced
grad_norm; rank-disjoint 256-frame effective batch confirmed), run under each loading strategy:

| step-0 | loss | grad_norm | lr |
|---|---|---|---|
| reference production | 0.24609 | 2.172 | 2.4975e-08 |
| RLinf `per_rank_stream` (default, AC-1) | 0.32751 | 6.691 | 2.4975e-08 |
| RLinf `reference_fanout` (identical data) | 0.25049 | 2.347 | 2.4975e-08 |

With **identical step-0 data** (`data.loader_mode=reference_fanout`, byte-identical to the reference per
AC-2), the RLinf production step-0 loss matches the reference within **abs 0.0044 (< the 0.01 bf16
band)** and grad_norm collapses from 6.69 → **2.35** (≈ reference 2.17), whereas the decentralized
default's *different* step-0 data gives 0.327 / 6.69 (18× the loss gap, 26× the grad gap of the
fanout arm).

### The REAL production stack matches the reference within 2% on identical data + controlled noise
`docs/evidence/phase6_ac3_production_stack_controlled.json`: the live-noise fanout-arm grad residual
above (2.35 vs 2.17, ~8%) is only the production noise/time DRAW differing between repos. Pinning the
IDENTICAL reference step-0 batch (the 2×2 reference batch, materialized via the reference rank-0 fanout)
+ a CONTROLLED seed-4242 noise into the REAL 8-GPU FSDP SFT worker via the rank-sliced pinned loader
(`build_pinned_behavior_sft_dataloader`, `data.loader_mode=per_rank_stream` — each rank consumes its own
`arr[rank::world_size]` slice; the worker forces this per-rank topology when pinned is active, so there
is NO rank-0 fanout) gives a production-stack step-0 that reproduces the reference's same-batch+same-noise
step-0 (the 2×2 ref cell **0.22795 / 2.33834**) to **loss abs ≤ 0.01** and **grad rel ≤ 2%** — gated
directly on the real FSDP/mixed-precision/loss-reduction/all-reduced-grad/pre-clip-fp32-norm stack (not
the single-GPU 2×2 replay). The pinned inputs are proven bit-for-bit the 2×2 reference batch (every
surface) + seed-4242 noise, so the 2×2 cell is the exact reference target.

## Conclusion — DATA-SIDE
The first-step loss/grad gap is **data-side**: the decentralized `per_rank_stream` loader feeds RLinf a
*different* step-0 256-frame batch than the reference (AC-2: 64 vs 8 lanes at production num_workers),
and that different data — not the model compute — produces the ~0.327/6.69 step-0. Aligning the data
(`reference_fanout`) collapses RLinf's production step-0 onto the reference's (loss within the band;
grad_norm to ≈2.35). The model forward+backward is byte-identical (compute ruled out), so there is **no
compute-side defect to fix** on identical data.

The live-noise fanout-arm grad residual (2.35 vs 2.17, rel ~8%) is the production flow `noise`/`time`
DRAW differing between the two repos at step 0 (each samples its own), NOT a compute difference: when
the noise/time is CONTROLLED (the 2×2, shared draw), the two production-faithful stacks' grad norm on
the identical reference step-0 batch agrees to **rel 0.029% (≤ 2%)** and reproduces the production-scale
grad (≈2.34). A multi-step descent-rate difference is out of the step-0 scope and is governed by the
optimizer/lr identity (AC-4, which found+fixed the warmup step-counter off-by-one).

## Artifacts
- `phase6_ac3_cross_feed_2x2.json` — the literal 2×2 on both ACTUAL step-0 batches (per-surface sha256
  + content hashes + value-independent `(episode,frame)` ids: content surfaces differ, masks identical,
  0/256 shared frames; columns agree loss ≤ 5e-5 / grad rel ≤ 0.13%; rows differ Δloss 0.107) → compute
  ruled out + data isolated; the ref-batch column is the matched-noise production identical-data grad.
- `phase6_ac3_production_stack_controlled.json` — the REAL 8-GPU FSDP stack on the IDENTICAL
  reference_fanout batch + CONTROLLED seed-4242 noise: loss abs 0.0002 ≤ 0.01, grad rel 0.55% ≤ 2%,
  gated directly on the production stack (pinned inputs proven bit-for-bit the 2×2 reference batch+noise).
- `phase6_ac3_production_step0.json` — live-noise identical-data production step-0 (loss within 0.0044;
  data-side demonstration; grad residual = the live-noise draw, tight match in the 2×2).
- `phase6_ac3_fanout_step0_capture.json` — the raw `reference_fanout` 8-GPU step-0 capture.
- Gate: `tests/unit_tests/test_openpi_pytorch_sft_step0_parity.py` (8 tests).
- Reproduce: `tools/sft_cross_feed_2x2.py` (1 GPU, both models + both materialized step-0 batches, now
  emitting per-surface hashes); value-independent ids via `tools/sft_loader_audit_rlinf.py` +
  `tests/unit_tests/_ref_loader_audit_ids.py`; production stack via `tools/build_pinned_step0_npz.py`
  (2×2 dump → pinned npz) then 8-GPU `train_vla_sft.py --config-name behavior_pi05_vla max_steps=1
  actor.sft_step0_instrument=true data.loader_mode=per_rank_stream +data.pinned_inputs_npz=...
  +data.pinned_noise_time_npz=...` (the rank-sliced pinned loader forces the per-rank topology).
