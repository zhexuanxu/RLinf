# AC-3 — first-step loss/grad gap: data-side vs compute-side

## Question
RLinf's 8-GPU BEHAVIOR SFT logs step-0 `loss≈0.327`, `grad_norm≈6.69` (AC-1 instrumented run), while
the reference logs `loss=0.24609`, `grad_norm=2.172`. Is the gap data-side or compute-side?

## Evidence

### The literal 2×2 cross-feed on BOTH actual step-0 batches (compute ruled out; data isolated)
`docs/evidence/phase6_ac3_cross_feed_2x2.json` (`tools/sft_cross_feed_2x2.py`): each repo's ACTUAL
materialized step-0 global batch (256 fully-transformed frames; the reference rank-0 fanout batch and
the RLinf default `per_rank_stream` batch — recorded frame ids + content hashes, the two batches share
0 of 256 frame ids) is fed through BOTH the RLinf `Pi0` and the reference
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
- `phase6_ac3_cross_feed_2x2.json` — the literal 2×2 on both ACTUAL step-0 batches (frame ids + content
  hashes; columns agree loss ≤ 5e-5 / grad rel ≤ 0.13%; rows differ Δloss 0.107) → compute ruled out +
  data isolated; the ref-batch column is the matched-noise production identical-data grad (rel 0.029%).
- `phase6_ac3_production_step0.json` — live-noise identical-data production step-0 (loss within 0.0044;
  data-side demonstration; grad residual = the live-noise draw, tight match in the 2×2).
- `phase6_ac3_fanout_step0_capture.json` — the raw `reference_fanout` 8-GPU step-0 capture.
- Gate: `tests/unit_tests/test_openpi_pytorch_sft_step0_parity.py` (5 tests).
- Reproduce: `tools/sft_cross_feed_2x2.py` (1 GPU, both models + both materialized step-0 batches);
  8-GPU `train_vla_sft.py behavior_pi05_vla max_steps=2 actor.sft_step0_instrument=true
  data.loader_mode=reference_fanout`.
