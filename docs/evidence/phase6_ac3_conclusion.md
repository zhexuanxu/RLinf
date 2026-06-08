# AC-3 — first-step loss/grad gap: data-side vs compute-side

## Question
RLinf's 8-GPU BEHAVIOR SFT logs step-0 `loss≈0.327`, `grad_norm≈6.69` (AC-1 instrumented run), while
the reference logs `loss=0.24609`, `grad_norm=2.172`. Is the gap data-side or compute-side?

## Evidence

### Compute is identical (model ruled out)
`docs/evidence/phase6_ac3_cross_feed.json` (`tools/sft_grad_parity_probe.py`): the SAME batch + SAME
weights (`pi05_base_pytorch_new`) + SAME flow `noise`/`time` + `train=True`/`rng=None` is fed through
BOTH the RLinf `Pi0` and the reference `openpi.models_pytorch_new.Pi0`. Holding data/weights/noise/
time/dtype/train-mode identical isolates COMPUTE:
- per-batch loss agrees to **|Δ| ≤ 0.0005** (well within the bf16 DEC-1 band of 0.01);
- global grad norm agrees to **rel ≤ 0.28%**; per-module grad norms and #params-with-grad also match.

RLinf vendored the reference forward+backward byte-for-byte, so the model **compute is ruled out** as
the cause of the first-step gap. (This is the cross-feed's compute-isolation: same data → same loss on
both models.)

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

The small remaining grad residual on the fanout arm (2.35 vs 2.17, rel ~8%) is the production flow
`noise`/`time` DRAW differing between the two repos at step 0 (each samples its own), NOT a compute
difference — confirmed by the cross-feed, where matching the noise/time makes the grad agree to 0.28%.
A multi-step descent-rate difference is out of AC-3's step-0 scope and is governed by the optimizer/lr
identity (AC-4, which found+fixed the warmup step-counter off-by-one).

## Artifacts
- `phase6_ac3_cross_feed.json` — same-batch+noise compute parity (loss ≤ 0.0005, grad rel ≤ 0.28%).
- `phase6_ac3_production_step0.json` — identical-data production step-0 (loss within 0.0044; data-side).
- `phase6_ac3_fanout_step0_capture.json` — the raw `reference_fanout` 8-GPU step-0 capture.
- Gate: `tests/unit_tests/test_openpi_pytorch_sft_step0_parity.py` (3 tests).
- Reproduce: `tools/sft_grad_parity_probe.py` (1 GPU); 8-GPU `train_vla_sft.py behavior_pi05_vla
  max_steps=2 actor.sft_step0_instrument=true data.loader_mode=reference_fanout`.
