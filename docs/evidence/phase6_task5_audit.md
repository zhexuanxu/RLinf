# Phase-6 task5 independent audit

Verdict: GAP

Scope: independent audit of the committed AC-1..AC-4 methodology and evidence before final AC-5 synthesis. The audit stayed on the named artifacts/code surfaces from the task request.

## Surface verdicts

SURFACE 1: PASS -- RLinf records step-0 loss/grad/lr from production metrics after the forward/backward path but before scheduler advance; loss is rank-averaged global mean and capture proves 8x32 rank-disjoint batch, `docs/evidence/phase6_sft_step0_capture.json:24`, `rlinf/workers/sft/fsdp_sft_worker.py:261`.

SURFACE 2: PASS -- `_prod_loss_grad` computes per-micro mean loss and backprops `loss_i / n_micro`, matching the 256-frame global-mean gradient, `tools/sft_cross_feed_2x2.py:170`, `tests/unit_tests/_ref_step0_2x2_dump.py:136`.

SURFACE 3: PASS -- reported grad norm is pre-clip fp32 global: proxy copies grads to fp32 master then `clip_grad_norm_` returns pre-clip norm, and production FSDP all-reduces fp32 norm before clipping, `tools/sft_cross_feed_2x2.py:193`, `rlinf/hybrid_engines/fsdp/strategy/fsdp.py:405`.

SURFACE 4: PASS -- full 30k fanout audit is `IDENTICAL_MULTISET`, both sides cover 30000 steps with 256 distinct frames and matching rolling hashes; non-perturb rerun is byte-identical, `docs/evidence/phase6_loader_audit_ids_fanout_compare.json:3`, `docs/evidence/phase6_loader_audit_ids_fanout_nonperturb.json:2`.

SURFACE 5: PASS -- 2x2 cells are identical-input per column with shared weights/noise/time/dtype/train-mode; columns match tightly while rows differ materially, supporting DATA_SIDE classification, `docs/evidence/phase6_ac3_cross_feed_2x2.json:526`.

SURFACE 6: GAP -- topology/gate are sound, but the recorded production command is not literally executable because it contains placeholders `<p9>` and `<scratch>`, `docs/evidence/phase6_ac3_production_stack_controlled.json:13`.

Concrete fix: commit the resolved absolute command actually run for `phase6_ac3_production_stack_controlled.json`, or add a separate `resolved_command` field while keeping the template command.

SURFACE 7: PASS -- pinned input/noise identity is supported: all surfaces match the 2x2 reference batch, noise/time are bit-identical, pinned loader rank-slices `arr[rank::world_size]`, and worker forces per-rank topology when pinned, `docs/evidence/phase6_ac3_production_stack_controlled.json:15`, `rlinf/data/datasets/behavior/behavior_pinned_loader.py:52`, `rlinf/workers/sft/fsdp_sft_worker.py:81`.

SURFACE 8: PASS -- AC-4 uses RLinf's real `FSDPModelManager.build_optimizer`, exercises fused reference AdamW, gates lr to <=1e-12, and verifies the warmup step counter reset to 0, `tools/sft_optimizer_parity_probe.py:97`, `tests/unit_tests/test_openpi_pytorch_sft_optimizer_parity.py:62`, `tests/unit_tests/test_openpi_pytorch_sft_optimizer_parity.py:110`.

## Overall

The data-side conclusion, no compute-side defect on identical data, and lr/AdamW alignment are substantively supported.

Blocker before AC-5: fix the AC-3 production-stack artifact command metadata in `docs/evidence/phase6_ac3_production_stack_controlled.json` so the committed evidence is directly reproducible.
