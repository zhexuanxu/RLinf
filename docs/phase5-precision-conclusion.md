# Phase-5 — precision-alignment conclusion (AC-2 verdict + AC-5 null result)

**Conclusion: the RLinf BEHAVIOR pi0.5 FSDP SFT mixed-precision recipe is VERIFIED EFFECTIVELY IDENTICAL to
the reference on every runtime dtype surface. Precision is therefore NOT the source of the residual
reference-minus-RLinf eval-success gap measured in the prior phase. The one real config issue — the
`param_dtype: ${actor.model.precision}` coupling footgun — has been DECOUPLED (set explicitly to `bf16`). The
draft's premise ("definitely not aligned") was a misreading of an overloaded config field; its literal
`precision: fp32` edit would have disabled bf16 compute and was correctly not applied.**

## AC-2 — per-surface effective-equivalence verdict

From the committed dual-repo runtime ledgers (`docs/evidence/phase5_precision_ledger_{rlinf,ref}_rank{0,1}.json`),
the per-surface verdict (`docs/evidence/phase5_precision_surface_verdict.json`, gate
`test_openpi_pytorch_precision_verdict.py`) is, on rank 0 AND a sharded rank:

- **PASS (effectively identical), 44 surface-rows:** FSDP `param_dtype` bf16; `reduce_dtype`/gradient-reduction
  fp32; MP flags; master fp32; during-forward compute bf16; runtime autocast off; loss/output/action/noise/time
  dtype; grad fp32; grad-norm fp32 accumulation dtype; optimizer-state fp32; saved + load-after-save fp32 (all
  667 tensors); buffer_count 0; buffer-default probe (omitted → fp32); EMA none.
- **BENIGN (documented mechanism difference), 8 surface-rows:** `fsdp_buffer_dtype` fp32-explicit vs None
  (MOOT — 0 buffers either side, FSDP leaves an omitted buffer at fp32 per the probe); `grad_scaler` disabled
  ShardedGradScaler vs none (both apply no scaling); `optimizer` non-fused vs fused (both fp32 state; identical
  betas/eps/wd; `lr` differs only by warmup-schedule state); `grad_norm` value differs (independent weights;
  same fp32 dtype).
- **CONTEXT, 2 rows:** reference-only `pytorch_training_precision` (`mp_bfloat16` → fp32 load).
- **MISMATCH: 0.** No surface diverges in effective dtype.

## AC-4 — the config footgun decoupled

`examples/sft/config/behavior_pi05_vla.yaml` now sets `fsdp_config.mixed_precision.param_dtype: bf16`
explicitly, no longer interpolated from `${actor.model.precision}`. The dry-run resolution gate
(`test_openpi_pytorch_precision_decouple.py`) proves overriding `actor.model.precision=fp32` leaves the FSDP
compute `param_dtype` at `bf16` (decoupled) while `load_for_training` stays True (fp32 master). The AC-1 ledger
re-run after the edit shows `fsdp_param_dtype` unchanged (bf16) and master unchanged (fp32) — the effective
recipe is preserved.

## AC-5 — verified-aligned null result

Every same-input precision surface is now proven aligned by RUNTIME observation on the same pinned input
(sha-proven identical), across both ranks. So the significant residual eval gap (Phase-4 AC-3, +9.2%) is **not**
a mixed-precision misconfiguration. Combined with Phase-4 (norm-stats byte-identical, converter value-lossless,
same-batch raw→loss match, pinned training-step parity, behavior divergence ~3.9%), the residual is the
trained-weights / training-trajectory difference of two independent SFT runs — the genuinely-new precision
surface this plan targeted is closed with a documented null result. No precision-changing fix is warranted
beyond the AC-4 decouple. (If a future effort wants to attribute the trajectory difference, the next suspects
are optimizer-update parity under `fused` vs non-fused, RNG/noise/time sampling, dataloader ordering, and
scheduler-step timing — beyond this plan's precision scope.)

## Note on the pinned input (R1/R2 review point)

The AC-1 ledger uses a SHARED deterministic pinned batch (`tests/unit_tests/_precision_pinned.py`,
regenerated from a committed spec via numpy legacy RNG so both venvs produce byte-identical tensors, proven by
the per-field sha256). This is a deliberate value-INDEPENDENT dtype probe: the dtype of every surface (master,
compute, grad, optimizer state, loss, saved/load) does not depend on the input VALUES, only their shapes. The
real-data pinned BEHAVIOR batch is reserved for numeric grad-norm parity on IDENTICAL weights (the
controlled-step criterion), where values matter; it is not required to establish the dtype ledger.

## Artifacts

- Ledgers + gate: `docs/evidence/phase5_precision_ledger_*` + `test_openpi_pytorch_precision_ledger.py` (10).
- Verdict + gate: `docs/evidence/phase5_precision_surface_verdict.json` (`_precision_verdict.py`) +
  `test_openpi_pytorch_precision_verdict.py` (3).
- Decouple + gate: `examples/sft/config/behavior_pi05_vla.yaml` +
  `test_openpi_pytorch_precision_decouple.py` (3).
- Detailed per-surface evidence: `docs/phase5-precision-ledger-evidence.md`.
