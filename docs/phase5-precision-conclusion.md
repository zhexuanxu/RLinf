# Phase-5 — precision-alignment conclusion (per-surface verdict + numeric grad parity + null result)

**Conclusion: the RLinf BEHAVIOR pi0.5 FSDP SFT mixed-precision recipe is VERIFIED EFFECTIVELY IDENTICAL to
the reference on every runtime dtype surface. Precision is therefore NOT the source of the residual
reference-minus-RLinf eval-success gap measured in the prior phase. The one real config issue — the
`param_dtype: ${actor.model.precision}` coupling footgun — has been DECOUPLED (set explicitly to `bf16`). The
draft's premise ("definitely not aligned") was a misreading of an overloaded config field; its literal
`precision: fp32` edit would have disabled bf16 compute and was correctly not applied.**

## Per-surface effective-equivalence verdict

From the committed dual-repo runtime ledgers (`docs/evidence/phase5_precision_ledger_{rlinf,ref}_rank{0,1}.json`),
the per-surface verdict (`docs/evidence/phase5_precision_surface_verdict.json`, gate
`test_openpi_pytorch_precision_verdict.py`) is, on rank 0 AND a sharded rank:

- **PASS (effectively identical), 46 surface-rows:** FSDP `param_dtype` bf16; `reduce_dtype`/gradient-reduction
  fp32; MP flags; master fp32; during-forward compute bf16; runtime autocast off; loss/output/action/noise/time
  dtype; grad fp32; grad-norm fp32 accumulation dtype; optimizer-state fp32; saved + load-after-save fp32 (all
  667 tensors); buffer_count 0; buffer-default probe (omitted → fp32); EMA none; **loaded-weight fingerprint
  (key-independent fp32 set-hash, 667 tensors) byte-identical across repos.**
- **BENIGN (documented mechanism difference), 8 surface-rows:** `fsdp_buffer_dtype` fp32-explicit vs None
  (MOOT — 0 buffers either side, FSDP leaves an omitted buffer at fp32 per the probe); `grad_scaler` disabled
  ShardedGradScaler vs none (both apply no scaling); `optimizer` non-fused vs fused (both fp32 state; identical
  betas/eps/wd; `lr` differs only by warmup-schedule state); `grad_norm` value differs by ~0.5% — **same base
  checkpoint and same pinned batch, so the gap is bf16-compute non-determinism across execution paths (RLinf
  eager vs reference `torch.compile`; FSDP wrap/all-reduce order), NOT independent weights; the dtype, the
  surface judged, is identically fp32.**
- **CONTEXT, 2 rows:** reference-only `pytorch_training_precision` (`mp_bfloat16` → fp32 load).
- **MISMATCH: 0.** No surface diverges in effective dtype.

## Numeric grad-norm parity on IDENTICAL weights (controlled step)

Both repos load the SAME base checkpoint `pi05_base_pytorch_new` (RLinf `behavior_pi05_vla.yaml:63`
`model_path`; reference `config.py:881` `pytorch_weight_path`, loaded strict at `train_pytorch_new.py:333`) and
run ONE forward/backward on the SAME real pinned SFT batch (same noise/time). The gate
`test_openpi_pytorch_precision_grad_parity.py` proves from RUNTIME tensors:

- **Loaded weights byte-identical across repos:** a key-independent set-hash over each fp32 param value,
  gathered from an unsharded `FULL_STATE_DICT` *before* the optimizer step, matches RLinf↔reference on every rank
  (`013e7bf3…`, 667 distinct tensors). This establishes "identical weights" without trusting config text.
- **Grad-norm numerically equivalent:** identical fp32 accumulation dtype, value 35.6304 (RLinf) vs 35.4615
  (reference) — relative gap 0.47%, within the documented 0.02 bf16-compute tolerance. The residual is execution-path
  non-determinism (compile-vs-eager, FSDP all-reduce order), not a precision-recipe difference.

This is the controlled-step numeric criterion the dtype ledger alone could not establish: with weights and input
held identical, the recipe produces the same gradient magnitude to bf16 precision.

## Independent audit

An independent analyze-route audit of the full evidence package (ledgers, verdict, harnesses, pinned artifact,
decouple gate) returned **OVERALL PASS** on all six questions (surface enumeration, runtime provenance,
pinned-input integrity, buffer-default observation, decouple, verdict integrity). Its one substantive finding —
the `grad_norm` BENIGN note had mis-attributed the value gap to "independent weights" when both repos load the
same base checkpoint — has been corrected in `_precision_verdict.py` (the gap is bf16-compute/compile/FSDP
non-determinism; the verdict itself was already correct). The full per-question verdict, the route deviation
(`analyze` → fresh-context auditor after `ask-codex` returned empty output, per
BL-20260603-codex-remote-compaction-fail), and the file:line references are committed in
**`docs/phase5-precision-independent-audit.md`**.

## AC-4 — the config footgun decoupled

`examples/sft/config/behavior_pi05_vla.yaml` now sets `fsdp_config.mixed_precision.param_dtype: bf16`
explicitly, no longer interpolated from `${actor.model.precision}`. The dry-run resolution gate
(`test_openpi_pytorch_precision_decouple.py`) proves overriding `actor.model.precision=fp32` leaves the FSDP
compute `param_dtype` at `bf16` (decoupled) while `load_for_training` stays True (fp32 master). The AC-1 ledger
re-run after the edit shows `fsdp_param_dtype` unchanged (bf16) and master unchanged (fp32) — the effective
recipe is preserved.

## Verified-aligned null result

Every same-input precision surface is now proven aligned by RUNTIME observation on the same real pinned input
(sha-proven identical), across both ranks, AND the grad-norm matches numerically on identical weights. So the
significant residual eval gap (Phase-4, +9.2%) is **not** a mixed-precision misconfiguration. Combined with
Phase-4 (norm-stats byte-identical, converter value-lossless, same-batch raw→loss match, pinned training-step
parity, behavior divergence ~3.9%), the residual is the trained-weights / training-trajectory difference of two
independent SFT runs — the genuinely-new precision surface this plan targeted is closed with a documented null
result. No precision-changing fix is warranted beyond the decouple. (If a future effort wants to attribute the
trajectory difference, the next suspects are optimizer-update parity under `fused` vs non-fused, RNG/noise/time
sampling, dataloader ordering, and scheduler-step timing — beyond this plan's precision scope.)

## Note on the pinned input

The ledgers and the grad-parity gate both consume a REAL pinned BEHAVIOR SFT batch + noise/time, generated by
the reference SFT data loader (`tests/unit_tests/_ref_pinned_run.py` on `/mnt/public/.../2025-challenge-demos`)
and committed as a compact rank-addressable artifact (`docs/evidence/phase5_precision_pinned_artifact.json` +
`phase5_precision_pinned_batches.npz` + `phase5_precision_pinned_noise_time.npz`). The loader
(`tests/unit_tests/_precision_pinned.py`) validates `format == ref_pinned_npz` and `synthetic is False` with no
synthetic fallback, and both venvs read it identically (per-field sha256 match, 12 fields). Surface DTYPES do
not depend on input values, but using the real batch also makes the loss and the grad-norm meaningful for the
numeric parity criterion above.

## Artifacts

- Ledgers + gate: `docs/evidence/phase5_precision_ledger_*` (`_precision_ledger_{rlinf,ref}.py`) +
  `test_openpi_pytorch_precision_ledger.py` (13).
- Verdict + gate: `docs/evidence/phase5_precision_surface_verdict.json` (`_precision_verdict.py`) +
  `test_openpi_pytorch_precision_verdict.py` (3).
- Numeric grad-norm parity gate: `test_openpi_pytorch_precision_grad_parity.py` (4) — identical-weight set-hash +
  grad-norm tolerance on the committed ledgers.
- Pinned real-batch artifact: `docs/evidence/phase5_precision_pinned_artifact.json` +
  `phase5_precision_pinned_batches.npz` + `phase5_precision_pinned_noise_time.npz`
  (`_ref_pinned_run.py` → `_precision_pinned.build_pinned`).
- Decouple + gate: `examples/sft/config/behavior_pi05_vla.yaml` +
  `test_openpi_pytorch_precision_decouple.py` (3).
- Independent analyze-route audit: `docs/phase5-precision-independent-audit.md` (task6, OVERALL PASS).
- Detailed per-surface evidence: `docs/phase5-precision-ledger-evidence.md`.
