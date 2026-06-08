# Phase-5 — independent analyze-route audit (task6)

**Verdict: OVERALL PASS.** An independent cross-check of the committed Phase-5 precision evidence
confirms the dual-repo runtime dtype ledger covers every mixed-precision surface, reads each from a
runtime object (never config text), observes the omitted-`buffer_dtype` default rather than assuming
it, keeps the FSDP compute `param_dtype` decoupled from the load/precision selector, and reaches 0
MISMATCH. The audit's one substantive finding — the `grad_norm` BENIGN rationale mis-said
"independent weights" — was integrated (the verdict itself was already correct). No precision surface
is missed or asserted from config text.

## Route and methodology (task6 is routed `analyze`)

- task6's plan routing is `analyze` → `/humanize:ask-codex`. The remote `ask-codex` run on this audit
  produced **EMPTY output after ~20 min** on the broad multi-file read (read ~25-surface ledgers +
  two harnesses + the verdict generator across BOTH repos). This is the failure mode recorded in
  **BL-20260603-codex-remote-compaction-fail** (Codex `exec` on a broad read-many-files investigation
  balloons context and the remote compaction step aborts the whole run with no usable output).
- Per that lesson, the audit was executed by an **independent read-only fresh-context auditor**
  (separate context; Read/Glob/Grep/Bash over the named files in both repos) which answered the six
  audit questions with file:line references. This is a documented route deviation (`analyze` → agent),
  the same fallback the lesson prescribes for broad reference-repo reads.
- To keep the artifact verifiable despite the auditor returning a conclusion rather than a Codex
  transcript, **Claude independently re-verified every load-bearing claim** (the base-checkpoint paths
  in both repos, the cross-repo fingerprint identity, the grad-norm values, and the current file:line
  anchors below). All references are current as of commit `6df73385`.

## Audited evidence

- `docs/evidence/phase5_precision_ledger_{rlinf,ref}_rank{0,1}.json` — per-repo, per-rank runtime
  dtype ledgers (each surface a `{surface, repo, rank, value, provenance, stage}` record).
- `docs/evidence/phase5_precision_surface_verdict.json` — per-surface verdict (46 PASS / 8 BENIGN /
  2 CONTEXT / 0 MISMATCH across both ranks).
- `docs/evidence/phase5_precision_pinned_artifact.json` + `phase5_precision_pinned_{batches,noise_time}.npz`
  — the committed real pinned SFT batch + noise/time.
- Harnesses `tests/unit_tests/_precision_ledger_{rlinf,ref}.py`, the pinned loader
  `tests/unit_tests/_precision_pinned.py`, the verdict generator `tests/unit_tests/_precision_verdict.py`.
- Decouple config `examples/sft/config/behavior_pi05_vla.yaml` + gate
  `tests/unit_tests/test_openpi_pytorch_precision_decouple.py`.

## Per-question verdicts

**Q1 — Surface enumeration completeness: PASS.** Every mixed-precision surface needed to judge effective
equivalence is present in both repos' ledgers on both ranks. master/load + post-FSDP-outside-forward
`master_param_dtype` (`_precision_ledger_rlinf.py:215`); during-forward compute `param_dtype_during_forward`
(`:262`); FSDP `param_dtype`/`reduce_dtype`/`buffer_dtype` + the three MP flags
(`surf[f"fsdp_{k}"]` `:197`); gradient-reduction dtype; grad dtype; grad-norm value+dtype (`:314`);
optimizer state + construction; loss/output/action/noise/time dtype; autocast (`:267`); grad-scaler;
buffer count+dtypes; EMA; saved + load-after-save dtype (`:358`, `:371`); plus the new identical-weight
`loaded_weight_fingerprint` (`:304`). The required-surface set is enforced by
`test_openpi_pytorch_precision_ledger.py` (`_REQUIRED_SURFACES`, `test_every_surface_present_*`). No
surface missing.

**Q2 — Runtime provenance (no config-text assertions): PASS.** Each dtype/flag surface is read off a
built object: the FSDP `MixedPrecision` policy object, `named_parameters()`/`named_buffers()` tensors,
an inner-Linear forward pre-hook that captures both the during-forward compute dtype and
`torch.is_autocast_enabled()` (autocast is genuinely runtime, not config-derived), `param.grad`
tensors, the real `optimizer.param_groups[0]`/`state` (betas read from param_groups per
BL-20260608-optimizer-hparams-param-groups-not-defaults, not the misleading `optimizer.defaults`), and
a gathered FSDP FULL_STATE_DICT. The gate asserts every surface provenance contains no `.yaml`/"config
text" (`test_every_surface_present_with_runtime_provenance_both_ranks`). Two absence-of-mechanism
claims (`ema` none, reference `grad_scaler` none) are structural, not dtype reads, and do not affect
equivalence; the reference-only `pytorch_training_precision` is config-derived but correctly classed
CONTEXT (excluded from the cross-repo verdict), not asserted as a dtype surface.

**Q3 — Pinned-input integrity: PASS.** The pinned input is a REAL SFT batch: the manifest sets
`format: ref_pinned_npz`, `synthetic: false`, generator `_ref_pinned_run.py`
(`phase5_precision_pinned_artifact.json`). The loader rejects any other shape with no synthetic
fallback — `_precision_pinned.py:97-98` (format), `:101` (generator allow-list), and a
`synthetic is not False` guard — documented "no synthetic fallback" at `:26`. Both repos consume it
identically (per-field sha256 match, 12 fields, computed at load time via `sha_tensor`, not copied from
the manifest), and the two committed NPZ files referenced by the artifact exist. Gated by
`test_pinned_input_uses_committed_ref_pinned_artifact`, `test_precision_pinned_loader_rejects_shape_only_manifest`,
`test_precision_pinned_builder_extracts_ref_pinned_npz`.

**Q4 — Buffer-default OBSERVED not assumed: PASS.** A real `nn.Linear` with an fp32 buffer is wrapped
in live FSDP `MixedPrecision(buffer_dtype=None)` and the buffer dtype is re-read after wrap
(`_precision_ledger_rlinf.py:88` `_buffer_default_probe`; reference `:91`). Both observe the buffer
stays fp32, proving an omitted `buffer_dtype` leaves the buffer at its original dtype — which makes the
`fsdp_buffer_dtype` fp32-vs-None difference genuinely moot (both models also have `buffer_count == 0`).
Gated by `test_omitted_buffer_dtype_default_observed`.

**Q5 — AC-4 decouple: PASS.** `fsdp_config.mixed_precision.param_dtype` is the literal `bf16`
(`behavior_pi05_vla.yaml:106`), no longer `${actor.model.precision}`; `reduce_dtype: fp32` (`:107`),
`buffer_dtype: fp32` (`:108`). The dry-run gate proves overriding `actor.model.precision=fp32` leaves
`param_dtype` at `bf16` while `load_for_training` stays True (fp32 master)
(`test_openpi_pytorch_precision_decouple.py`). The runtime decouple is independently established by the
loader's `load_for_training=True` branch (strict fp32 load, no model downcast; compute dtype owned by
FSDP MixedPrecision). No remaining path lets the precision/load selector silently change the FSDP
compute dtype.

**Q6 — Verdict integrity: PASS (one rationale finding, integrated).** All 8 BENIGN rows are
effective-runtime-equivalent on dtype: `fsdp_buffer_dtype` (moot, 0 buffers + probe); `grad_scaler`
(both disabled); `optimizer` (identical betas/eps/wd + fp32 state; differs only by fused kernel and
warmup-step lr); `grad_norm` (dtype identically fp32; value differs at bf16-noise scale). No row should
be MISMATCH, and no value is asserted from config text. **Finding:** the `grad_norm` BENIGN note
originally read "the two models have independent weights." That is false — BOTH repos load the SAME
base checkpoint `pi05_base_pytorch_new` (RLinf `behavior_pi05_vla.yaml:63` `model_path`; reference
`config.py:881` `pytorch_weight_path`, loaded strict at `train_pytorch_new.py:333`), proven at runtime
by the identical `loaded_weight_fingerprint` set-hash (`013e7bf3…`, 667 tensors) across repos. The
~0.5% value gap (35.6304 vs 35.4615) is bf16-compute non-determinism across execution paths (RLinf
eager vs reference `torch.compile`; differing FSDP wrap/all-reduce order), NOT independent weights. The
verdict (BENIGN, dtype identical fp32) was already correct; only the explanatory text was wrong.
**Integrated:** the rationale was corrected in `_precision_verdict.py:41` (and the CONTEXT row no longer
mirrors the reference value into the `rlinf` field). This same insight is recorded as
BL-20260608-identical-weights-fullstatedict-sethash.

## Result

The audit found NO missed precision surface, NO surface asserted from config text, and confirmed the
omitted-`buffer_dtype` default is observed. Its single finding (the `grad_norm` rationale) was a
documentation inaccuracy, now fixed; it does not change any verdict. With the fix integrated, the
Phase-5 precision evidence soundly establishes — by runtime observation on a real pinned batch and on
identical weights — that the two FSDP SFT mixed-precision recipes are effectively dtype-identical on
every surface (44/46 PASS plus the identical-weight fingerprint; 8 BENIGN documented; 0 MISMATCH).
