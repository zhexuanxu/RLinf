# Phase-5 AC-1 — dual-repo runtime mixed-precision dtype ledger

**Verdict: a non-invasive, runtime-observed dtype ledger was produced for BOTH repos (RLinf BEHAVIOR pi0.5
FSDP SFT and the reference `pi05_b1k-task0000_sft_pytorch_mixed` FSDP1 training), on rank 0 AND a sharded
rank. Every value is read off a built runtime object (provenance recorded), never from YAML. The effective
mixed-precision + optimizer recipe is IDENTICAL on every dtype surface, with two benign, documented mechanism
differences.**

This is the surface the draft's config-text reasoning bypassed: the ledger drives each repo's REAL build
(RLinf `FSDPModelManager.setup_model_and_optimizer` → `get_model(load_for_training=True)` → `wrap_model`;
reference `init_model(...)`) and reads the effective runtime dtype off the built objects + one forward /
backward / optimizer step.

## Per-surface ledger (rank 0; identical on the sharded rank)

| surface | RLinf | Reference | verdict |
|---|---|---|---|
| FSDP `param_dtype` (compute) | bfloat16 | bfloat16 | **identical** |
| FSDP `reduce_dtype` | float32 | float32 | **identical** |
| FSDP `buffer_dtype` | float32 (explicit) | None (default) | benign mechanism diff* |
| MP flags `cast_forward_inputs` / `cast_root_forward_inputs` / `keep_low_precision_grads` | False / True / False | False / True / False | **identical** |
| master / loaded params (outside forward) | float32 | float32 | **identical** |
| gradient dtype (post-backward) | float32 | float32 | **identical** |
| loss / flow-matching compute | bfloat16 | bfloat16 | **identical** |
| `torch.autocast` | off | off | **identical** |
| grad-scaler | off | off | **identical** |
| AdamW state (`exp_avg` / `exp_avg_sq`) | float32 | float32 | **identical** |
| AdamW betas | (0.9, 0.95) | (0.9, 0.95) | **identical** |
| AdamW eps | 1e-08 | 1e-08 | **identical** |
| AdamW weight_decay | 1e-10 | 1e-10 | **identical** |
| AdamW `fused` | None (non-fused) | True (fused) | benign mechanism diff** |
| `pytorch_training_precision` (reference) | n/a | `mp_bfloat16` → fp32 load | (reference load selector) |

\* **buffer_dtype:** RLinf sets `buffer_dtype=fp32` explicitly; the reference omits it (None → buffers keep
their original dtype). This is moot here: the reference model has NO persistent buffers
(`buffer_dtype_observed = {}`), so neither repo casts any buffer to bf16. Effective: identical (no low-precision
buffers either way).

\** **fused AdamW:** the reference constructs AdamW with `fused=True`; RLinf is non-fused (`fused=None`). Both
keep fp32 optimizer state; `fused` only changes the update KERNEL (a tiny arithmetic-ordering difference), not
any dtype. Benign under the effective-runtime-equivalence standard. (Candidate for the per-surface verdict in
the next task.)

## Method note — read the EFFECTIVE value, not a proxy

An early ledger read `optimizer.defaults["betas"]` and reported RLinf betas as (0.9, 0.999), implying a
mismatch. That was a harness proxy bug: RLinf passes betas via the PER-PARAM-GROUP dict (so `defaults` shows
the AdamW constructor default), while the configured value is `adam_beta2: 0.95`. Reading
`optimizer.param_groups[0]["betas"]` shows the real (0.9, 0.95), which MATCHES the reference. The ledger now
reads the per-group hyperparameters on both sides.

## Artifacts

- Harnesses (non-invasive; drive each repo's real build): `tests/unit_tests/_precision_ledger_rlinf.py`
  (torchrun, RLinf venv) and `tests/unit_tests/_precision_ledger_ref.py` (torchrun, reference venv).
- Ledgers: `docs/evidence/phase5_precision_ledger_{rlinf,ref}_rank{0,1}.json`.
- Gate test: `tests/unit_tests/test_openpi_pytorch_precision_ledger.py` (both ledgers exist on rank 0 + a
  sharded rank; every surface recorded with runtime provenance; reference buffer-default observed; optimizer
  hyperparameters read from param groups).
