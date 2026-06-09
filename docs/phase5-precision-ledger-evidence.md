# Phase-5 AC-1 — complete dual-repo runtime mixed-precision dtype ledger

**Verdict: a non-invasive, runtime-observed dtype ledger is produced for BOTH repos (RLinf BEHAVIOR pi0.5 FSDP
SFT and the reference `pi05_b1k-task0000_sft_pytorch_mixed` FSDP1 training) on rank 0 AND a sharded rank, on
the SAME pinned input (matching per-field sha256). Every record is a `{value, provenance, stage}` object read
off a runtime object — never YAML. All 25 enumerated precision surfaces are observed; the effective recipe is
identical on every dtype surface, with two benign mechanism differences.**

The harnesses drive each repo's REAL build (RLinf `FSDPModelManager.setup_model_and_optimizer` →
`get_model(load_for_training=True)` → `wrap_model`; reference `init_model`) and read the effective runtime
dtype/flag off the built objects + one pinned forward / backward / real-grad-norm / optimizer step. Both repos
consume one deterministic pinned batch (`tests/unit_tests/_precision_pinned.py`, regenerated from the committed
spec via numpy legacy RNG so both venvs produce byte-identical tensors — proven by the recorded per-field
sha256, identical across repos).

## Per-surface ledger (rank 0; identical surface set + dtypes on the sharded rank)

| surface | provenance | RLinf | Reference |
|---|---|---|---|
| same pinned input (state sha256) | `_precision_pinned.load_pinned` | `4a766691…` | `4a766691…` (**identical**) |
| master_param_dtype (outside forward) | `named_parameters()` pre-forward | float32 | float32 |
| **param_dtype_during_forward** | inner Linear forward pre-hook (compute view) | **bfloat16** | **bfloat16** |
| compute_autocast_enabled | `torch.is_autocast_enabled()` inside forward | False | False |
| fsdp_param_dtype | `mixed_precision.param_dtype` | bfloat16 | bfloat16 |
| fsdp_reduce_dtype / gradient_reduction_dtype | `mixed_precision.reduce_dtype` | float32 | float32 |
| fsdp_buffer_dtype | `mixed_precision.buffer_dtype` | float32 | None (default) |
| mp flags (cast_fwd / cast_root / keep_low) | `mixed_precision.*` | F / T / F | F / T / F |
| loss_dtype / output_dtype | forward return | bfloat16 | bfloat16 |
| action / noise / time input dtype | pinned batch tensors | float32 / float32 / float32 | float32 / float32 / float32 |
| grad_dtype (after backward) | `param.grad` | float32 | float32 |
| grad_norm (value, dtype) | RLinf `optimizer_step`→`clip_grad_norm_`; ref `model.clip_grad_norm_` | 134.26, fp32 | 135.43, fp32 |
| optimizer_state_dtype (exp_avg/_sq) | `optimizer.state[*]` after step | fp32 / fp32 | fp32 / fp32 |
| buffer_count / buffer_dtypes | `named_buffers()` | 0 / [] | 0 / [] |
| ema | RLinf build / `config.ema_decay` | none | none |
| grad_scaler | real object | ShardedGradScaler (enabled=False) | None (no scaler) |
| saved_checkpoint_dtype | FSDP `FULL_STATE_DICT` | float32 | float32 |
| load_after_save_dtype | reload into fresh model / re-read | float32 | float32 |
| optimizer betas / eps / weight_decay | `param_groups[0]` | (0.9,0.95) / 1e-8 / 1e-10 | (0.9,0.95) / 1e-8 / 1e-10 |
| optimizer `fused` | `optimizer.defaults` | None (non-fused) | True (fused) |

## Benign mechanism differences (effective-runtime-equivalence standard)

1. **buffer_dtype config:** RLinf sets `fsdp_buffer_dtype=fp32` explicitly; the reference leaves it `None`
   (constructor default). MOOT: BOTH models have **0 persistent buffers** (`buffer_count = 0` on both), so no
   buffer is ever cast — effective behavior identical.
2. **fused vs non-fused AdamW + grad-scaler object:** RLinf builds a (disabled) `ShardedGradScaler` and a
   non-fused AdamW; the reference uses no scaler and `fused=True`. Both keep fp32 optimizer state; the scaler
   is disabled on both (no scaling), and `fused` only changes the AdamW update kernel. Numeric impact of
   `fused` is deferred to the controlled-step parity (a later criterion).

Non-precision context (not a mismatch): the recorded `lr` differs (RLinf 2.4975e-08 vs reference 2.5e-05) only
because the RLinf harness runs the real `optimizer_step` which advances the warmup LR scheduler, while the
reference records the construction-time peak; both configs share peak_lr 2.5e-5 / 1000-step warmup.
`grad_norm` values differ (134 vs 135) because the two models have different (independent) weights — the dtype
is identically fp32; numeric grad-norm parity on IDENTICAL weights is the controlled-step criterion, not AC-1.

## Method note — read the EFFECTIVE value, not a proxy

The optimizer betas are read from `optimizer.param_groups[0]`, not `optimizer.defaults`: RLinf passes betas
per-group, so `defaults` would show the AdamW constructor default `(0.9, 0.999)` — a misleading proxy. The
per-group read shows the configured `(0.9, 0.95)`, matching the reference. (See
`BL-20260608-optimizer-hparams-param-groups-not-defaults`.)

## Artifacts

- Pinned input: `tests/unit_tests/_precision_pinned.py` + committed spec
  `docs/evidence/phase5_precision_pinned_spec.json`.
- Harnesses: `tests/unit_tests/_precision_ledger_{rlinf,ref}.py` (non-invasive; torchrun, respective venvs).
- Ledgers: `docs/evidence/phase5_precision_ledger_{rlinf,ref}_rank{0,1}.json` (25/26 surfaces each).
- Gate: `tests/unit_tests/test_openpi_pytorch_precision_ledger.py` (7 tests): all four ledgers ok on rank 0 +
  a sharded rank; every surface present with runtime provenance on both ranks; same pinned input; non-empty
  sharded-rank grad; runtime autocast/scaler; during-forward compute dtype + save/load; grad-norm value+dtype.
