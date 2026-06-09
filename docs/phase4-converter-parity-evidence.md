# Phase-4 AC-2 — converter value-parity evidence

**Verdict: `sft_to_new_pytorch.py` is value-lossless up to the documented bf16 cast. The converted checkpoint
is bitwise-identical to the consolidated SFT weights (strip wrapper prefix + fp32→bf16), structurally
identical to the reference new-format eval checkpoint, and produces bitwise-identical loss and actions. The
converter is NOT the cause of the `0.2578`-vs-`0.3203` eval gap.**

The converter's only value transform is `_strip_wrapper_prefix` (`utils/export_sft_checkpoint.py:51`): strip
the wrapper/FSDP key prefixes (`model.` / `_fsdp_wrapped_module.` / `_orig_mod.` / `module.`) and cast float
tensors to bf16. Three independent gates confirm it preserves the model.

## 1. Weight identity (value-lossless up to the bf16 cast) — decisive, CPU

Verified WITHOUT the converter's own helper, so it is a real check of the transform, not a tautology: for
every key in the converted `model.safetensors`, the matching consolidated full-weights tensor (which is
uniformly `model.`-prefixed, fp32) cast to bf16 is **bitwise-equal** to the converted tensor.

| Fact | Value |
|---|---|
| tensors | 667 |
| key map is exactly "strip wrapper prefix" (bijection) | true |
| no wrapper prefix survives | true |
| all converted floats bf16 | true |
| `max_abs_diff` vs independent fp32→bf16 cast | **0.0** (bitwise) |

The consolidated checkpoint is fp32 (667 tensors, `model.`-prefixed); the converted is bf16 (667 tensors,
bare keys). The fp32→bf16 cast is the only difference, and it is the documented, accepted transform.

## 2. Structural identity vs the reference new-format checkpoint — CPU

The converted checkpoint is compared against the reference-trained **eval** checkpoint
`jax_task0000_sft_29999_ptnew/model.safetensors` (same role + format as the converter output, i.e. bf16):

| Fact | Value |
|---|---|
| key-set equal | true |
| n keys | 667 |
| per-key shape equal | true |
| per-key dtype equal (all bf16) | true |
| wrapper prefixes present | none |

(The SFT *base* `pi05_base_pytorch_new` is fp32 — it is the training-init checkpoint, not an eval checkpoint —
so the bf16 eval checkpoint is the correct structural comparator.)

## 3. Forward identity on a fixed input — GPU

The model built from the consolidated full-weights checkpoint and the model built from the converted
checkpoint, on a fixed observation ("turn on radio", seed-0 images/state) + injected flow-matching noise/time
+ `num_steps=5`:

| Quantity | max |Δ| | tolerance |
|---|---|---|
| flow-matching loss | **0.0** | 1e-2 |
| sampled action chunk (normalized) | **0.0** | 1e-2 |
| sampled action chunk (post-denormalization) | **0.0** | 1e-2 |

Exactly 0 — expected, since the weights are bitwise-identical (gate 1) and the inputs/noise/time are fixed.
This confirms the full load → forward → denormalize path is preserved by the conversion.

The same parity holds at the `OpenPiPytorchActionModel` **wrapper** boundary (the interface eval actually
uses) — pre-conversion wrapper vs converted wrapper, same processor:

| Wrapper quantity | max |Δ| |
|---|---|
| SFT loss (`compute_loss` / `sft_forward`, injected noise/time) | **0.0** |
| eval actions via `predict_action_batch` (denormalized) | **0.0** |
| eval `forward_inputs.model_action` (normalized) | **0.0** |

The eval action parity uses a new optional `noise`/`rng` passthrough on `predict_action_batch` (a no-op when
absent, so production sampling is unchanged — verified by `test_predict_action_batch_contract`); the same hook
is reused for the AC-3 paired/deterministic eval.

## Artifacts

- Evidence JSON: `docs/evidence/phase4_converter_parity.json`.
- Generator (reproducible): `tests/unit_tests/_converter_parity_dump.py`.
- Gate tests: `tests/unit_tests/test_openpi_pytorch_converter_parity.py` (3 tests — weight identity,
  structural match, forward parity; skip-gated when the external checkpoints / GPU are absent).

## Consequence for the eval-gap investigation

With AC-1 (norm-stats) and AC-2 (converter) both cleared as null results, the `0.2578`-vs-`0.3203` gap is not
explained by a norm-stats misresolution or a converter defect. The remaining hypothesis is AC-3: the gap is
within eval stochasticity (~1.5 SE on an unpaired 128-episode eval) and must be re-measured under a paired /
confidence-interval protocol at the tuned `num_steps=5` before being treated as a defect.
