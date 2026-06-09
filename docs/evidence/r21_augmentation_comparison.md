# Augmentation is NOT the first-50 residual — actual reference-path verification (corrected R22)

**Correction.** An earlier version of this note (R21) claimed the residual tail gap (RLinf descends
to ≈0.046 vs the reference ≈0.090) was a missing-augmentation recipe divergence, by reading
`openpi.models_pytorch` (`preprocessing_pytorch.py` / `pi0_pytorch.py`), which augments
unconditionally at `train=True`. **That was the wrong module.** The actual reference training script
imports `openpi.models_pytorch_new` — which gates augmentation on `rng` and trains with `rng=None`,
byte-equivalent to RLinf. There is **no augmentation divergence**. This file is the corrected,
verified record.

## The actual reference path (`openpi.models_pytorch_new`)
Driven/diffed directly against `openpi-comet-pytorch-mixed`:
- `scripts/train_pytorch_new.py:38-42` imports **`openpi.models_pytorch_new`** (`pi0`, `model`,
  `pi0_config`, `checkpoint_format`) — NOT `openpi.models_pytorch`.
- `scripts/train_pytorch_new.py:520` — the training step calls
  `model(observation, actions, train=True, noise=noise, time=time)` with **no `rng`** (and
  `noise`/`time` are `None` unless `use_consistent`).
- `models_pytorch_new/pi0.py:414-425` — `Pi0.forward(..., rng=None)` forwards
  `compute_loss(..., rng=rng)` → `rng=None`.
- `models_pytorch_new/pi0.py:297` — `compute_loss` calls
  `model.preprocess_observation(observation, train=train, rng=rng)` → `rng=None`.
- `models_pytorch_new/model.py` `preprocess_observation` — under `train=True` the crop offset,
  rotation angle, and the entire color-jitter block are gated on **`rng is not None`**
  (`top/left = randint(..., generator=rng) if rng is not None else 0`;
  `angle = ... if rng is not None else 0.0`; `if rng is not None:` color jitter). With `rng=None`:
  only a deterministic top-left 95 % crop — no random crop, no rotation, no color jitter.

## Byte-equivalence with RLinf's vendored model
- `diff openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new/model.py:161-235`
  vs `rlinf/models/embodiment/openpi_pytorch/pi0_model/model.py:161-235` → **IDENTICAL**
  (the entire `preprocess_observation` augmentation block).
- RLinf's `sft_forward` (`openpi_action_model.py:122`) calls `compute_loss(train=True)` with
  `rng=None` (default), the SAME as the reference training call.

So both the reference (`models_pytorch_new`) and RLinf apply only the deterministic crop during
training (`rng=None`) — **the augmentation states are identical**. This is also consistent with the
committed same-batch parity harness: `tests/unit_tests/_ref_model_loss_dump.py` evaluates the
reference MODEL at `train=True, rng=None` and `tests/unit_tests/test_openpi_pytorch_sft_loss_parity_gpu.py`
matches RLinf against it (R15 parity |Δ|=0.0024).

## Consequence
Augmentation is **ruled out** as the first-50 residual. The residual must be localized elsewhere
(see `docs/sft-first50-step-evidence.md` → "Residual localization (R22)"). Note the older
`models_pytorch` path *would* augment at `train=True`, but it is not the path the reference run
uses — auditing it was the R21 error.
