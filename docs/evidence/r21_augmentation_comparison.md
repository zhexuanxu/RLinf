# R21 — Training-time image augmentation: reference vs RLinf SFT (measured)

The Round-20 evidence attributed the residual tail gap (RLinf descends to ≈0.046 vs the reference
≈0.090) to "RNG/aggregation" **without measurement** — a reviewer rejected that. This note records
the MEASURED divergence found by reading both training paths. It is a **measured recipe
divergence** (fact); whether it *fully* accounts for the residual is a **hypothesis** not yet
demonstrated by a re-run (see "Status").

## Reference: augments unconditionally at `train=True`
`openpi-comet-pytorch-mixed`:
- `src/openpi/models_pytorch/pi0_pytorch.py:318` — `compute_loss` calls
  `self._preprocess_observation(observation, train=True)`.
- `src/openpi/models_pytorch/preprocessing_pytorch.py:57-140` — when `train=True`, applies, via the
  **global** torch RNG (`torch.randint` / `torch.rand`, no explicit generator):
  - geometric (base_0_rgb): **random** crop to 95 % at a **random** position + resize back; **random**
    rotation in **[-5°, +5°]** (grid_sample);
  - color (all cameras): **random brightness ×[0.7, 1.3]**, **random contrast ×[0.6, 1.4]** (no
    saturation jitter).
- `scripts/train_pytorch_new.py:520` — the training loop calls `model(observation, actions,
  train=True, noise=noise, time=time)`, so **every training step augments**.

## RLinf: SFT path passes `rng=None` → augmentation is OFF
`rlinf/models/embodiment/openpi_pytorch/`:
- `openpi_action_model.py:122` — `sft_forward` calls `self.model.compute_loss(observation, actions,
  train=True)` with **`rng` defaulting to `None`**.
- `pi0_model/pi0.py:294` — `compute_loss` forwards `preprocess_observation(observation, train=train,
  rng=rng)` (`rng=None`).
- `pi0_model/model.py:192-230` — under `train=True` the crop offset, rotation angle, and the entire
  color jitter are gated on **`rng is not None`**:
  - crop: `top/left = randint(..., generator=rng) if rng is not None else 0` → with `rng=None`, a
    **deterministic top-left** 95 % crop + resize;
  - rotation: `angle = ... if rng is not None else 0.0` → **no rotation**;
  - color jitter (brightness/contrast/saturation): the whole block is `if rng is not None:` → **no
    color jitter**.

So with `train=True, rng=None`, RLinf applies only a deterministic top-left crop — **no random
crop, no rotation, no color jitter**. The reference applies all three, every step.

## Consequence (hypothesis for the tail-level offset)
The reference trains on harder, augmented images → higher training loss; RLinf trains on
near-clean (deterministic-crop-only) images → lower training loss. This is consistent with RLinf
descending **below** the reference at the tail (RLinf ≈0.046 vs reference ≈0.090). It does NOT
fully explain the warmup-region (steps 5–15) per-step variance or the period-8 streaming spike at
step 19.

A second, separate plumbing issue blocks a trivial "just pass an `rng`" demonstration: in
`pi0.py:294,303` the SAME `rng` feeds both the CPU augmentation ops (`preprocess_observation`,
`torch.randint`/`torch.rand` on CPU) and the CUDA flow-matching noise
(`torch.randn(..., device=cuda, generator=rng)`); a single generator cannot serve both devices.
Aligning RLinf's training-time augmentation to the reference therefore requires (a) separating the
augmentation RNG (CPU, per-step) from the noise RNG (CUDA/None), and (b) matching the augmentation
ops/ranges (symmetric brightness/contrast, no saturation, random crop position, ±5° rotation).

## Status
- **Measured (fact):** the reference augments at `train=True`; RLinf's SFT path (`rng=None`) does
  not. The augmentation **state differs** — this replaces the unproven R20 "RNG/aggregation" claim.
- **Hypothesis (not yet demonstrated):** that this augmentation divergence is the dominant cause of
  the residual level offset. Confirming it requires the augmentation-alignment fix above + an 8-GPU
  re-run comparing the first-50 loss with vs without the reference-matched augmentation. Registered
  as the next blocking task for task15 closure (not done this round).
