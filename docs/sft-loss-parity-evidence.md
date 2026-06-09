# BEHAVIOR SFT Fixed-Batch Forward-Loss Parity — Evidence (task14, AC-11 / DEC-1 (a))

**Verdict: the DEC-1(a) fixed-batch forward-loss parity gate is MET.** On an IDENTICAL
batch + identical weights + identical noise/time, RLinf's flow-matching loss matches the
reference *model's* loss to a max per-batch difference of **0.0024** (well within the DEC-1
band of ±0.0123) — both at `train=False` and at the production `train=True, rng=None`
(deterministic-crop) path. The committed gate is
`tests/unit_tests/test_openpi_pytorch_sft_loss_parity_gpu.py`.

## The R14 "divergence" was a measurement artifact

R14's probe (`tools/sft_loss_parity_probe.py`) compared RLinf's *marginal* loss over a
random 8-batch sample (~0.29–0.31) to the reference's *logged* step-0/first-10 loss
(≈0.244). That is not a same-batch comparison (the limitation Codex flagged in the R14
review). The reference's logged ≈0.244 is computed on the reference run's specific early
training frames, which are easier than a random sample. When the **reference model itself**
is run on the same random 8-batch sample, it reports ≈0.318 — i.e. RLinf and the reference
agree on the same frames; the gap was entirely the apples-to-oranges comparison.

## Same-batch reference forward (the correct gate)

`tests/unit_tests/_ref_model_loss_dump.py` (run in the reference py3.11 venv on a GPU) loads
the reference Pi0 from `pi05_base_pytorch_new` (cast to bf16, like the RLinf training build),
builds fixed `turning_on_radio` batches, generates per-batch noise/time, and dumps the
reference per-batch loss at `train=False` and `train=True, rng=None`. RLinf's
`Pi0.compute_loss` is then run on the SAME batches + noise/time.

Measured (1× A800; identical batch+weights+noise+time):

| Path | reference (mean) | RLinf (mean) | max per-batch \|Δ\| | within DEC-1 |
|------|------------------|--------------|---------------------|--------------|
| `train=False` (no aug) | 0.317756 | 0.317590 | **0.002438** | yes |
| `train=True, rng=None` (production) | 0.318108 | 0.318064 | **0.000953** | yes |

Per-batch `train=False` table (8 batches): ref/rlinf agree to ≤0.0024 on every batch
(e.g. 0.298801/0.301240, 0.255309/0.255618, 0.367815/0.366448). The model forward is
identical within bf16 numerics.

## What this establishes

- **The RLinf model forward is provably identical to the reference** on the SFT
  flow-matching path (eval action parity already showed the denoising forward; this shows the
  training loss too).
- **The augmentation is not a divergence.** Both the reference training and the RLinf worker
  (`sft_forward`) call `compute_loss(train=True)` with **no `rng`** (`scripts/train_pytorch_new.py:520`;
  `openpi_action_model.sft_forward`), so `preprocess_observation` applies only a deterministic
  top-left 95% crop (rotate/jitter are gated on `rng is not None`), which moves the loss by
  <0.001. (R14's probe passed a real `rng` for reproducibility — full random aug — which is
  NOT the production path; that is the only reason its `train=True` marginal differed from the
  worker path.)
- **DEC-1(a) is satisfied.** task14's fixed-batch parity gate passes; task15 is unblocked.

## Run

```
# reference per-batch losses (reference venv, GPU):
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
  /mnt/public/xzxuan/repos/openpi-comet/.venv/bin/python \
  tests/unit_tests/_ref_model_loss_dump.py <out>
# RLinf parity (committed, GPU + reference-skip-gated):
TMPDIR=/mnt/public/xzxuan/tmp CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYTHONPATH=. \
  python -m pytest tests/unit_tests/test_openpi_pytorch_sft_loss_parity_gpu.py
# -> 1 passed (134s)
```

- Model: `pi05_base_pytorch_new`, `model.safetensors` sha256[:16] (first 64MB) `650d624bc119a28f`.
- Norm stats: task-0000 `norm_stats.json` sha256[:16] `d66ed16830a98f90`.
- `tools/sft_loss_parity_probe.py` remains as a marginal-loss probe; its marginal-vs-logged
  DEC-1 line is NOT the gate (see above) — the same-batch test is.

## Next (task15, now unblocked)

task15 (first-50-step run on 8 GPUs, `grad_accum=1`) must compare the first 50 rank-0 logged
pre-step losses against the reference log band. Note the reference's own loss curve decays
quickly from ≈0.246 (step 0) to ≈0.11 (step 100) at base/near-base weights; matching the
early-step values depends on the data/streaming order, which is task15's concern. The
model-correctness gate (fixed-batch parity) is now met.
