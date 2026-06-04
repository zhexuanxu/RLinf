# BEHAVIOR SFT First-50-Step Training Evidence (task15, AC-11 / DEC-1 (b))

GPU run-evidence (DEC-5) for the `use_skill:false` first-50-step training-loss gate.

**Verdict: the run executes cleanly on 8 GPUs and the first-50 losses are captured, but they
do NOT satisfy DEC-1 (b).** RLinf's per-step training loss is **flat at ≈0.24** over the first
50 steps while the reference loss **drops sharply from ≈0.246 to ≈0.090** — a real
training-dynamics divergence. Per-step `|Δ| ≤ 0.03` holds for only 12/50 steps. This is
documented (no silent pass) and is the next round's blocking investigation.

## Run

```
REPO_PATH=/mnt/public/xzxuan/repos/RLinf_pi05 EMBODIED_PATH=$REPO_PATH/examples/sft \
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TMPDIR=/mnt/public/xzxuan/tmp \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=$REPO_PATH \
python examples/sft/train_vla_sft.py \
  --config-path $REPO_PATH/examples/sft/config --config-name behavior_pi05_vla \
  runner.max_steps=60 runner.save_interval=100000 \
  runner.logger.log_path=/mnt/public/xzxuan/tmp/r16_sft_results
# -> EXIT=0; 60 steps in ~8 min; train/loss logged per step to tensorboard.
```

- 8× A800-80GB; FSDP FULL_SHARD bf16; `grad_accum = 256/(32×8) = 1`; seed 42.
- Model `pi05_base_pytorch_new`; task-0000 norm stats; `turning_on_radio`; `use_skill:false`.
- Loss = rank-AVG-reduced per-step `train/loss` (the worker all-reduces AVG; the runner logs
  it) — the same quantity the reference logs.

## First-50 comparison (RLinf vs reference log)

| metric | RLinf | reference |
|--------|-------|-----------|
| step 0 | 0.2286 | 0.2461 (\|Δ\|=0.0175, within 0.03) |
| first-10 mean | 0.2256 | 0.2438 |
| first-50 mean | 0.2352 | 0.1675 |
| first-50 min / max | 0.1804 / 0.3923 | 0.0903 / 0.2598 |
| trend | flat (≈0.24, mean 0.226→0.240) | **drops 0.246 → 0.090** |

Per-step `|Δ| ≤ 0.03`: **12/50**. Within the reference's mean±2σ band `[0.055, 0.280]`:
45/50 — but that band is inflated by the reference's own steep drop, so the band reading is
not meaningful here; the per-step and trend readings are.

Representative per-step (RLinf | ref | |Δ|): step0 0.229|0.246|0.017; step10 0.218|0.248|0.031;
step20 0.239|0.167|0.072; step30 0.236|0.133|0.103; step40 0.257|0.111|0.146; step49
0.220|0.090|0.130. The full series is reproducible from the tensorboard log.

## Findings

1. **Steps 0–14 track the reference closely** (mostly `|Δ| < 0.04`) — consistent with the
   verified fixed-batch parity (task14): at base/near-base weights RLinf ≈ reference.
2. **From step ~15 the reference drops steeply (to ≈0.09 by step 50) while RLinf stays flat
   at ≈0.24.** At the same tiny LR (~1e-6 at step 50), the reference reduces its loss ~2.7×;
   RLinf does not reduce at all. This is a real divergence in the optimization/training
   dynamics, not a forward-pass divergence.
3. **RLinf shows period-8 loss spikes** (steps 19, 27, 35, 43, 51 ≈ 0.34–0.39) absent from
   the reference — a structured signal (the run uses 8 GPUs and `num_workers=8`), the
   strongest lead toward a data-streaming / sharding cause.

## Ruled out / verified

- **Freeze filter**: ruled out. The reference `get_freeze_filter()` returns `nnx.Nothing` for
  `gemma_2b` + `gemma_300m` (no LoRA) — it freezes nothing; RLinf also trains all params.
- **Forward / weights / recipe config / LR logging**: verified identical/aligned in prior
  rounds (task12 audit, task14 fixed-batch parity, task13 LR-logging fix). step-0 agreement
  (0.229 vs 0.246) reconfirms the forward.

## Mechanism — PROVEN divergence, mechanism NOT yet proven (next round)

The divergence is proven and reproducible; the *mechanism* is not yet proven (stating a
hypothesis as fact is exactly the R14 trap). Candidate mechanisms, to be localized next round
(strongest first):

1. **Data streaming coverage/order** (period-8 spikes): does RLinf's rank/worker-sharded
   keyframe-chunk streaming present the same unique-frame coverage and order as the reference
   loader? If RLinf streams more/different frames per step (no memorization), the loss would
   stay flat while the reference (memorizing a smaller repeated set) drops. Compare the
   streamed frame sequences/coverage across the two loaders.
2. **Gradient / optimizer-update path under FSDP** (grad reduction, grad-scaler, clip, bf16
   gradient handling): confirm RLinf's applied updates reduce the loss the way the reference's
   do (e.g. run both for N steps from identical weights on an identical data stream).
3. **The reference's own fast-drop mechanism**: confirm the reference's 0.24→0.09 drop is real
   learning (not a logging/data-memorization artifact) before treating it as the target.

## Next round (blocking)

Localize the mechanism to a PROVEN cause (start with the streamed-frame coverage/order and the
period-8 spike), fix it, and re-run the first-50-step comparison until it satisfies DEC-1 (b).
Only then is task15 met; task16 (advisory ~1h trend) and task18 (final AC-13) follow.
