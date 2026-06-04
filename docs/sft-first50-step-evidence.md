# BEHAVIOR SFT First-50-Step Training Evidence (task15, AC-11 / DEC-1 (b))

GPU run-evidence (DEC-5) for the `use_skill:false` first-50-step training-loss gate.

**Verdict (Round 21): the flat-loss MECHANISM is fixed and proven, but task15's strict DEC-1 (b)
gate is NOT met (29/50).** The first-50-step flat-vs-dropping divergence was caused by RLinf
training the model in **pure bf16** — the `openpi_pytorch` factory cast the fp32-loaded training
weights to bf16 *before* FSDP wrapped them, so AdamW updated **bf16 master weights** and the tiny
warmup-LR updates (~1e-6, below the bf16 ULP ≈0.0078 near 1.0) were lost to rounding → the loss
stayed flat. The R20 fix keeps **fp32 master weights** for training (FSDP MixedPrecision casts to
bf16 only for compute, matching the reference recipe), and is now PROVEN by a committed mechanism
probe (`docs/evidence/r21_master_dtype_probe.{csv,json}`: on the real model, fixed batch, 50 AdamW
steps — bf16 master changes 0.86 % of params and the loss is flat 0.275→0.246; fp32 master changes
79.9 % and the loss descends 0.275→0.101).

After the fix the first-50 loss **descends 0.226 → 0.046** (was flat ≈0.24). Evaluated against the
DEC-1 (b) protocol **as written** (per-step `|Δ| ≤ 0.03` OR within ±2σ of the reference's *step-to-
step variation*), the correct band is `max(0.03, 2·σ_step)` with σ_step = std of the reference's
consecutive first-differences = **0.0094 → 2σ = 0.019 < 0.03**, so the band reduces to `|Δ| ≤ 0.03`
and the count is **29/50** (`docs/evidence/r21_production_band_corrected.csv`). The R20 "46/50"
used the WRONG band (±2σ of the reference's *absolute-value* series, ≈0.11, inflated by the
descent) and is corrected here. **task15's hard gate is therefore not met** (30/50 in the latest
run). The residual (RLinf descends to ≈0.048 vs the reference ≈0.090) has been localized against the
ACTUAL reference path (`openpi.models_pytorch_new`): augmentation, noise/time distribution,
autocast, and `reduce_dtype`/`buffer_dtype` are all **ruled out** (the `reduce_dtype`/`buffer_dtype`
mismatch was a real recipe divergence, now FIXED, but not causal). **R24 ran the external reference
trainer twice (seeds 42, 123)** and found the reference is highly reproducible (run-to-run spread
~0.004; both reach ~0.09 at step 49) while RLinf sits OUTSIDE that envelope (1/50) — so RLinf's
faster descent is a **real systematic divergence, NOT benign run-to-run noise** (the R23 RNG
hypothesis is REFUTED; no pass-rule is justified). The one untested difference is `torch.compile`
(reference uses it, RLinf is eager) — the next localization step. The earlier R20 "RNG/aggregation"
and R21 "missing-augmentation" attributions were both wrong and are
retracted. The residual is benign and not localized to a correctness bug; no DEC-1 (b) pass-rule is
accepted yet. See **"Round 20 / 21 / 22"** below. The R16 run that follows is retained as the
pre-fix baseline.

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

**The full first-50 scalar table is committed at `docs/evidence/r16_first50_losses.csv`**
(columns: `step, rlinf_loss, ref_loss, abs_delta, within_0.03, rlinf_lr, rlinf_grad_norm`).
Representative rows (RLinf | ref | |Δ|): step0 0.229|0.246|0.017; step10 0.218|0.248|0.031;
step20 0.239|0.167|0.072; step30 0.236|0.133|0.103; step40 0.257|0.111|0.146; step49
0.220|0.090|0.130.

## Reproducibility / DEC-5 artifact

- **RLinf scalars** extracted from the tensorboard event file
  `/mnt/public/xzxuan/tmp/r16_sft_results/tensorboard/events.out.tfevents.*` via
  `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` (tags
  `train/loss`, `train/learning_rate`, `train/grad_norm`), step 0..49 → committed CSV.
- **Reference loss** extracted from the reference log
  `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log`
  by regex `Step (\d+):.*loss=([0-9.]+)`, step 0..49.
- **Run output dir**: `/mnt/public/xzxuan/tmp/r16_sft_results` (tensorboard under `tensorboard/`).
- **Artifact hashes** (full-file sha256[:16]): `pi05_base_pytorch_new/model.safetensors`
  `f6391204c480d6c5`; task-0000 `norm_stats.json` `d66ed16830a98f90`;
  `paligemma_tokenizer.model` `8986bb4f423f07f8`.
- **Verdict**: 12/50 within `|Δ| ≤ 0.03`; DEC-1 (b) NOT met.

## Findings

1. **Steps 0–14 track the reference closely** (mostly `|Δ| < 0.04`) — consistent with the
   verified fixed-batch parity (task14): at base/near-base weights RLinf ≈ reference.
2. **From step ~15 the reference drops steeply (to ≈0.09 by step 50) while RLinf stays flat
   at ≈0.24.** At the same tiny LR (~1e-6 at step 50), the reference reduces its loss ~2.7×;
   RLinf does not reduce at all. This is a real divergence in the optimization/training
   dynamics, not a forward-pass divergence.
3. **RLinf shows period-8 loss spikes** (steps 19, 27, 35, 43, 51 ≈ 0.34–0.39) absent from
   the reference — a structured signal (the run uses 8 GPUs and `num_workers=8`). It reflects
   the contiguous-chunk streaming (each batch from one worker), but is NOT the cause of the
   flat-vs-dropping divergence (see the Round 17 verification below).

## Ruled out / verified

- **Freeze filter**: ruled out. The reference `get_freeze_filter()` returns `nnx.Nothing` for
  `gemma_2b` + `gemma_300m` (no LoRA) — it freezes nothing; RLinf also trains all params.
- **Forward / weights / recipe config / LR logging**: verified identical/aligned in prior
  rounds (task12 audit, task14 fixed-batch parity, task13 LR-logging fix). step-0 agreement
  (0.229 vs 0.246) reconfirms the forward.

## Round 17–19 — matched-topology + exact per-sample stream comparison; data-stream RULED OUT

### Observed facts (committed under `docs/evidence/`)
**RLinf single-process stream is contiguous** (R17 observation): the RLinf loader streams long
contiguous runs of consecutive frames within a keyframe chunk.

**Matched 8-rank / 8-worker stream identity** (R18 — driving the REAL
`BehaviorSftDataset._select_streaming_chunk` for every `(rank, worker)`; reference via the
subprocess dumper `tests/unit_tests/_ref_stream_dump.py` on the reference dataset):

- `docs/evidence/r18_rlinf_stream_coverage.csv`: RLinf per-step unique-frame coverage = **256**
  for every step (rank-folding → 8 ranks disjoint).
- `docs/evidence/r18_ref_stream_coverage.csv`: reference per-step unique-frame coverage = **32**
  for every step (rank-independent → all 8 ranks identical). Both loaders compute the same 1825
  keyframe chunks.
- `docs/evidence/r18_{rlinf,ref}_stream_sample.csv`: exact per-sample identities differ — e.g.
  RLinf `(rank 0, worker 0)` starts at episode 2160 chunk `[362366, 362616]`, reference
  `(worker 0)` starts at episode 1080 chunk `[229063, 229313]` (partition strides 64 vs 8).

So the RLinf and reference production streams **DIFFER** under the matched topology: RLinf folds
the distributed rank into the partition (`partition_chunk_indices` / `global_worker_id`) → 256
unique frames/step; the reference does not (`range(worker_id, n, num_workers)`, seed `+
worker_id`) → 32 unique frames/step.

### Exact per-sample stream identity (R19) — rank-independent RLinf == reference
Coverage equality (256 vs 32) is weaker than stream identity, so R19 added the EXACT per-sample
comparison. `tools/sft_stream_identity_probe.py` (RLinf) and `tests/unit_tests/_ref_stream_dump.py`
(reference) dump, for the first 50 global steps, every accepted sample's full identity via the
REAL helpers (`_select_streaming_chunk`, `_get_query_indices`, `_get_fine_grained_task`):
`episode_index, frame_index, chunk tuple, action_query_start/end, action_is_pad, prompt`.

- `docs/evidence/r19_identity_comparison.md` + `r19_{rlinf,ref}_identity_hashes.json`: the
  per-worker sha256 over the full first-50-step identity stream is **identical for all 8 workers**
  between RLinf-made-rank-independent and the reference (e.g. worker 0 `270aec08ec65` on both).
  So RLinf with the rank-independent partition emits the EXACT same frames, keyframe chunks,
  action query windows, and prompts as the reference loader — not merely the same coverage count.
- Production RLinf (rank-folding) differs (256 vs 32 unique/step; per-`(rank0,worker0)` sha256
  differs from the reference); `r19_rlinf_perworker_identity.csv` / `r19_ref_perworker_identity.csv`
  are the human-auditable per-sample samples.

### The data-stream difference is NOT causal (auditable rerun)
`docs/evidence/r17_rank_independent_rerun.md` (the exact source override + command + output path)
and `docs/evidence/r17_rank_independent_first50_losses.csv` record an 8-GPU rerun with RLinf's
partition made rank-independent — i.e. emitting the EXACT reference stream identity (above). The
first-50 loss stayed **flat** (mean 0.234, step0 0.226 → step49 0.210, **12/50** within
`|Δ| ≤ 0.03` — unchanged from the production R16 run). So feeding RLinf the reference's exact data
stream (frames, action windows, prompts) does **not** fix the flat-vs-dropping loss. The patch was
reverted (no committed `rlinf/` source change). **Conclusion (supported by the committed
exact-identity artifacts): the data stream is RULED OUT as the cause.** AC-6's rank-aware sharding
is retained.

### Narrowed cause (next round)
With the forward verified identical (task14 / R15 same-batch parity 0.0024), the recipe aligned
(task12), and the data stream ruled out by committed artifacts, the divergence is in the
**optimizer / gradient / weight-update path under FSDP** — gradient reduction, the grad-scaler /
bf16 gradient handling, gradient clipping, weight decay, or the FSDP mixed-precision update. On
identical data + identical forward, the reference's updates reduce the loss and RLinf's do not.

## Round 20 / 21 — bf16 → fp32 master weights (mechanism proven; strict gate not met)

### Proven mechanism (the optimizer/weight-update divergence)
`pi0_config.create()` builds the Pi0 params in **fp32** (`nn.Linear` / `nn.Parameter` use the
default torch dtype; `embed_dtype` only casts *activations*), and the training checkpoint is
loaded in fp32 (`expected_dtype = torch.float32`, validated). RLinf's FSDP1 wrap
(`rlinf/hybrid_engines/fsdp/strategy/fsdp.py::wrap_model`) applies
`MixedPrecision(param_dtype=bf16, …)` — byte-for-byte the reference's `apply_fsdp1`
(`FSDP1(..., mixed_precision=…, use_orig_params=True, FULL_SHARD)`). The **sole** divergence was
in the model factory (`rlinf/models/embodiment/openpi_pytorch/__init__.py::get_model`), which did
`model = model.to(bf16)` for training **before** FSDP wrapped it. That collapsed the fp32 master to
bf16, so `MixedPrecision(param_dtype=bf16)` became a no-op and AdamW updated bf16 weights. At the
warmup LR the per-step update (~1e-6) is below the bf16 ULP near 1.0 (≈0.0078) and is lost to
rounding.

- **Committed mechanism probe** (`tools/sft_master_dtype_probe.py` →
  `docs/evidence/r21_master_dtype_probe.{csv,json}`; single-GPU, fixed batch, 50 AdamW steps at the
  openpi_cosine warmup LR, identical noise/time, bf16 compute on BOTH arms): the **bf16-master** arm
  changes only **0.86 % of params** and the loss is flat (0.275 → 0.246); the **fp32-master** arm
  (fp32 storage + bf16 compute, the production fix) changes **79.9 %** of params (delta-L1 124×
  larger) and the loss descends (0.275 → 0.101). The only difference between the arms is the master
  dtype — proving the tiny warmup-LR updates vanish under bf16 rounding and accumulate under fp32.

### The fix
`get_model` no longer downcasts the training model: for `load_for_training=True` the fp32 master
is **kept** and FSDP MixedPrecision casts to bf16 only for the forward/backward (eval is unchanged
— still bf16-strict). This matches the reference recipe exactly (fp32 load + FSDP1 MixedPrecision).

### Production re-run (8 GPUs, fix applied) — descends, but 29/50 (strict gate not met)
Same command/config as the R16 run (`runner.logger.log_path=/mnt/public/xzxuan/tmp/r20_sft_results`).
RLinf's first-50 loss now **descends 0.226 → 0.046** (first-50 mean 0.151 vs reference 0.168),
tracking the reference's 0.246 → 0.090 — the flat-loss MECHANISM is fixed. Evaluated against the
DEC-1 (b) protocol as written (per-step `|Δ| ≤ 0.03` OR within ±2σ of the reference's step-to-step
variation), σ_step = std of the reference's consecutive first-differences = **0.0094**, so
2σ_step = 0.019 < 0.03 and the band is `|Δ| ≤ 0.03`: **29/50** (was 12/50 pre-fix). Committed
scalars: `docs/evidence/r20_first50_losses.csv` and the corrected band
`docs/evidence/r21_production_band_corrected.csv`. **The R20 "46/50" used the WRONG band**
(±2σ of the reference's *absolute-value* series ≈0.11, inflated by the descent) — corrected here.
**The hard gate (first-50 within band) is NOT met; task15 is not verified.**

### Controlled rank-independent comparison (rank-folding is NOT the residual)
The 4 band-outliers are RLinf descending *faster/lower* than the reference (a benign direction),
not flat. To attribute that residual without overclaiming, a second 8-GPU run applied the fix
**and** the temporary rank-independent partition (R19's exact-reference stream, 32 unique/step;
patch reverted after the run). Its curve is **near-identical to production** step-by-step (e.g.
step 37 0.075 vs 0.079; the step-19 spike 0.313 vs 0.319; same 29/50 under the corrected band).
Committed: `docs/evidence/r20_control_rank_independent_first50_losses.csv` +
`docs/evidence/r21_control_band_corrected.csv`. So AC-6 rank-folding (256 vs 32 unique frames/step)
makes **no meaningful difference** to the first-50 loss — the residual gap with the reference is
**not** the data stream (already ruled out exactly in R19, and again here).

### Residual localization (R22) — augmentation & noise/time RULED OUT against the real reference
**Correction:** the R21 version of this section claimed the residual (RLinf ≈0.046 vs reference
≈0.090) was a missing-augmentation recipe divergence. That was based on the WRONG reference module
(`openpi.models_pytorch`). The actual reference run imports `openpi.models_pytorch_new`, whose
`preprocess_observation` is **byte-identical to RLinf's** and gates augmentation on `rng`; the
training call passes `rng=None` → no augmentation on either side. **Augmentation is RULED OUT**
(`docs/evidence/r21_augmentation_comparison.md`, now corrected).

Localizing the remaining ~21 first-50 outlier steps against the real `models_pytorch_new` path
(read-diff; starting from `docs/evidence/r21_production_band_corrected.csv`):

| Axis | Reference (`models_pytorch_new` / `train_pytorch_new.py`) | RLinf | Status |
|------|-----------------------------------------------------------|-------|--------|
| augmentation | gated on `rng`; training `rng=None` → deterministic crop only | byte-identical; `rng=None` | RULED OUT |
| noise/time distribution | `Beta(1.5,1.0)·0.999+0.001` (internal, or `_make_noise_time` Dirichlet([1.5,1.0])) | vendored `Beta(1.5,1.0)·0.999+0.001` | RULED OUT (same distribution) |
| autocast | `use_autocast=False` → uses FSDP1 MixedPrecision mp_policy, no `torch.amp.autocast` | amp disabled (`amp_autocast.enabled=False`) | RULED OUT (both off) |
| FSDP `param_dtype` | bf16 (`init_model` fsdp1 branch) | bf16 | MATCH |
| **FSDP `reduce_dtype`** | **fp32** (`MixedPrecision(param_dtype=bf16, reduce_dtype=torch.float32)`, `train_pytorch_new.py:298-300`) | **bf16** (`behavior_pi05_vla.yaml` set all three to `${precision}`) | **DIVERGENCE — FIXED R22** |
| **FSDP `buffer_dtype`** | unset → fp32 (buffers not cast) | bf16 | **DIVERGENCE — FIXED R22** |

**Proven divergence FIXED (R22), but it is NOT the residual cause.** The reference's FSDP1
MixedPrecision reduces gradients in **fp32** (`reduce_dtype=torch.float32`) and leaves buffers fp32,
while the BEHAVIOR SFT config set `reduce_dtype=buffer_dtype=bf16` — so RLinf's multi-rank gradient
all-reduce was rounded to bf16. (RLinf's other SFT configs, e.g. `qwen3_vl_sft_vlm.yaml`, already
use `reduce_dtype: fp32`, so the BEHAVIOR config was the outlier.) Fixed in
`examples/sft/config/behavior_pi05_vla.yaml` (`reduce_dtype: fp32`, `buffer_dtype: fp32`) — a
correct recipe alignment regardless. The 8-GPU first-50 re-run with the fix
(`docs/evidence/r22_reduce_dtype_first50_losses.csv`) is **near-identical to the bf16-reduce run
step-by-step** (step 49 0.048 vs 0.046; **30/50** within `|Δ| ≤ 0.03` vs 29/50 — RNG-level), so
the bf16 gradient reduction was **not** the residual. The remaining axes (logged-loss aggregation,
LR, clip norm, optimizer state) were aligned by the task12 recipe audit.

### Residual: status after R22
After ruling out augmentation, noise/time distribution, autocast, `reduce_dtype`/`buffer_dtype`, the
data stream (R19 exact identity + the R20 rank-independent control), the forward (R15 same-batch
parity), and the optimizer master dtype (R20 fix), RLinf's first-50 loss **descends and tracks the
reference's trend** but settles ≈0.048 vs the reference ≈0.090 — RLinf descends **at least as
fast**. The **logged-loss aggregation is identical**, not a residual source: the reference
(`train_pytorch_new.py:528-529`) computes `torch.stack(loss_acc).sum()` then
`dist.all_reduce(loss_acc, op=ReduceOp.AVG)` and logs `loss_acc.item()`, i.e. the AVG-all-reduced
loss across all ranks — exactly as RLinf's worker does (the earlier "reference logs rank-0's
32-sample loss" claim was wrong and is retracted; this was already established in the task12/13
recipe audit).

### R24 — reference-repeat variance packet: the residual is a REAL divergence, not RNG
To test whether RLinf's faster descent is just run-to-run noise (the R23 "noise/time RNG
realization" hypothesis), the **external reference trainer was run twice** for the first 50 steps
(its own venv, the documented `pi05_b1k-task0000_sft_pytorch_mixed` recipe, fsdp1/USE_AUTOCAST=0/
USE_CONSISTENT=0, 8 GPUs): seed 42 and seed 123. Committed:
`docs/evidence/r24_ref_seed{42,123}_first50.csv` + `r24_reference_variance_packet.json` (commands,
hashes, paths). Result:
- The reference is **highly reproducible**: seed-42 step49=0.0903 (mean 0.1675) — exactly the
  committed R16 log; seed-123 step49=0.0928 (mean 0.1685). Run-to-run spread across the three
  reference curves is **~0.0042 mean, ~0.0024 at step 49** — a NARROW envelope.
- RLinf-postfix (step49 ≈0.048, mean 0.151) lies **OUTSIDE** that envelope: only **1/50** steps
  inside the per-step [min,max] reference band (12/50 with ±0.01 pad). The pre-fix flat curve is also
  outside (2/50), in the opposite direction.

So the **"noise/time RNG realization" hypothesis is REFUTED**: the reference's run-to-run variance
is tiny, and RLinf descending to ≈0.048 vs the reference ≈0.090 is a **real, systematic divergence**
(RLinf learns faster), NOT benign run-to-run noise. **No variance-backed DEC-1 (b) pass-rule is
justified**, and the R23 proposal is withdrawn. task15 remains NOT met with a real residual.

**Next localization (the one untested difference):** the reference run (committed log + both
replays) uses **`torch.compile`** (`TORCH_COMPILE_MODE=default`), while RLinf trains eager. A
reference EAGER run (`TORCH_COMPILE_MODE=null`) to test whether compile causes the slower descent
was attempted this round but did not complete (torchrun launch instability after the repeated runs);
it is the next step. (Caveat from `r24_reference_variance_packet.json`: compile's triton `.so`
cannot load from the `/mnt/public` network FS, so `TMPDIR` was pointed at `/dev/shm` — exec-allowed
tmpfs, not `/tmp`/overlay — for the runs.)

### DEC-5 artifact (this round)
- **RLinf scalars** (both runs) from the tensorboard event files under
  `/mnt/public/xzxuan/tmp/r20_sft_results/tensorboard/` and `…/r20_control_sft_results/tensorboard/`
  via `EventAccumulator` (tags `train/loss`, `train/learning_rate`, `train/grad_norm`), step 0..49.
- **Reference loss**: reused from the committed `docs/evidence/r16_first50_losses.csv` `ref_loss`
  column (same reference log as R16).
- **Artifact hashes** (full-file sha256[:16]): `pi05_base_pytorch_new/model.safetensors`
  `f6391204c480d6c5`; task-0000 `norm_stats.json` `d66ed16830a98f90`;
  `paligemma_tokenizer.model` `8986bb4f423f07f8` (unchanged from R16 — same inputs).
- **Mechanism probe** (R21): `tools/sft_master_dtype_probe.py` →
  `docs/evidence/r21_master_dtype_probe.{csv,json}` (manifest records the command, hashes, param
  dtype, changed-param fraction, delta-L1, and loss for both the bf16-master and fp32-master arms).
- **Corrected band** (R21): `docs/evidence/r21_{production,control}_band_corrected.csv` (σ_step =
  0.0094; band `max(0.03, 2σ_step)` = 0.03; 29/50). **Augmentation comparison**:
  `docs/evidence/r21_augmentation_comparison.md`.
- **Reference-repeat variance packet** (R24): `docs/evidence/r24_ref_seed{42,123}_first50.csv` +
  `docs/evidence/r24_reference_variance_packet.json` (the external reference trainer run twice;
  commands, venv, recipe, hashes, the run-to-run envelope, and the RLinf-outside-envelope verdict).

### task15 status and next step (R24)
The flat-loss MECHANISM is fixed and proven (probe above), but **task15's hard DEC-1 (b) gate is
NOT met** (30/50 in the latest run; 29/50 in the R20 run). The residual has been localized against
the real `models_pytorch_new` reference path: augmentation, noise/time distribution, autocast, and
`reduce_dtype`/`buffer_dtype` are all **ruled out** — the `reduce_dtype`/`buffer_dtype` mismatch was
a genuine recipe divergence (now FIXED) but the re-run shows it is not the residual. There is **no
augmentation alignment to do** (the reference and RLinf both train with `rng=None`). **R24's
reference-repeat variance packet (seeds 42 + 123) REFUTED the benign-RNG hypothesis**: the reference
is highly reproducible (run-to-run spread ~0.004) and RLinf is OUTSIDE that envelope (1/50) — RLinf's
faster descent is a real systematic divergence. So **no DEC-1 (b) pass-rule is justified** (the R23
proposal is withdrawn), and task15 stays active on a real residual. The next localization step is to
test **`torch.compile`** (the reference uses it; RLinf is eager) as the cause of the slower reference
descent — a reference EAGER run, attempted in R24 but not completed (torchrun launch instability). If
compile is the cause, RLinf may already match the reference's eager behavior; otherwise localize
further or rerun toward the original 50/50 gate. task16 (advisory ~1 h trend) and task18 (final
AC-13) remain blocked on task15's strict verification.
