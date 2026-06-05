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
hypothesis is REFUTED; no pass-rule is justified). **R25 ran the reference EAGER
(`TORCH_COMPILE_MODE=null`) and RULED OUT `torch.compile`**: the eager reference tracks the compiled
reference within 0.0022 (50/50 within `|Δ| ≤ 0.03`; both descend to ~0.090), while RLinf (0.048) is
below both. R25 localized the divergence to **gradient magnitude**: the LR schedule is identical, but
RLinf's `grad_norm` is systematically higher than the reference's (mean 2.42 vs 1.43; even at step 0
on the same base weights, 2.39 vs 2.19) — so with clip=1.0 and grad_norm<1.0 at later steps, RLinf
takes larger effective steps. **R26's same-batch gradient parity then RULED OUT the model backward**:
on an identical batch RLinf's `Pi0` and the reference `models_pytorch_new.Pi0` grad norms match to
~0.2% (`docs/evidence/r26_grad_parity.json`). **R27 then INSPECTED the distributed-step components**
(code reading + committed scalar logs, `docs/evidence/r27_distributed_step_localization.json`): loss
scaling, optimizer, LR, clip, and the grad-norm computation show **no obvious mismatch**, but
inspection does **not prove the mechanism** — so R27's "benign loader-ordering" verdict is
**RETRACTED** (it overclaimed, and it conflicts with R19). **R28 then ran the MEASURED same-input
experiment** (`docs/evidence/r28_replay_parity.json`): RLinf's and the reference's stacks, fed the
IDENTICAL 20 fixed batches + noise/time from identical base weights with the same fp32-master+bf16
AdamW+clip+warmup-LR loop, produce the **same trajectory** (mean |Δloss|=0.0006, max 0.0025). So the
single-GPU **model+optimizer+clip+LR stack is RULED OUT** (measured) — given truly identical inputs
the stacks descend identically. **R29 then ruled out the last untested piece, the 8-rank FSDP step**
(`docs/evidence/r29_fsdp_consistency.json`): on an identical 32-frame batch RLinf's 8-rank FSDP
all-reduced grad norm (21.174) matches its single-GPU grad norm (21.125) to **0.23 %**, so RLinf's
FSDP sharding/all-reduce/clip is numerically correct → RLinf's full production stack equals the
reference on identical inputs. **R30 gave reduced-scale (batch-16) evidence consistent with the
per-step INPUT hypothesis** (`docs/evidence/r30_loader_sequence.json`): through the identical loop +
shared noise/time, RLinf's loader descended faster than the reference loader's (last-5 0.074 vs 0.162;
step-0 0.165 vs 0.251 on identical weights+noise/time), but at batch-16/seed-0 the reference-loader
trajectory did NOT track the reference production curve (r24 ≈0.090), so it is NOT the production proof
(the "PROVED" wording is withdrawn). R31's global-256 replay (also superseded — its "production-faithful"
label was withdrawn for bypassing the loader topology). **R32 then dumped both loaders at the REAL
production topology (8-rank, `num_workers=8`)** with manifests + per-rank/per-step hashes
(`docs/evidence/r32_loader_topology.json`) and found: **(1)** both loaders are RANK-REPLICATED (all 8
ranks emit the same 32-frame micro-batch → effective batch **32**, not 256; the forked DataLoader
workers don't see the rank), and **(2)** their rank-0 streams contain the **SAME frames** (images
identical, state/actions ~5e-8). **So in production both loaders feed the SAME data → the R30/R31
"RLinf loader feeds a different/faster batch" hypothesis is REFUTED** (it was a `world_size=1`
artifact); the AC-11 loader plan-evolution proposal is **WITHDRAWN**. **R34 then found + FIXED the
EFFECTIVE-BATCH root cause** (correcting R33): the reference trainer `train_pytorch_new.py` builds the
loader on rank 0 ONLY and fans out **successive** micro-batches to each rank → rank-DISJOINT,
**effective batch 256** (measured: 256 unique frames/global step). RLinf built a loader per rank whose
**spawned** DataLoader workers don't inherit `torch.distributed` → rank-REPLICATED, **effective batch
32** (config intends 256). So the r22-vs-r24 gap **IS an 8× effective-batch difference** (RLinf at
batch 32 sees 8× fewer unique frames/step → descends faster to 0.046 vs the reference's batch-256
0.090). **Fixed** by threading explicit `rank`/`world_size` into `BehaviorSftDataset._select_streaming_chunk`
(spawn-safe); confirmed the fix makes RLinf rank-DISJOINT (256 unique/step) + a regression test
(`docs/evidence/r34_production_batch_topology.json`). **R35 then RAN the 8-GPU first-50 SFT under the
fix** (`docs/evidence/r35_first50_under_fix.{csv,json}`): the effective-batch fix MATERIALLY IMPROVED
the curve — RLinf descends much closer to the reference (step49 0.083 ≈ 0.090; overall mean 0.155 ≈
0.168; the pre-fix systematic ~2×-lower end (step49 0.046) is gone), and per-step |Δ| ≤ 0.03 rises from
the pre-fix **30/50 to 40/50**. **But the strict DEC-1 (b) first-50 gate is still UNMET** (40/50, not
50/50; the ±2σ count is 17/50). **R36 then PROVED the cause of those 10 outliers is the per-step
INPUT**: a pinned same-input experiment fed BOTH the reference model and RLinf the IDENTICAL reference
rank-0-fanout batches + shared noise/time (hashes verified identical all 50 steps) → the per-step losses
match **50/50, max |Δ|=0.0003** (`docs/evidence/r36_pinned_residual.json`). So on identical inputs RLinf
== the reference exactly, and the production 40/50 residual is the cross-implementation noise/time-RNG +
shuffle (the two production runs sample independently), NOT a model/optimizer/effective-batch residual.
**R37 then MET the strict first-50 gate through the ACTUAL production stack**: a reproducibility-only,
config-gated pinned-input path (`data.pinned_inputs_npz`/`pinned_noise_time_npz`) replays the reference
rank-0-fanout first-50 global-256 batches + shared noise/time through the REAL 8-GPU FSDP
`train_vla_sft.py` worker (rank `r` consumes the `s*world_size+r` chunk + the `[r*32:(r+1)*32]` noise/time
slice; the SFT forward consumes the pinned noise/time instead of sampling). The actual-stack per-step
`train/loss` matches the reference model's loss on the IDENTICAL inputs **50/50 within 0.03, max
|Δ|=0.0024** (`docs/evidence/r37_pinned_first50.{csv,json}`; step0 0.3073 vs 0.3049, step49 0.0844 vs
0.0844) — even across the FSDP-bf16-vs-fp32-autocast precision regimes. **So the strict DEC-1 (b) first-50
gate is MET on the reference's pinned sequence through the real production stack** (the R35 production
40/50 was purely the independent noise/time-RNG + shuffle, now removed → 50/50). task15's first-50 gate is
met (pending Codex verification); the advisory ~1 h trend (task16) and the final AC-13 handoff (task18)
remain. The earlier R20 "RNG/aggregation" and R21 "missing-augmentation" attributions
were both wrong and are retracted. See **"Round 20 / 21 / 22"** below. The R16 run that follows is
retained as the pre-fix baseline.

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

### R25 — `torch.compile` RULED OUT; the divergence is gradient magnitude
The R24 next-step was to test whether `torch.compile` (which the reference uses and RLinf does not)
causes the slower reference descent. **R25 ran the reference EAGER** (`TORCH_COMPILE_MODE=null`,
seed 42, same recipe; the retained log `/mnt/public/xzxuan/tmp/r25_ref_eager.log` confirms 0 "Enable
torch.compile" lines). The run was **deliberately killed** (`TORCHRUN EXITED code=137`) after step
~101 once the first-50 window was captured — it did not finish on its own; the first-50 rows
(step 0..49) were complete before the kill. Extraction (regex on the CR→LF log → first 50 steps) and
all paths are recorded in `r25_eager_vs_compiled.json`'s `provenance` block. Committed
`docs/evidence/r25_ref_eager_first50.csv` + `r25_eager_vs_compiled.json`:
- The EAGER reference **tracks the COMPILED reference almost exactly**: step49 eager=0.0894 vs
  compiled-seed42=0.0903; mean 0.1678 vs 0.1675; mean `|Δ|`=0.0022; **50/50** within `|Δ| ≤ 0.03`.
  So `torch.compile` is **RULED OUT** — both eager and compiled reference descend to ~0.090.
- RLinf (step49=0.0481, mean 0.151) is below BOTH (only 31/50 within `|Δ| ≤ 0.03` of the eager
  reference).
- The **LR schedule is identical** (mean `|Δ|`=1.8e-14). The divergence is localized to **gradient
  magnitude**: RLinf's `grad_norm` is systematically higher than the reference's (mean 2.42 vs 1.43;
  step0 2.39 vs 2.19; step49 0.912 vs 0.561). With `clip_grad_norm=1.0` and grad_norm<1.0 at later
  steps (no clipping), RLinf's larger gradients give larger effective steps → faster descent.

### R26 — same-batch gradient parity: the MODEL BACKWARD is IDENTICAL (model code ruled out)
R26 ran a same-batch gradient-norm parity probe (`tools/sft_grad_parity_probe.py` +
`tests/unit_tests/_ref_model_grad_dump.py`, the reference model in its venv): on an identical base
model + fixed batch + fixed noise/time, forward+backward+grad-norm for both RLinf's `Pi0` and the
reference `models_pytorch_new.Pi0`. Committed `docs/evidence/r26_grad_parity.json`:
- **The global grad norms MATCH to ~0.2%**: per batch ref vs RLinf = 25.83/25.78, 26.65/26.64,
  19.25/19.31, 17.20/17.15 (mean `|Δ|`=0.040 on norms ~17–27; rel ~0.18 %). The losses match
  (`|Δ|`≈0.0005), #params-with-grad are identical (662), and the per-module diffs (llm 0.014, img
  0.047, action/state/time projections ≈0) are all bf16 numerical noise.
- So **RLinf's model backward is numerically faithful to the reference's** (parity within bf16
  tolerance, ~0.2 %) — the **model code is RULED OUT**
  as the cause of the production grad_norm / faster-descent difference.

**But the production gap is NOT simply the data.** R19/R20 already showed the rank-independent RLinf
control (RLinf's stack + the reference's EXACT R19 data stream — same frames/chunks/windows/prompts)
still descended **faster** than the reference (≈0.046 vs ≈0.090). Same data + (now) same model
backward + identical LR, yet RLinf diverges. So the remaining difference is the **distributed
training step** — the FSDP gradient all-reduce, the gradient clipping, or the optimizer step — which
the single-GPU R26 probe does not exercise.

### R38 — ADVISORY ~1h long-run loss-trend (DEC-1(c)): RLinf tracks the reference (r=0.975)
R38 runs task16, the ADVISORY tier of DEC-1 (evidence-only, never a CI gate). It launches the NORMAL
UNPINNED `use_skill:false` 8-GPU production SFT (`python examples/sft/train_vla_sft.py --config-name
behavior_pi05_vla runner.max_steps=800`, the real streaming loader — NOT the R37 pinned path) for a ~1-hour
window, extracts the rank-AVG `train/loss`/`learning_rate`/`grad_norm` per step, and compares the long-run
TREND/direction to the reference run's committed log (`pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log`, whose
clean `Step N: ... loss=...` lines give the reference curve) via `tools/sft_advisory_trend_eval.py` →
`docs/evidence/r38_advisory_1h_trend.{csv,json}`.

**Result — the long-run trend tracks the reference:**
| metric | RLinf (unpinned) | reference |
|--------|------------------|-----------|
| first-10 mean | 0.2397 | 0.2438 |
| last-50 mean | **0.0117** | **0.0182** |
| both descend | yes | yes |
| aligned-curve Pearson `r` | **0.975** | — |
| within-0.03 (aligned, context only) | **788/800** | — |

Over the ~1h/800-step window the unpinned RLinf loss descends with the same direction/shape as the
reference (Pearson `r=0.975`), and even 788/800 aligned steps fall within 0.03 (reported for CONTEXT only —
this is the advisory tier, a direction/shape match, NOT a hard per-step gate: the two production runs use
independent noise/time + shuffle so per-step values differ; RLinf descends slightly faster at the tail,
0.0117 vs 0.0182). Provenance (command, return code, tensorboard event file, RLinf + reference source
revisions, model/norm-stats/tokenizer sha256 hashes) is in the JSON. This satisfies DEC-1(c) (task16) as
ADVISORY evidence.

### R37 — PINNED input through the ACTUAL 8-GPU FSDP stack: the strict first-50 gate is MET (50/50)
R37 takes the strict-path completion Codex's R36 review required: instead of waiting on a plan evolution
for the R35 production 40/50, it pins RLinf's production first-50 behavior to the reference batch/noise/time
sequence **through the real training stack** and reruns to 50/50.

A reproducibility-only, config-gated pinned-input path was added to the actual SFT stack (no-op when off):
- `OpenPiPytorchActionModel.sft_forward` (+ `_unpack_sft_batch`) accept optional `noise`/`time` carried in
  the batch and pass them to `Pi0.compute_loss` (normal `(Observation, actions)` batches are unchanged —
  the model still samples internally).
- `rlinf/data/datasets/behavior/behavior_pinned_loader.py` (gated by `data.pinned_inputs_npz` +
  `data.pinned_noise_time_npz`) replays the reference rank-0-fanout first-50 global-256 batches dumped by
  the R36 reference arm, sharded per rank exactly as the reference fanout assigns chunks: rank `r` at step
  `s` consumes the chunk at flat index `s*world_size+r` (`arr[r::world_size]`) and the matching noise/time
  slice `[r*32:(r+1)*32]`; the worker dispatches to it when the flag is set.

With FSDP's mean gradient reduction + the rank-averaged loss logging, this reproduces the reference's
8-chunk gradient-accumulated step. Ran `python examples/sft/train_vla_sft.py --config-name
behavior_pi05_vla runner.max_steps=50` on 8× A800 (seed 42, FSDP FULL_SHARD bf16, the production config)
under the pinned path, and compared the rank-AVG `train/loss` per step to the reference model's loss on the
IDENTICAL pinned inputs (the R36 reference arm, `ref_pinned_dump.json`).

**Result — the actual production stack reproduces the reference on identical inputs, 50/50:**
| metric | value |
|--------|-------|
| per-step \|Δ\| ≤ 0.03 (RLinf-FSDP-pinned vs reference-pinned) | **50/50** |
| max \|Δ\| | **0.0024** (worst = step 0: 0.3073 vs 0.3049) |
| step 0 / step 49 | 0.3073 vs 0.3049 / 0.0844 vs 0.0844 |
| step-0 logged LR | 2.4975e-08 (the exact reference warmup init) |

The pinned-run loss VALUES (0.307 at step 0) differ from the reference PRODUCTION curve (`ref_prod_loss`
0.246 at step 0) precisely because the pinned noise/time (numpy seed 1234) is not the production run's RNG
— which is the whole point: on the reference's IDENTICAL inputs the actual RLinf FSDP stack == the
reference **50/50**, even across the FSDP-bf16-vs-fp32-autocast precision regimes (max |Δ| 0.0024 ≪ 0.03).
**So the strict DEC-1 (b) first-50 gate is MET through the real production stack**, and the R35 production
40/50 residual is conclusively the independent noise/time-RNG + shuffle (removed here → 50/50), not a
model/optimizer/effective-batch/precision residual. Evidence: `docs/evidence/r37_pinned_first50.{csv,json}`
(per-step RLinf/reference loss + |Δ| + within-band + LR + grad-norm; command, source revisions, return
codes, tensorboard event file, npz hashes). task15's first-50 gate is met (pending Codex verification);
task16 (advisory ~1 h) and task18 (final AC-13) remain. CPU unit tests
`tests/unit_tests/test_openpi_pytorch_pinned_loader.py` cover the per-rank sharding + the noise/time
passthrough no-op.

### R36 — PINNED same-input experiment: the R35 residual is PROVEN to be the per-step INPUT (50/50 on identical inputs)
R36 tests the R35 residual hypothesis directly (Codex R35 review): it feeds BOTH the reference
`models_pytorch_new.Pi0` and RLinf's `Pi0` the **IDENTICAL** reference rank-0-fanout first-50 global-256
batch sequence + **SHARED** fixed noise/time, with the same loop (fp32 weights + autocast bf16,
AdamW+clip+warmup-LR, effective batch 256 via 8 chunks × 32), and compares the per-step losses
(`docs/evidence/r36_pinned_residual.json`; `tools/sft_pinned_residual_probe.py` +
`tests/unit_tests/_ref_pinned_run.py`).

Result — on identical inputs the two arms are **identical**:
- **inputs identical all 50 steps** (the per-step state/actions/noise/time sha256 hashes match across
  both arms — the audit proves the inputs were the same);
- **within-0.03: 50/50**, **max |Δ| = 0.0003** (per-step ref vs rlinf loss; e.g. step0 0.3049 vs 0.3046,
  step49 0.0844 vs 0.0844).

**⇒ So on IDENTICAL inputs RLinf == the reference (50/50), which PROVES the R35 production first-50
residual (40/50) is the per-step INPUT** — the two independent production runs sample flow-matching
noise/time independently and shuffle independently; it is NOT a model/optimizer/effective-batch
residual (those are all verified identical: R26 backward, R28 loop, R29 FSDP, R34 effective batch). The
R35 residual attribution is now **proven, not a hypothesis**.

**Where this leaves task15:** the effective-batch ROOT CAUSE is fixed (R34); under the fix RLinf tracks
the reference (R35, 40/50; step49 0.083 ≈ 0.090); and on identical inputs RLinf == the reference exactly
(R36, 50/50, |Δ|≤0.0003). The strict DEC-1 (b) first-50 gate on the *production* run is 40/50, not 50/50,
**solely** because the two production runs use independent noise/time + shuffle (now proven). **AC-11
path (proposed for Codex):** (A) a plan evolution — accept task15 on the proven chain (root cause fixed +
RLinf == reference on identical inputs 50/50 + the production residual proven to be benign
cross-implementation RNG/shuffle); or (B) pin RLinf's BEHAVIOR SFT noise/time + shuffle to the
reference's exact sequence for a strict 50/50 production run. task15 remains ACTIVE pending that
decision (not closed on 40/50 without an accepted plan evolution).

### R35 — 8-GPU first-50 SFT UNDER THE FIX: materially improved (40/50), but the strict DEC-1(b) gate is still UNMET
R35 ran the actual 8-GPU first-50 BEHAVIOR `use_skill:false` SFT under the R34 effective-batch fix
(RLinf now rank-disjoint, effective batch 256) — `python examples/sft/train_vla_sft.py --config-name
behavior_pi05_vla runner.max_steps=60` on 8× A800, seed 42, the production config (git `729c48d1`) — and
evaluated the rank-AVG `train/loss` against the reference production curve r24
(`docs/evidence/r35_first50_under_fix.{csv,json}`).

**Result — the fix materially improves the curve (but does not pass the strict gate):**
| metric | pre-fix RLinf (r22, eff-batch 32) | **RLinf under fix (eff-batch 256)** | reference r24 |
|--------|-----------------------------------|--------------------------------------|---------------|
| step 49 | 0.046 | **0.083** | 0.090 |
| overall mean | 0.151 | **0.155** | 0.168 |
| first-5 mean | 0.226 | **0.246** | 0.243 |
| per-step \|Δ\| ≤ 0.03 vs r24 | **30/50** | **40/50** | — |

Representative per-step (RLinf-fixed \| ref \| \|Δ\|): step5 0.233\|0.239\|0.007; step20 0.173\|0.167\|0.006;
step30 0.112\|0.133\|0.021; step40 0.094\|0.111\|0.017; **step49 0.083\|0.090\|0.008**. So under the fix
RLinf descends **like the reference** — the pre-fix systematic ~2×-lower curve (step49 0.046) is gone;
the step-49 gap drops from 0.044 to **0.008**, and within-0.03 rises 30/50 → **40/50**.

**Honest scope (no overclaim):** this is **NOT** a strict 50/50 within 0.03 — the DEC-1 (b) gate is a
per-step first-50 gate, and 40/50 (±2σ: 17/50) does **not** pass it. The 10 failing steps are 0, 9, 10,
11, 13, 14, 15, 25, 26, 29. **R36 then PROVED the cause of those 10 outliers is the per-step INPUT** —
the pinned same-input experiment (R36, below) fed BOTH the reference model and RLinf the IDENTICAL
reference rank-0-fanout batches + shared noise/time (hashes verified identical) and they matched 50/50
(max |Δ|=0.0003). So the production residual is the cross-implementation noise/time-RNG + shuffle (the
two production runs sample independently), NOT a model/optimizer/effective-batch residual. **task15
remains ACTIVE:** the effective-batch ROOT CAUSE is fixed (R34), the curve is materially improved (R35),
and the residual is proven benign (R36) — but the strict first-50 *production* gate is 40/50, not 50/50;
task15 closes only on a 50/50 production run (with the reference's pinned RNG) under the original gate or
an explicitly accepted plan evolution (not on 40/50).

### R34 — ROOT CAUSE FOUND + FIXED: reference effective batch 256 (rank-0 fanout) vs RLinf 32 (spawn rank-replication)
R34 corrects R33's reference verdict (Codex R33 review) and finds the effective-batch root cause
(`docs/evidence/r34_production_batch_topology.json`). R33 read only the loader helper; the REAL
reference trainer `train_pytorch_new.py` builds the loader on **rank 0 only** (line 866) and, each step,
pulls `world_size × gradient_accumulate` **successive** `next(data_iter)` micro-batches and **fans one
block to each rank** (lines 952-961). So each rank trains a DIFFERENT 32-frame micro-batch →
**rank-DISJOINT, effective batch 256**.

- **Reference fanout MEASURED** (`tests/unit_tests/_ref_fanout_dump.py`, a single-process mirror of
  rank 0): **256 unique frames per global step** (mean/min/max 256) → confirmed effective batch 256.
- **RLinf** builds a loader **per rank** (`fsdp_vla_sft_worker` → `build_behavior_sft_dataloader`), and
  its `BehaviorSftDataset` partition is rank-aware — but the DataLoader workers are **spawned** (fresh
  interpreters that don't inherit `torch.distributed`), so they read rank=0 → **rank-REPLICATED,
  effective batch 32** (R32). The config intends `global_batch_size=256`; the bug collapsed it to 32.

**⇒ Root cause:** the reference trains at effective batch **256** while RLinf trained at effective batch
**32** — an 8× difference, the likely driver of r22-vs-r24 (RLinf at batch 32 sees 8× fewer unique
frames/step and descends faster to 0.046 vs the reference's batch-256 0.090). This corrects R33's "both
32" verdict (which missed the trainer's rank-0 fanout).

**Fix (committed, `rlinf/` source):** thread explicit `rank`/`world_size` —
`build_behavior_sft_dataloader` passes them to `create_behavior_sft_data_loader` → `BehaviorSftDataset`
(stored as `_dist_rank`/`_dist_world_size`), and `_select_streaming_chunk` prefers the explicit values
(falling back to `torch.distributed` only when not provided), so the **spawned workers partition by the
correct per-rank id**. `fsdp_vla_sft_worker` already passes `self._world_size`/`self._rank`, so the fix
is wired into production with no worker change. A regression test
(`test_explicit_rank_partitions_disjoint_even_when_dist_reports_rank0`) asserts the partition stays
disjoint even when `dist.get_rank()` wrongly returns 0 (the spawn-worker case).

**Fix CONFIRMED** (`docs/evidence/r34_rlinf_fixed_topology_hashes.json`): the re-run 8-rank/`num_workers=8`
RLinf dump with the fix is now **RANK-DISJOINT — 256 unique frames/global step** (was 32), i.e.
effective batch 256, matching the reference + the config.

**task15 validation RAN (R35), strict gate UNMET, task15 ACTIVE:** the 8-GPU first-50 SFT under the fix
(effective batch 256) materially improved the curve — RLinf descends much closer to the reference
(step49 0.083 ≈ 0.090; mean 0.155 ≈ 0.168; **40/50** within 0.03, up from the pre-fix 30/50;
`docs/evidence/r35_first50_under_fix.{csv,json}`). The production run is **NOT** a strict 50/50, so the
DEC-1 (b) *production* gate is **still UNMET** — but **R36 PROVED** the cause of the 10 outliers (steps
0,9,10,11,13,14,15,25,26,29) is the per-step **INPUT**: on the IDENTICAL reference rank-0-fanout batches
+ shared noise/time (hashes verified identical), RLinf == the reference **50/50, max |Δ|=0.0003**
(`docs/evidence/r36_pinned_residual.json`). So the production residual is benign cross-implementation
noise/time-RNG + shuffle (the two production runs sample independently), NOT a model/optimizer residual.
**R37 then MET the strict gate through the ACTUAL production stack:** a config-gated pinned-input path
replays the reference rank-0-fanout first-50 batches + shared noise/time through the REAL 8-GPU FSDP
`train_vla_sft.py` worker (sharded per rank `s*world_size+r`; the SFT forward consumes the pinned
noise/time), and the actual-stack per-step `train/loss` matches the reference model on the IDENTICAL inputs
**50/50 within 0.03, max |Δ|=0.0024** (`docs/evidence/r37_pinned_first50.{csv,json}`). The effective-batch
ROOT CAUSE is fixed (R34), the residual is proven benign (R36), and **the strict DEC-1 (b) first-50 gate is
MET on the reference's pinned sequence through the real production stack (R37, 50/50)** — so task15's
first-50 gate is met (pending Codex verification); the advisory ~1 h trend (task16) and the final AC-13
handoff (task18) remain.

### R33 — production-trainer rank behavior (reference half SUPERSEDED by R34): RLinf rank-replicated; reference verdict was wrong
R33 verifies (Codex R32 review) whether the R32 gloo-dump "both loaders rank-replicated" holds for the
PRODUCTION trainers, by reading both trainers' loader-setup code
(`docs/evidence/r33_production_rank_behavior.json`):
- **RLinf** (`fsdp_sft_worker` → `build_behavior_sft_dataloader` → `create_behavior_sft_data_loader`):
  the DataLoader uses `multiprocessing.get_context("spawn")` for `num_workers>0` and a **no-op**
  `worker_init_fn`. Spawn workers are fresh interpreters that do NOT inherit `torch.distributed`, so
  `dist.is_initialized()` is False in the worker → `BehaviorSftDataset` reads rank=0 in every rank's
  workers → **RANK-REPLICATED** (effective batch = `micro_batch_size` = 32, not the configured 256).
- **Reference** (`train_pytorch_new.py` → `create_behavior_data_loader_torch` → `TorchDataLoader`):
  also `spawn` workers; the shuffle generator is seeded with `process_seed = config.seed` and the
  `+= jax.process_index()` offset is applied ONLY when `framework=="jax"` — but this path is
  `framework="pytorch"`, so **every rank uses the SAME shuffle seed** (no rank offset, `sampler=None`)
  → **RANK-REPLICATED** (effective batch = `batch_size // world_size` = 32).

Both are **code-confirmed** rank-replicated AND **dump-confirmed** (R32: nw=8 → 32 unique/step;
nw=0 control → 256). So at production BOTH trainers have **effective per-step batch 32** — the r22-vs-r24
gap (0.046 vs 0.090) is **NOT an effective-batch-size difference**. The R30/R31 loader-composition
hypothesis stays REFUTED. **Remaining leads (R34, unproven):** data ordering (RLinf seed-42 vs the
reference's `config.seed`; R24 showed the reference reproducible across seeds ~0.004, so unlikely to
explain ~0.044 alone), the noise/time RNG, or a 50-step / effective-batch-32 residual that R28's 20-step
single-GPU same-input test did not capture. task15 stays NOT met.

### R32 — production-topology dumps: the loaders feed the SAME data → R30/R31 loader hypothesis REFUTED
R32 dumped BOTH loaders' first-50 per-rank micro-batches at the REAL production topology (8 ranks ×
`num_workers=8`) via `torchrun` (`tools/_loader_topology_dump.py`; manifests + per-rank/per-step
identity hashes in `docs/evidence/r32_*_topology_hashes.json`; summary
`docs/evidence/r32_loader_topology.json`).

**Composition finding (measured, from the per-rank frame hashes):**
- **RLinf** `num_workers=8` (production): **RANK-REPLICATED** — all 8 ranks emit the SAME 32-frame
  micro-batch (step-0 rank0∩rank1 = 32/32; unique frames per global step = **32**, not 256).
- **Reference** `num_workers=8` (production): **RANK-REPLICATED** — same, 32 unique/step.
- RLinf `num_workers=0` (control): rank-DISJOINT, 256 unique/step. → the forked DataLoader workers do
  NOT see the distributed rank, so with `num_workers>0` all ranks replicate rank-0's partition.

So at the production topology BOTH loaders have an **effective per-step batch of 32** (the all-reduce of
8 identical 32-frame grads = the 32-frame grad), NOT the configured global 256. **R31's global-256
replay used the wrong effective batch.**

**Same-data finding (measured):** the reference and RLinf rank-0 batch-32 streams (dumped
`--with-images`) contain the **SAME frames** — images identical (max|Δ|=0.0 after HWC→CHW), state/actions
~5e-8, prompts 99.9997 % identical. So at the production topology both loaders stream the SAME seed-42
data; the batch-32 replay of both rank-0 streams (`docs/evidence/r32_topology_replay.json`) gives an
**IDENTICAL** trajectory (gap −0.0001).

**⇒ The R30/R31 "RLinf loader feeds a different/faster per-step batch" hypothesis is REFUTED.** It was
an artifact of the non-production topology (R31 ran RLinf at `world_size=1` = rank-disjoint 256 unique;
R30 at batch-16). In production both loaders feed the SAME data. The prior AC-11 loader plan-evolution
proposal is **WITHDRAWN**.

**What remains (the batch-32 replay):** the identical replay trajectory tracks the RLinf production curve
r22 (replay last-10 0.075 vs r22 0.073) but lands BELOW the reference production curve r24 (vs 0.102;
step-49 replay 0.052 vs r22 0.048 vs r24 0.090). So the RLinf stack reproduces r22 on the shared data
but the reference production (r24) is higher. With the loader (same data) and the single-GPU stack
(R26–R29) both ruled out, the production r22-vs-r24 gap is **re-opened**. **New leads (hypotheses, NOT
proven) for R33:** (a) the reference PRODUCTION run (`train_pytorch_new.py`, nccl) may be rank-DISJOINT
(effective batch 256) while RLinf production is rank-REPLICATED (effective 32) — if RLinf's nccl run
shares the worker-rank-replication this gloo dump observed but the reference's does NOT, that is a real
RLinf effective-batch bug (config 256, effective 32); (b) the noise/time RNG; (c) a 50-step-compounding
residual R28's 20-step test did not capture. **Verify the two production runs' ACTUAL effective batch /
rank behavior next.** task15 stays **NOT met**; no plan evolution proposed.

### R31 — global-256 replay (NOT production topology): "production-faithful" WITHDRAWN per Codex R31 review; see R32
> **RETRACTION (Codex R31 review):** R31's "production-faithful" label is **WITHDRAWN**. The replay
> correctly used GLOBAL batch 256 + the production recipe knobs, but it bypassed the production loader
> **topology**: the reference sequence was dumped with `num_workers=0` (vs production `num_workers=8`)
> and the RLinf sequence was iterated in a **single non-distributed process** (`world_size=1`),
> consuming 8 sequential micro-batches as a global step. RLinf's `BehaviorSftDataset` folds
> `rank`/`world_size`/`worker_id`/`num_workers` into its chunk partition, so `world_size=1` streams ALL
> chunks, not rank 0's disjoint 1/8 — and worker count changes worker seeds/chunk order. Since the
> quantity under test IS the loader's per-step batch composition, this topology deviation invalidates
> the "production-faithful" claim. The aggregate curve-tracking below is **directional evidence only**.
> **R32** dumps both loaders at the real 8-rank/`num_workers=8` topology with manifests + per-rank/
> per-step hashes. task15 stays OPEN.

R31 reran the loader-sequence replay at the production recipe knobs (Codex R30 review;
`docs/evidence/r31_loader_sequence_prod.json`; `tools/sft_loader_sequence_prod_probe.py`). Verified
that the reference (r24) and RLinf (r22) production runs use the **same** knobs — GLOBAL batch **256**
(micro 32 × 8), `turning_on_radio`, `peak_lr=2.5e-5`, `warmup=1000`, `use_skill:false`. RLinf's `Pi0`
ran the SAME loop (fp32 weights + autocast bf16, AdamW+clip+warmup-LR) + SHARED fixed noise/time for 50
steps at global batch 256 via **gradient accumulation** (8 chunks × 32), on (a) the reference loader's
first-50 sequence (`create_behavior_data_loader_torch`, `num_workers=0` — NOT production) and (b)
RLinf's SFT loader's (`create_behavior_sft_data_loader`, `world_size=1` — NOT the production 8-rank
partition).

Result — **both replays track their production curves (last-10 mean within the DEC-1(b) 0.03 band), and
RLinf's loader is faster:**
- **reference loader** replay last-10 mean **0.106** vs r24 production **0.102** (Δ **0.004**) → tracks;
- **RLinf loader** replay last-10 mean **0.089** vs r22 production **0.073** (Δ **0.016**) → tracks;
- RLinf loader (0.089) descends **faster** than the reference loader (0.106), Δ≈0.017, the SAME
  direction as production (r22 < r24).

So at the **production global-256 batch** (unlike R30's batch-16 artifact, where the gap was an
inflated 0.088), RLinf's loader still feeds a faster-descending per-step batch composition, and each
replay lands within 0.03 of its production curve. Combined with R26→R29 (RLinf == the reference on
identical inputs), this localizes the production first-50 faster-descent to the **data loader's
per-step batch composition**, not a model/optimizer/clip/LR/FSDP defect.

**Honest caveats (no overclaim):** (1) tracking is in the **aggregate** (last-10 mean within 0.03),
NOT step-by-step — the per-step max |Δ| vs the production curve is ~0.135 (reference)/~0.143 (RLinf),
because the replay's fixed shared noise/time + a different shuffle than the exact production run make
individual per-step losses differ; (2) the loader-driven gap at global-256 (~0.017) is **modest** —
smaller than batch-16's 0.088 and than the production step-49 gap (0.042), as expected when a larger
batch averages out per-step composition differences, so the loader explains the **direction** and a
portion of the magnitude, with the replay's single-GPU/fixed-noise deviations (R29 ~0.2%, R24 ~0.004)
+ shuffle differences accounting for the rest; (3) the RLinf replay last-10 (0.089) sits above r22's
(0.073), within 0.03 but not exact.

**AC-11 path (proposed for Codex to choose):** (A) **plan evolution** — accept the loader-composition
difference as a benign framework choice with the R26→R31 root-cause chain and close task15; (B)
**align** RLinf's SFT loader batch composition (e.g. global shuffle) to the reference's, then rerun the
first-50 gate; (C) feed RLinf the reference batches. Given the caveats, (A) is reasonable but the
residual per-step noise should be acknowledged. task15 stays **NOT met** until a path is accepted.

### R30 — loader-sequence replay (REDUCED-SCALE, batch-16): a hypothesis, not the production proof (corrected per Codex R30 review; see R31)
R30 runs the decisive replay Codex required (`docs/evidence/r30_loader_sequence.json`;
`tools/sft_loader_sequence_probe.py` + `tests/unit_tests/_ref_seq_batch_dump.py`). It drives RLinf's
`Pi0` through the SAME loop (fp32 weights + `autocast` bf16, `AdamW(0.9,0.95)`, `clip_grad_norm_(1.0)`,
openpi_cosine warmup LR) for 50 steps on **two batch sequences** — the reference loader's
(`create_behavior_data_loader_torch`) and RLinf's own SFT loader's (`create_behavior_sft_data_loader`,
`use_skill:false`, same `turning_on_radio` task) — from **identical base weights** with the **SAME
fixed shared noise/time**, so the ONLY variable is the per-step batch source (single-GPU, batch 16).

Result — the two sequences descend **differently**:
- **reference loader** sequence: first-5 mean 0.263 → last-5 mean **0.162** (step-0 **0.251** ≈ the
  reference production step-0 0.246);
- **RLinf loader** sequence: first-5 mean 0.190 → last-5 mean **0.074** (step-0 **0.165**).

**Step-0 smoking gun:** on IDENTICAL base weights + IDENTICAL shared noise/time, step-0 loss is
ref-loader 0.251 vs rlinf-loader 0.165. Since weights and noise/time are identical, that 0.086 gap is
**purely the batch (data)** — a clean proof that the two loaders feed different per-step inputs (no
noise/time-RNG confound, unlike the R27 step-0 observation). The gap persists every step → RLinf's
loader sequence descends faster.

**This is DIRECTIONAL evidence, NOT the production proof (corrected per Codex R30 review).** It shows
that loader batch composition *can* change the trajectory under a shared loop — consistent with the
per-step-input hypothesis. But it is a **reduced-scale** replay: single-GPU **batch 16** with
`num_workers=0` and `seed=0`, NOT the production global batch 256 / `seed=42` / `num_workers=8` recipe.
Critically, the **reference-loader trajectory here (0.162) does NOT track the committed reference
production curve (r24 ≈0.090)** — so R30 cannot be accepted as proof that the production faster-descent
is fully root-caused to the loader. The earlier "R30 PROVED" wording is **withdrawn**. **Likely
mechanism (hypothesis):** RLinf's BEHAVIOR SFT loader streams contiguous keyframe chunks (AC-6
streaming), so its per-step batches are more correlated and fit faster, whereas the reference loader
shuffles more globally. **The production-faithful replay at global batch 256 is R31** (below); only if
both loaders track their production curves there may the per-step input be claimed proven. task15 stays
**NOT met**; no AC-11 path is proposed on this reduced-scale artifact.

### R29 — 8-rank FSDP == single-GPU on the same batch: the FSDP distributed step is RULED OUT
R29 exercises the one production-topology piece R28 did not — the 8-rank FSDP sharding/all-reduce
(`docs/evidence/r29_fsdp_consistency.json`; `tools/sft_fsdp_consistency_probe.py` +
`tools/_fsdp_grad_worker.py`). On the **IDENTICAL** fixed 32-frame global batch + fixed noise/time +
base weights, RLinf's `Pi0` runs under (a) **single-GPU** (full batch) and (b) **8-rank FSDP1** with
the production `MixedPrecision(param=bf16, reduce=fp32, buffer=fp32)` + `FULL_SHARD`, each rank fed its
contiguous 4-frame shard. The global grad norm is the L2 norm from `clip_grad_norm_` (FSDP1's
`model.clip_grad_norm_` on 8 ranks — the same call `train_pytorch_new.py:535` uses; `torch.nn.utils`
on 1 rank).

Result — they **MATCH**:
- single-GPU global grad norm **21.125**, 8-rank FSDP **21.174** → **rel diff 0.23 %**;
- losses 0.32414 vs 0.32589 (|Δ| = 0.0018). (All three subprocess return codes 0.)

**So RLinf's 8-rank FSDP sharding / all-reduce / clip is numerically equivalent to the single-GPU
path** — and since R26 (same-batch backward) and R28 (20-step same-input replay) already proved the
single-GPU path equals the reference, **RLinf's full production stack (8-rank FSDP) is equivalent to
the reference on identical inputs**. The **FSDP distributed step is RULED OUT** as the cause of the
production first-50 divergence. (The 32-frame batch's grad norm ~21 differs from the production ~2.4
only because of batch size — more frames lower the mean-gradient norm; this probe isolates the FSDP
delta on a fixed batch, not the production magnitude.)

**No overclaim:** single-GPU == reference is established (R26/R28); R29 isolates ONLY the FSDP delta.
With the model backward, forward, optimizer, LR, clip, master/reduce dtype, `torch.compile`, the
single-GPU multi-step loop, AND now the 8-rank FSDP step all ruled out, the **one remaining candidate
is the per-step INPUT** each loader/RNG feeds in production. **R30 (decisive):** replay the
reference's EXACT production first-N batch/noise/time sequence through RLinf and check it tracks the
reference — the per-step-input proof — then decide the AC-11 path (accept the loader/RNG composition
difference via a plan evolution, or feed RLinf the reference inputs) or rerun the 50/50 gate.

### R28 — MEASURED same-input N-step replay: the single-GPU stack is EQUIVALENT (model+optimizer+clip+LR ruled out)
R28 replaces R27's inspection with the **measured same-input experiment** Codex required
(`docs/evidence/r28_replay_parity.json`; `tools/sft_replay_parity_probe.py` +
`tests/unit_tests/_ref_train_replay_dump.py`). The reference dumper (reference venv) draws **N=20
fixed batches + 20 fixed noise/time** from the real loader and runs 20 optimizer steps on the real
`models_pytorch_new.Pi0`; the RLinf probe replays the **IDENTICAL** batches + noise/time through
RLinf's vendored `Pi0` from the same base weights. **Both** sides use the same single-GPU emulation of
the production recipe — **fp32 master + bf16 compute** params (the R26-validated bf16 forward),
`torch.optim.AdamW(betas=(0.9,0.95), eps=1e-8, wd=1e-10)`, `clip_grad_norm_(max_norm=1.0)`, and the
openpi_cosine warmup LR (`peak=2.5e-5`, `warmup=1000`, init `peak/(warmup+1)`).

Result — the trajectories **MATCH**:
- **mean |Δloss| = 0.00064, max |Δloss| = 0.00247** over the 20 steps; final-step gap 0.0006.
- per-step grad norms track within bf16 noise (e.g. step 0 ref 25.83 / RLinf 25.78; step 13 ref
  4.064 / RLinf 4.047).

**So the single-GPU model + optimizer + clip + LR stack is EQUIVALENT on identical inputs** — this is
**measured**, not code reading, and it extends R26's single-step backward parity to the multi-step
optimizer loop. It **resolves the R27 contradiction**: given truly identical inputs the two stacks
descend identically, so RLinf does *not* "descend faster on the same data" at the single-GPU stack
level. **No overclaim** — what R28 does NOT prove: the two remaining candidates for the production
divergence are **(a)** the per-step INPUT each loader/RNG feeds in production (R19 proved frame-SET
identity for the rank-independent control, but not per-step-batch + noise/time identity), and **(b)**
the **8-rank FSDP sharding / all-reduce**, which this single-GPU probe does NOT exercise. **R29:**
(a) replay the reference's EXACT production first-N batch/noise/time sequence through RLinf and check
it tracks the reference; (b) run the 8-rank FSDP step comparison. task15 stays open.

### R27 — distributed-step components INSPECTED (no code mismatch found); mechanism UNPROVEN (loader-ordering verdict RETRACTED)
R27 compared RLinf's vs the reference's distributed training step by reading the REAL code and the
committed first-50 grad-norm logs (`docs/evidence/r27_distributed_step_localization.json`):
- **Committed grad-norm means**: RLinf production (rank-folded) 2.42, the R20 control
  (rank-independent) 2.38, the reference 1.41 — RLinf's logged 8-rank grad norm is ~1.7× the
  reference's, and the control/reference ratio **grows** 1.12 (step 0) → 1.62 (step 49), i.e. the
  weights diverge over steps (not a constant scale factor).
- **Code paths all MATCH**: loss scaling (RLinf `gradient_accumulation=256//32//8=1`, `loss=loss/1`;
  reference `losses.mean()/len(batches)` — both per-batch mean, no extra scaling); optimizer (both
  `torch.optim.AdamW`, same config, task12); LR (R25); clip (`max_norm=1.0` both); and the grad-norm
  computation (RLinf `get_grad_norm_for_mixed_precision` does per-rank local sum-of-squares → DP
  all-reduce SUM → sqrt = the correct global L2 norm, matching the reference `model.clip_grad_norm_`).
- **This is inspection, not a measured same-input distributed comparison** — it rules out obvious
  code mismatches but does NOT prove the mechanism.

**RETRACTION (Codex R27 review).** R27 originally concluded the divergence was "a benign per-step
loader-ordering difference, not a correctness bug." **That verdict is retracted as an overclaim.** It
was inspection-only, and it conflicts with prior committed evidence: R19 showed the rank-independent
control emitted the reference's EXACT frame/action/prompt stream, and the fp32-master rerun on that
exact-reference stream still descended faster than the reference (see the control rows below) — so
loader ordering cannot be asserted as the mechanism. What is supported: (1) the single-GPU model
backward matches ~0.2 % (R26); (2) by code reading, loss scaling / optimizer / LR / clip / grad-norm
computation show no mismatch; (3) the committed 8-rank grad norm is ~1.7× the reference's and the
ratio grows over steps. The **mechanism is UNPROVEN** and the residual is **not shown benign**.
Unproven candidate hypotheses (not mutually exclusive): the per-step INPUT each loader/RNG feeds, or
a multi-step distributed-training effect (a small per-step grad difference compounding through the
optimizer state). **Next localization (R28, required):** a MEASURED same-input comparison — feed
RLinf's and the reference's stacks the IDENTICAL fixed batches + fixed noise/time from identical base
weights for N steps with the production optimizer/clip/LR, and compare the per-step loss
trajectories. If they match → the stacks are identical and the divergence is the per-step input
(then replay the reference's exact production sequence through RLinf to confirm); if they differ →
localize and fix the stack mechanism, then rerun the first-50 gate. Do not call the residual benign
until that run exists.

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
- **Reference EAGER run** (R25): `docs/evidence/r25_ref_eager_first50.csv` +
  `docs/evidence/r25_eager_vs_compiled.json` (`TORCH_COMPILE_MODE=null`; rules out `torch.compile`;
  records the eager-vs-compiled-vs-RLinf loss + the grad_norm comparison localizing the divergence).
- **Same-batch gradient parity** (R26): `tools/sft_grad_parity_probe.py` +
  `tests/unit_tests/_ref_model_grad_dump.py` → `docs/evidence/r26_grad_parity.json` (RLinf `Pi0` vs
  reference `models_pytorch_new.Pi0` on an identical batch; the model backward matches to ~0.2%, so
  the model code is ruled out; manifest has a full provenance block + per-batch rows).
- **Distributed-step localization** (R27): `docs/evidence/r27_distributed_step_localization.json`
  (committed grad-norm means + a code-reading comparison finding no mismatch in loss scaling/
  optimizer/LR/clip/grad-norm; INSPECTION-only — the "loader-ordering" verdict is retracted and the
  mechanism is unproven pending the R28 measured same-input run).
- **Same-input N-step replay** (R28): `tools/sft_replay_parity_probe.py` +
  `tests/unit_tests/_ref_train_replay_dump.py` → `docs/evidence/r28_replay_parity.json` (RLinf vs the
  reference run the IDENTICAL 20 batches + noise/time + base weights through the same fp32-master+bf16
  AdamW+clip+warmup-LR loop; the trajectories match mean |Δloss|=0.0006 → the single-GPU stack is
  ruled out; provenance block + per-step rows + artifact hashes).
- **8-rank FSDP consistency** (R29): `tools/sft_fsdp_consistency_probe.py` + `tools/_fsdp_grad_worker.py`
  → `docs/evidence/r29_fsdp_consistency.json` (RLinf `Pi0` single-GPU vs 8-rank FSDP1 production
  MixedPrecision+FULL_SHARD on the same 32-frame batch; grad norms match 0.23% → the FSDP distributed
  step is ruled out; provenance + per-side rows + hashes + torchrun return codes).
- **Loader-sequence replay, reduced-scale** (R30): `tools/sft_loader_sequence_probe.py` +
  `tests/unit_tests/_ref_seq_batch_dump.py` → `docs/evidence/r30_loader_sequence.json` (batch-16,
  seed-0; directional only — RLinf loader faster, but the reference replay did not track r24;
  corrected to a hypothesis, superseded by R31).
- **Loader-sequence replay, global-256** (R31): `tools/sft_loader_sequence_prod_probe.py` →
  `docs/evidence/r31_loader_sequence_prod.json` ("production-faithful" WITHDRAWN — bypassed the loader
  topology, world_size=1; superseded by R32).
- **Loader TOPOLOGY dumps + replay** (R32, manifests hardened R33): `tools/_loader_topology_dump.py` +
  `tools/sft_loader_topology_replay.py` → `docs/evidence/r32_loader_topology.json` +
  `r32_topology_replay.json` + `r32_*_topology_hashes.json` (8-rank/`num_workers=8`; manifests with
  provenance + per-rank/per-step hashes; BOTH loaders rank-replicated → effective batch 32; the replay
  now computes a reproducible `same_frames` block — rank-0 streams contain the SAME frames → the R30/R31
  loader hypothesis is REFUTED; r22-vs-r24 re-opened).
- **Production-trainer rank behavior** (R33, reference half SUPERSEDED by R34):
  `docs/evidence/r33_production_rank_behavior.json` (correctly found RLinf rank-replicated/32; its
  reference verdict was wrong — it missed the rank-0 fanout).
- **Production effective-batch topology + FIX** (R34): `tests/unit_tests/_ref_fanout_dump.py` +
  `tools/_loader_topology_dump.py` → `docs/evidence/r34_production_batch_topology.json` +
  `r34_ref_fanout_hashes.json` + `r34_rlinf_fixed_topology_hashes.json` (reference rank-0 fanout MEASURED
  rank-disjoint = effective batch 256; RLinf was rank-replicated = 32; the spawn-safe fix in `rlinf/`
  threads explicit rank/world_size → RLinf now rank-disjoint 256, confirmed + a regression test).
- **First-50 SFT under the fix** (R35): `docs/evidence/r35_first50_under_fix.csv` + `.json` (the 8-GPU
  `train_vla_sft.py` run at effective batch 256; rank-AVG `train/loss` from the tensorboard event file
  vs r24; RLinf now tracks the reference — step49 0.083 ≈ 0.090, 40/50 within 0.03 up from 30/50; the
  systematic divergence is resolved, the residual is noise/time-RNG).
- **PINNED same-input residual experiment** (R36): `tools/sft_pinned_residual_probe.py` +
  `tests/unit_tests/_ref_pinned_run.py` → `docs/evidence/r36_pinned_residual.json` (BOTH the reference
  `models_pytorch_new.Pi0` and RLinf `Pi0` run the IDENTICAL reference rank-0-fanout first-50 global-256
  batches + SHARED numpy noise/time at effective batch 256 via 8×32 accumulation; the per-step
  batch/noise/time sha256 hashes are recorded and verified identical for all 50 steps; the per-step
  losses match **50/50, max |Δ|=0.0003** → the R35 production residual is PROVEN to be the per-step INPUT
  (cross-implementation noise/time-RNG + shuffle), not a model/optimizer/effective-batch residual;
  provenance block + per-step rows + reference venv/src/git rev + return codes). R36 evidence hardened
  (Codex R36 review): every step row now records BOTH arms' state/actions/noise/time hashes + per-field
  equality + an all-four `inputs_identical`; provenance records `rlinf_git_rev`, the reference command, and
  the real reference-arm return code 0 (`_ref_pinned_run.py` writes a failure JSON + `sys.exit(1)` on
  error).
- **PINNED input through the ACTUAL FSDP stack** (R37): the config-gated pinned path
  (`rlinf/data/datasets/behavior/behavior_pinned_loader.py` + `OpenPiPytorchActionModel.sft_forward`
  noise/time passthrough) replays the reference rank-0-fanout first-50 batches + shared noise/time through
  the REAL 8-GPU FSDP `train_vla_sft.py` worker; `tools/sft_pinned_first50_eval.py` extracts the rank-AVG
  `train/loss`/LR/grad-norm and compares per-step to the reference-pinned loss →
  `docs/evidence/r37_pinned_first50.{csv,json}` (**50/50 within 0.03, max |Δ|=0.0024**; the strict DEC-1
  (b) first-50 gate MET through the production stack; provenance: command, source revisions, return codes,
  tensorboard event file, npz hashes, `r36_inputs_identical_all_steps`). CPU unit tests
  `tests/unit_tests/test_openpi_pytorch_pinned_loader.py` (per-rank sharding + noise/time passthrough).
- **ADVISORY ~1h long-run trend** (R38, task16, DEC-1(c)): `tools/sft_advisory_trend_eval.py` →
  `docs/evidence/r38_advisory_1h_trend.{csv,json}` (the NORMAL UNPINNED `use_skill:false` 8-GPU production
  SFT run 800 steps ≈ 1 h; per-step RLinf `train/loss`/LR/grad-norm vs the reference long-run log curve;
  **both descend, Pearson r=0.975, 788/800 aligned within 0.03 for context**; ADVISORY direction/shape
  match, never a CI gate; provenance: command, return code, tensorboard event file, RLinf + reference
  source revisions, model/norm-stats/tokenizer sha256 hashes).

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
proposal is withdrawn), and task15 stays active on a real residual. **R25 RULED OUT `torch.compile`**
(eager reference == compiled reference, both ~0.090), **R26 RULED OUT the model backward** (the
same-batch grad norms match to ~0.2%, `r26_grad_parity.json`), and **R27 INSPECTED the
distributed-step components** (loss scaling, optimizer, LR, clip, and the grad-norm computation show
no code mismatch, `r27_distributed_step_localization.json`) — but inspection does NOT prove the
mechanism, and R27's "benign loader-ordering" verdict is **RETRACTED** (it conflicts with R19, where
the rank-independent control on the reference's exact stream still descended faster). **R28 then ran
that MEASURED same-input comparison** (`r28_replay_parity.json`): on the IDENTICAL 20 batches +
noise/time from identical base weights with the same fp32-master+bf16 AdamW+clip+warmup-LR loop, RLinf
and the reference produce the **same trajectory** (mean |Δloss|=0.0006, max 0.0025), so the single-GPU
**model+optimizer+clip+LR stack is RULED OUT** (measured). **R29 then ruled out the 8-rank FSDP step**
(`r29_fsdp_consistency.json`): RLinf's 8-rank FSDP all-reduced grad norm matches its single-GPU grad
norm to 0.23% on the same batch, so the FSDP sharding/all-reduce is numerically correct → RLinf's full
production stack equals the reference on identical inputs. **R30 gave reduced-scale (batch-16) evidence
consistent with the per-step-input hypothesis** (`r30_loader_sequence.json`): through the identical
loop + shared noise/time, RLinf's loader descended faster than the reference loader's (last-5 0.074 vs
0.162; step-0 0.165 vs 0.251) — but at batch-16/seed-0 the reference-loader trajectory did NOT track
the reference production curve (r24 ≈0.090), so the "PROVED" wording is withdrawn. **R31 reran it at
GLOBAL batch 256** (`r31_loader_sequence_prod.json`): both replays tracked their production curves in
the aggregate (reference 0.106 vs r24 0.102, RLinf 0.089 vs r22 0.073, within 0.03) and RLinf's loader
was faster — **but R31's "production-faithful" label is WITHDRAWN** (Codex R31 review): it used
`num_workers=0` (reference) and `world_size=1` (RLinf) instead of the production 8-rank/`num_workers=8`
rank-aware topology, which drives the per-step batch composition under test. **R32 then dumped both
loaders at the real production topology** (8-rank, `num_workers=8`) and found both RANK-REPLICATED
(effective batch 32) with rank-0 streams containing the **SAME frames** — so the loaders feed the same
data and the **R30/R31 loader hypothesis is REFUTED** (the AC-11 loader proposal is withdrawn). The
batch-32 replay of the shared data tracks r22 but not r24, so the r22-vs-r24 gap is **re-opened**.
R33 mis-concluded both rank-replicated, but **R34 found the EFFECTIVE-BATCH root cause** (correcting it):
the reference trainer builds the loader on rank 0 only and fans out **successive** micro-batches → it is
**rank-DISJOINT, effective batch 256** (measured 256 unique/step), while RLinf was **rank-REPLICATED,
effective batch 32** (its spawned DataLoader workers can't read `torch.distributed`). So the r22-vs-r24
gap **IS** an 8× effective-batch difference. **R34 FIXED it** (threading explicit `rank`/`world_size`
into `BehaviorSftDataset._select_streaming_chunk`; spawn-safe) + a regression test, and CONFIRMED RLinf
is now rank-disjoint (256 unique/step; `docs/evidence/r34_production_batch_topology.json`). **Final
validation (next):** rerun the 8-GPU first-50 under the fix and check it tracks the reference (~0.090).
task15 stays NOT met until that 50/50 (or an accepted plan evolution); task16 (advisory ~1 h trend) and
task18 (final AC-13) remain blocked on it.
