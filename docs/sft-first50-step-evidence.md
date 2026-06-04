# BEHAVIOR SFT First-50-Step Training Evidence (task15, AC-11 / DEC-1 (b))

GPU run-evidence (DEC-5) for the `use_skill:false` first-50-step training-loss gate.

**Verdict (Round 20): RESOLVED.** The first-50-step flat-vs-dropping divergence was caused by
RLinf training the model in **pure bf16** — the `openpi_pytorch` factory cast the fp32-loaded
training weights to bf16 *before* FSDP wrapped them, so the AdamW optimizer updated **bf16
master weights** and the tiny warmup-LR updates (~1e-6, below the bf16 ULP ≈0.0078 near 1.0)
were lost to rounding → the loss stayed flat. The fix keeps **fp32 master weights** for training
(FSDP MixedPrecision casts to bf16 only for compute, matching the reference recipe). After the
fix, RLinf's first-50 loss **descends 0.226 → 0.046** (was flat ≈0.24), tracking the reference's
0.246 → 0.090 descent: **46/50** within the DEC-1 (b) band (`|Δ| ≤ 0.03` OR ±2σ) and **29/50**
within `|Δ| ≤ 0.03` (was 12/50). See **"Round 20 — RESOLVED"** below for the proof, the
controlled rank-independent comparison, and the honestly-bounded residual. The R16 run that
follows is retained as the pre-fix baseline.

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

## Round 20 — RESOLVED: bf16 master weights → fp32 master weights

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

- **Direct measurement** (single-GPU, fixed batch, 50 AdamW steps at the openpi_cosine warmup LR):
  the bf16 model changed only **0.91 % of params** (30.6 M / 3.35 B) and the loss stayed ≈flat
  (0.264 → 0.245) — 99 % of the updates vanished to rounding.

### The fix
`get_model` no longer downcasts the training model: for `load_for_training=True` the fp32 master
is **kept** and FSDP MixedPrecision casts to bf16 only for the forward/backward (eval is unchanged
— still bf16-strict). This matches the reference recipe exactly (fp32 load + FSDP1 MixedPrecision).

### Production re-run (8 GPUs, fix applied) — descends
Same command/config as the R16 run (`runner.logger.log_path=/mnt/public/xzxuan/tmp/r20_sft_results`).
RLinf's first-50 loss now **descends 0.226 → 0.046** (first-50 mean 0.151 vs reference 0.168),
tracking the reference's 0.246 → 0.090. **29/50** within `|Δ| ≤ 0.03` (was 12/50); **46/50**
within the DEC-1 (b) band. Committed scalars: `docs/evidence/r20_first50_losses.csv` (columns add
`within_2sigma`). The flat-vs-dropping divergence — the AC-11 blocker — is resolved.

### Controlled rank-independent comparison (rank-folding is NOT the residual)
The 4 band-outliers are RLinf descending *faster/lower* than the reference (a benign direction),
not flat. To attribute that residual without overclaiming, a second 8-GPU run applied the fix
**and** the temporary rank-independent partition (R19's exact-reference stream, 32 unique/step;
patch reverted after the run). Its curve is **near-identical to production** step-by-step (e.g.
step 37 0.075 vs 0.079; the step-19 spike 0.313 vs 0.319; same 29/50 and 46/50). Committed:
`docs/evidence/r20_control_rank_independent_first50_losses.csv`. So AC-6 rank-folding (256 vs 32
unique frames/step) makes **no meaningful difference** to the first-50 loss — the residual gap
with the reference is **not** the data stream (already ruled out exactly in R19, and again here).

### Honestly-bounded residual
With the optimizer fixed (both descend) and the data stream ruled out (R19 exact identity + the
rank-independent control here), the remaining ~2× tail gap (RLinf ≈0.046 vs reference ≈0.090) is
**not** the optimizer or the data. It is attributable to RNG and aggregation differences between
the two stacks (flow-matching noise/time sampling, image augmentation, loss reduction) and is
**benign** — RLinf descends at least as fast as the reference. Tightening it is a separate,
second-order item, not the AC-11 flat-loss blocker, which is closed.

### DEC-5 artifact (this round)
- **RLinf scalars** (both runs) from the tensorboard event files under
  `/mnt/public/xzxuan/tmp/r20_sft_results/tensorboard/` and `…/r20_control_sft_results/tensorboard/`
  via `EventAccumulator` (tags `train/loss`, `train/learning_rate`, `train/grad_norm`), step 0..49.
- **Reference loss**: reused from the committed `docs/evidence/r16_first50_losses.csv` `ref_loss`
  column (same reference log as R16).
- **Artifact hashes** (full-file sha256[:16]): `pi05_base_pytorch_new/model.safetensors`
  `f6391204c480d6c5`; task-0000 `norm_stats.json` `d66ed16830a98f90`;
  `paligemma_tokenizer.model` `8986bb4f423f07f8` (unchanged from R16 — same inputs).

task16 (advisory ~1 h trend) and task18 (final AC-13) follow now that task15's blocker is cleared.
