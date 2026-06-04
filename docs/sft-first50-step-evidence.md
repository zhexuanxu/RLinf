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

## Next round (blocking)

Localize the **optimizer/gradient/update** divergence to a PROVEN cause (Codex R16 step 5):
from the same base weights, feed RLinf and the reference the same first-N global batches and
compare post-step loss, LR, grad-norm, and a small fixed set of parameter-delta norms; fix the
FSDP/optimizer update path; re-run the first-50-step comparison until DEC-1 (b) holds. AC-6's
rank-aware sharding is retained (it is not the cause). task16 / task18 follow once task15 passes.
