# BEHAVIOR SFT Recipe Audit — RLinf vs Reference (`use_skill: false`)

Knob-by-knob comparison of the RLinf `openpi_pytorch` BEHAVIOR SFT path against the
reference run `pi05_b1k-task0000_sft_pytorch_mixed`, for the `use_skill: false`
alignment effort (AC-11). Each knob carries a verdict and file:line evidence on both
sides. This audit is the CPU-verifiable half of AC-11; the loss-curve confirmation is
the GPU-evidenced half (task14/task15/task16, next round, per DEC-1/DEC-5).

## Sources

**Reference** (`/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed`):
- `src/openpi/training/config.py` — `TrainConfig("pi05_b1k-task0000_sft_pytorch_mixed")` (lines 851-883).
- `src/openpi/training/optimizer.py` — `AdamW` + `CosineDecaySchedule` (lines 15-86).
- `run.sh` — launch (lines 20-25); `scripts/train_pytorch_new.py` — FSDP wrap (lines 161-175, 364-367).
- `outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log` — reference loss log (step-0 line).

**RLinf** (`/mnt/public/xzxuan/repos/RLinf_pi05`):
- `examples/sft/config/behavior_pi05_vla.yaml` — experiment config.
- `examples/sft/config/model/pi0_5_pytorch.yaml` — model-shape template.
- `rlinf/hybrid_engines/fsdp/utils.py::get_lr_scheduler` (`openpi_cosine`, lines 550-579).
- `rlinf/data/datasets/behavior/behavior_sft_data_loader.py` / `behavior_sft_dataset.py`.

## Knob comparison

| Knob | Reference (value @ file:line) | RLinf (value @ file:line) | Verdict |
|------|-------------------------------|---------------------------|---------|
| per-device batch | 32 (`config.py:875` `batch_size=8*32`) | 32 (`behavior_pi05_vla.yaml:54` `micro_batch_size`) | ALIGNED |
| global batch | 256 = 8×32 (`config.py:875` + `run.sh:22` `--nproc_per_node=8`) | 256 (`behavior_pi05_vla.yaml:55` `global_batch_size`) | ALIGNED† |
| gradient accumulation | 1 (no accumulation; 256 in one step over 8 ranks) | 1 **iff run on 8 GPUs** = 256/(32×8) (`fsdp_sft_worker` grad-accum derivation) | ALIGNED† |
| num train steps | 30000 (`config.py:866`) | 30000 (`behavior_pi05_vla.yaml:27,79`) | ALIGNED |
| optimizer | AdamW (`optimizer.py:66-86`) | AdamW (`fsdp_model_manager` optimizer build) | ALIGNED |
| β1 / β2 | 0.9 / 0.95 (`optimizer.py:70-71`) | 0.9 / 0.95 (`behavior_pi05_vla.yaml:74-75`) | ALIGNED |
| eps | 1e-8 (`optimizer.py:72`) | 1e-8 (`behavior_pi05_vla.yaml:76`) | ALIGNED |
| weight_decay | 1e-10 (`optimizer.py:74`) | 1e-10 (`behavior_pi05_vla.yaml:77`) | ALIGNED |
| grad clip (global norm) | 1.0 (`optimizer.py:75`) | 1.0 (`behavior_pi05_vla.yaml:78`) | ALIGNED |
| LR schedule | optax warmup+cosine (`optimizer.py:25-32`) | `openpi_cosine` LambdaLR (`utils.py:550-579`) | ALIGNED |
| peak LR | 2.5e-5 (`config.py:868`, `optimizer.py:20`) | 2.5e-5 (`behavior_pi05_vla.yaml:72`) | ALIGNED |
| warmup steps | 1000 (`optimizer.py:19`) | 1000 (`behavior_pi05_vla.yaml:80`) | ALIGNED |
| decay length | 30000 total → cosine over `decay-warmup`=29000 (`config.py:869`, `optimizer.py:21`) | `num_training_steps-warmup`=29000 (`utils.py:572-574`) | ALIGNED |
| min / end LR | 0.0 (`optimizer.py:23`) | 0.0 (`behavior_pi05_vla.yaml:85`) | ALIGNED |
| step-0 LR (scheduler) | `peak/(warmup+1)`=2.4975e-8 (`optimizer.py:27`; log: `learning_rate=2.4975e-08`) | `peak/(warmup+1)` (`utils.py:568`) | ALIGNED (exact; the `openpi_cosine` warmup ramp matches log steps 0/1/2) |
| EMA | None / disabled (`config.py:872`) | off (no EMA in worker / config) | ALIGNED |
| seed | 42 (`config.py:540` default) | 42 (`behavior_pi05_vla.yaml:56`) | ALIGNED |
| weights | `pi05_base_pytorch_new`, fp32 load (`config.py:881`) | `pi05_base_pytorch_new` (`behavior_pi05_vla.yaml:63`), `load_for_training` fp32 (`model/pi0_5_pytorch.yaml:10`) | ALIGNED |
| precision / FSDP MixedPrecision | `mp_bfloat16` → fp32 load + `MixedPrecision(param_dtype=bf16, reduce_dtype=torch.float32)`, buffers fp32 by omission (`init_model`, `train_pytorch_new.py:298-300`) | `param_dtype: ${actor.model.precision}` (bf16), `reduce_dtype: fp32`, `buffer_dtype: fp32` (`behavior_pi05_vla.yaml:101-103`); fp32 master + bf16 compute | ALIGNED (R22: reduce/buffer dtype changed bf16→fp32 to match the reference; pinned by `test_sft_fsdp_full_shard_matches_reference`) |
| FSDP | fsdp1 FULL_SHARD (`run.sh:21`, `train_pytorch_new.py:161`) | `full_shard` (`behavior_pi05_vla.yaml:88,92`) | ALIGNED |
| FSDP fwd/bwd prefetch | disabled (`USE_PREFETCH=0` → `forward_prefetch=False`, `backward_prefetch=None`, `train_pytorch_new.py:163-168,364`) | FSDP defaults | ALIGNED-in-effect‡ |
| gradient checkpointing | used | enabled (`behavior_pi05_vla.yaml:94`) | ALIGNED (memory-only; numerically identical) |
| action_horizon | 32 (`config.py:855` `Pi0Config(action_horizon=32)`) | 32 (`model/pi0_5_pytorch.yaml:21` `num_action_chunks`) | ALIGNED |
| model action_dim (padded) | 32 (`pi0_config.py:27`) | 32 (`model/pi0_5_pytorch.yaml:36` `model_action_dim`) | ALIGNED |
| paligemma_variant | gemma_2b (`pi0_config.py:22`) | gemma_2b (`model/pi0_5_pytorch.yaml:37`) | ALIGNED |
| action_expert_variant | gemma_300m (`pi0_config.py:23`) | gemma_300m (`model/pi0_5_pytorch.yaml:38`) | ALIGNED |
| max_token_len | 200 (`pi0_config.py:42` pi05 `__post_init__`) | 200 (`model/pi0_5_pytorch.yaml:39`) | ALIGNED |
| repo_id | behavior-1k/2025-challenge-demos (`config.py:857`) | behavior-1k/2025-challenge-demos (builder default) | ALIGNED |
| task set | `["turning_on_radio"]` (`config.py:862`) | `["turning_on_radio"]` (`behavior_pi05_vla.yaml:37`) | ALIGNED |
| prompt source | `prompt_from_task=True` (`config.py:859`) | `use_skill: false` → main-task text (`behavior_pi05_vla.yaml:38`) | ALIGNED |
| fine_grained_level | 0 (`config.py:863`) | 0 (builder default, `behavior_sft_data_loader.py:449`) | ALIGNED |
| episode set | `list(range(200))` (`config.py:860`) | task-0 filter → all 200 (`behavior_sft_dataset.py:855-868`) | ALIGNED§ |
| norm stats | task-0000 `norm_stats.json` (`config.py` asset default) | task-0000 via `assets_dir`+`asset_id` (`behavior_pi05_vla.yaml:68-69`) | ALIGNED |
| num_workers | 8 (`config.py:874`) | 8 (builder default, `behavior_sft_data_loader.py:448`) | ALIGNED |
| shuffle | True | True (`behavior_sft_data_loader.py:300`) | ALIGNED |
| DataLoader prefetch_factor | n/a (reference uses LeRobot loader) | PyTorch default 2 | NOT loss-relevant‡ |
| loss logging | log_interval=1; loss AVG-reduced across ranks then logged (`config.py:878`, `train_pytorch_new.py:528`) | every global step; loss AVG-reduced across ranks (`fsdp_sft_worker.py:198-200`) | ALIGNED |
| logged LR | `lr_schedule(global_step)` — the LR USED for that step (`train_pytorch_new.py:500,547`); step 0 logs 2.4975e-8 | the step's LR via `lr_list[0]` captured before `lr_scheduler.step()` (`fsdp_sft_worker.py:178-190`) | ALIGNED (R13 fix; previously logged the next step's LR) |

### Notes

- **† global batch / grad-accum.** RLinf reaches the reference's 256-sample, single-step
  update only when launched on **8 GPUs** (`grad_accum = global/(micro×world_size) =
  256/(32×8) = 1`). On fewer GPUs the same config yields `grad_accum>1`, which changes the
  micro-batch granularity of the logged loss. **The task15 first-50-step run MUST use 8
  GPUs** to match the reference loss aggregation. This is a run-configuration constraint,
  not a code divergence.
- **‡ prefetch.** The reference `USE_PREFETCH` flag toggles **FSDP** `forward_prefetch` /
  `backward_prefetch` (compute/comm overlap), not data ordering — it does not change batch
  contents, RNG, or numerics (`train_pytorch_new.py:161-175`). The RLinf DataLoader's
  `prefetch_factor` is a different, also-numerically-irrelevant mechanism (load/compute
  overlap; the deterministic streaming partition + fixed seed fix the sample→batch mapping
  regardless). Neither affects the loss curve. No change required.
- **§ episode set.** The reference `episodes_index=list(range(200))` positionally selects
  the first 200 episodes of the task. `turning_on_radio` (task id 0) has **exactly 200**
  episodes in `2025-challenge-demos` (verified by counting `meta/episodes.jsonl`), and the
  RLinf streaming dataset's task filter (`episode_index // 1e4 == 0`) loads exactly those
  200. So `range(200)` and the RLinf default load the identical episode set. This is pinned
  by a skip-gated real-loader test asserting the loaded episode count is 200, so a future
  data change that breaks the equivalence is caught.

## Conclusion

**Every CPU-verifiable knob of the `use_skill: false` SFT recipe is aligned with the
reference.** No config or code divergence remains at the recipe level (the M3 YAML/config
work and the `openpi_cosine` scheduler landed the alignment; the R13 logged-LR fix below
closed the last logging-semantics gap). The two items that are not pure config — the
global-batch/grad-accum equivalence (needs 8 GPUs) and the loss-curve match itself — are
exactly DEC-1's GPU-evidenced tier and are the subject of the next round.

This audit is pinned against regression by:
- `tests/unit_tests/test_openpi_pytorch_sft_recipe.py` — asserts every RLinf recipe knob
  equals the reference value (the constants in that test cite the reference file:line), plus
  a skip-gated real-loader gate asserting the `turning_on_radio` episode count is 200.
- `tests/unit_tests/test_openpi_pytorch_sft_schedule.py` — the `openpi_cosine` LR schedule
  is reference-exact across warmup and decay, AND a regression test driving the real
  `FSDPSftWorker.run_training` ordering asserts the logged step-0 LR is the LR used for that
  step (`peak/(warmup+1)`), not the next step's value.
- `tests/unit_tests/test_openpi_pytorch_sft_ref_parity.py` — per-sample tokenizer/transform/
  loader-output parity vs the real reference loader (existing).

## Next round (AC-11 GPU evidence, DEC-5)

1. **task14 — fixed-batch forward-loss parity.** Load `pi05_base_pytorch_new`, build one
   fixed production batch, fix the flow-matching noise + timestep RNG so the MSE is
   reproducible, and assert the forward loss is within DEC-1 tolerance (rel 5% or abs 0.01)
   of the reference step-0 loss `0.24609375`. Commit command, env, and model/norm-stats/
   tokenizer hashes under `/mnt/public/xzxuan/tmp`.
2. **task15 — first-50-step band.** Launch the matched recipe on **8 GPUs** and capture the
   first 50 rank-0 per-step pre-optimizer losses; assert each is within `|Δ| ≤ 0.03` of the
   reference log band (`≈0.246` at step 0, grad-norm `≈2.0`). Commit the run log.
3. **task16 — advisory ~1h trend.** Evidence-only loss-trend comparison (never a CI gate).
