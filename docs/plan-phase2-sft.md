# Phase 2 — SFT Training for the Self-Contained PyTorch OpenPI 0.5 Model on BEHAVIOR

## Goal Description

Implement SFT (supervised fine-tuning) training for the **self-contained new PyTorch OpenPI 0.5 model** (`rlinf/models/embodiment/openpi_pytorch/`, built in Phase 1 for eval) on the BEHAVIOR environment, **fully aligned with `openpi-comet-pytorch-mixed`'s training**, so that:

```
bash examples/sft/run_vla_sft.sh behavior_pi05_vla
```

runs end to end and shows a **clear training-loss decrease comparable to the reference run** (whose log is `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log`).

This phase **builds on Phase 1** (eval is complete and committed): the vendored model core already contains `Pi0.compute_loss` (flow-matching MSE), and the new `openpi_action_model.py` reserves `sft_forward`/`compute_loss` as `NotImplementedError`. Phase 2 implements those and the surrounding data/training pipeline.

SFT must support **two modes**:
1. **Direct task-string** ("turn on radio" → actions) — what `openpi-comet-pytorch-mixed` implements (the runnable acceptance target).
2. **Skill-granularity** (subtask-level), aligned with the JAX `openpi-comet` implementation, centered on `enable_gap`, `allow_left`, `allow_right`.

Approach: **reuse RLinf's existing SFT runner / FSDP worker / FSDP machinery** plus the vendored `Pi0.compute_loss`; do **not** port `train_pytorch_new.py` wholesale (that would duplicate runner infrastructure).

**Out of scope** (deferred): the `full_pi05` full-VLM-autoregressive (think-then-act) SFT path and `static_kv_cache`; RL / DAgger; non-BEHAVIOR environments. These must fail loudly under the new SFT path.

### Resolved key decisions (see Pending User Decisions for detail)
- **SFT base checkpoint** = `/mnt/public/xzxuan/models/pi05_base_pytorch_new` (new-format, **fp32**, ~13.4 GB) — exactly what the reference trainer loads; loaded directly by the new `Pi0` (no conversion), then cast to bf16 for training.
- **Data parallelism** = **rank-aware streaming chunk partitioning**: BEHAVIOR is a streaming dataset whose `__getitem__` ignores `idx` and whose chunk partition is keyed only on `(seed, worker_id)`, so `DistributedSampler` has no effect and all ranks would otherwise load identical data (the reference adopted rank-0 p2p fanout for this reason). The fix folds `rank` into the chunk partition so each rank loads a distinct shard (256 unique samples per global batch, not 32 replicated 8×).
- **Loss acceptance** = directional / smoothed decrease comparable to the reference bands (not byte-exact numeric gates).
- **Skill mode** = dataset capability + parity test; the direct-task launch is the runnable acceptance.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests. The **runnable acceptance is AC-9** (the SFT launch runs) and **AC-10** (loss decreases comparably, directional per DEC-3); data-pipeline correctness (AC-4/AC-5/AC-6/AC-7) is the highest-risk area per the draft.

- AC-1: The SFT loss path is wired into the new action model.
  - Positive Tests (expected to PASS):
    - `OpenPiPytorchActionModel.forward(forward_type=ForwardType.SFT, data=batch)` and/or `sft_forward(data)` accepts a tuple `(observation, actions)` OR a dict `{"observation": ..., "actions": ...}`, builds the new `model.Observation`, moves observation + actions to the model device, and returns a scalar loss (the `(B, action_horizon)` output of `Pi0.compute_loss` reduced via `.mean()`).
    - The SFT worker's train step (`self.model(forward_type=ForwardType.SFT, data=batch)`) receives a scalar it can call `.backward()` on.
  - Negative Tests (expected to FAIL):
    - A malformed batch (missing observation or actions) raises a clear error.
    - Returning an unreduced `(B, action_horizon)` tensor where a scalar is required is caught.

- AC-2: SFT config validation allows `openpi_pytorch` while still rejecting unsupported SFT paths.
  - Positive Tests (expected to PASS):
    - `validate_sft_cfg` passes for `behavior_pi05_vla` with `model_type: openpi_pytorch` and a BEHAVIOR data config.
    - The existing test that asserts SFT is rejected for `openpi_pytorch`, and the existing "reserved SFT methods raise" test, are updated to the new (supported) behavior.
  - Negative Tests (expected to FAIL):
    - `full_pi05: True` (or dsrl / value-head) under SFT raises; a non-BEHAVIOR data config raises — using data-config / `config_name` signals (SFT configs have no `env.*`).

- AC-3: The SFT model build loads the fp32 new-format base, casts to bf16, and enables gradient checkpointing — via a dedicated training path distinct from the eval bf16-strict loader.
  - Positive Tests (expected to PASS):
    - A training build path (e.g. a `load_for_training` flag or dedicated builder) loads `pi05_base_pytorch_new` (fp32, new-format) into `Pi0` with strict keys and **no** dtype rejection, casts to bf16, and exposes `gradient_checkpointing_enable`/`disable`. The Phase-1 eval loader is unchanged.
  - Negative Tests (expected to FAIL):
    - Routing the fp32 base through the eval loader's bf16-strict validation fails (proving the paths are distinct).
    - A wrong-key checkpoint fails fast.

- AC-4: Worker dispatch + a self-contained BEHAVIOR SFT dataloader.
  - Positive Tests (expected to PASS):
    - `FSDPVlaSftWorker.build_dataloader()` handles `SupportedModel.OPENPI_PYTORCH` and builds a dataloader under `openpi_pytorch/dataconfig/` that yields `(new model.Observation, actions)` via an explicit `collate_fn`.
    - The new dataconfig imports no installed `openpi` (lerobot is allowed); the AST import-isolation gate, extended to cover `dataconfig`, passes.
    - The first batch has the verified contract: 3 RGB images (`base_0_rgb`/`left_wrist_0_rgb`/`right_wrist_0_rgb`) + image masks, a normalized state padded to 32, tokenized prompt + mask, and actions `[B, action_horizon, 32]`.
  - Negative Tests (expected to FAIL):
    - Any `from openpi …` in the new dataconfig fails the isolation gate.
    - A batch missing a required field raises.

- AC-5: Per-rank data sharding is correct via rank-aware streaming chunk partitioning (the DEC-1 fix).
  - Context: BEHAVIOR is a **streaming** dataset whose `__getitem__` ignores `idx` and whose chunk partition is keyed only on `(seed, worker_id)` — not `rank`. So `DistributedSampler` (which works by subsetting `idx`) has **no effect**, and under 8-GPU torchrun all ranks would otherwise load identical data (a "256 global batch" being only 32 unique samples replicated 8×). The fix is to fold `rank` into the streaming chunk partition.
  - Positive Tests (expected to PASS):
    - With `rank` incorporated into the streaming chunk partition (alongside `seed` and `worker_id`), under `torchrun` with ≥2 ranks each rank loads a **distinct, non-overlapping** chunk set: per-rank first-batch identity hashes differ, and a global batch of 256 (32×8) is 256 unique samples rather than 32 replicated across 8 ranks.
  - Negative Tests (expected to FAIL):
    - A chunk partition keyed only on `(seed, worker_id)` (rank-independent), or relying on `DistributedSampler` over the streaming dataset, yields identical data across ranks and is detected and rejected/fixed.

- AC-6: Direct-task-mode data parity vs the reference loader.
  - Positive Tests (expected to PASS):
    - On fixed BEHAVIOR samples, the new loader's prompt + tokenized_prompt, images (resize/normalize), state (extract + normalize + pad to 32), and actions (normalize + pad to 32) match `openpi-comet-pytorch-mixed`'s `create_behavior_data_loader_torch` after transforms, within tolerance.
  - Negative Tests (expected to FAIL):
    - Changing the image resize or normalization-stats source breaks the match and is caught.

- AC-7: Skill mode aligned with `openpi-comet` (dataset capability + parity test).
  - Positive Tests (expected to PASS):
    - The dataset supports `skill_labels` + `enable_gap` / `allow_left` / `allow_right` with the exact `openpi-comet` `_build_skill_boundaries` / `_get_skill_label` semantics: separate left/right window extension by frame counts at contiguous (no-gap) boundaries, `enable_gap` absorbing gap frames into adjacent skills, and overlap → random per-sample skill choice.
    - A parity test on fixed annotations matches the JAX frame→skill-label mapping for `enable_gap=True`, `allow_left=100`, `allow_right=100`.
  - Negative Tests (expected to FAIL):
    - The repository's prior midpoint-split gap behavior fails the parity test; mismatched window logic is caught.

- AC-8: FSDP / precision / optimizer / LR aligned with the reference.
  - Positive Tests (expected to PASS):
    - The SFT config runs the model in bf16 with gradient checkpointing on; AdamW betas (0.9, 0.95) / eps 1e-8 / weight_decay 1e-10 / clip_grad 1.0; warmup = 1000 (init = peak/(warmup+1)) → cosine-to-0 over 30000; global batch 256 / 32-per-GPU / grad_accum 1.
    - A unit test asserts the first few LR-schedule values match the reference warmup+cosine formula within tolerance.
  - Negative Tests (expected to FAIL):
    - A scheduler that doesn't match the warmup init, or a non-bf16 compute path, is flagged by the LR / precision tests.

- AC-9: The SFT launch runs end to end (the runnable acceptance).
  - Positive Tests (expected to PASS):
    - `bash examples/sft/run_vla_sft.sh behavior_pi05_vla` with small params reaches model init, first batch, one forward/backward, grad clip, one optimizer + scheduler step, and a checkpoint save, without error.
    - A tiny `dummy`-variant CPU unit test exercises `sft_forward` (Observation + actions → finite scalar loss) before any GPU run.
  - Negative Tests (expected to FAIL):
    - A missing base checkpoint / data path produces a clear error rather than a silent hang.

- AC-10: Training loss decreases comparably to the reference (directional, per DEC-3).
  - Positive Tests (expected to PASS):
    - A short run shows the smoothed loss drop from ~0.25 toward <~0.05 over the first ~100–200 steps.
    - A longer run (≥ ~30 min) shows a clear continuing smoothed decrease in the same ballpark as the reference bands (≈ 0.085 at step 50, 0.05 at 100, 0.021 at 1000, ~0.008 at 5000) — advisory, not byte-exact.
  - Negative Tests (expected to FAIL):
    - A loss that stays flat, diverges, or remains near the init value indicates a real defect and fails review.

- AC-11: Configs — `behavior_pi05_vla.yaml` retargeted + a clean `pi0_5_pytorch.yaml` model config.
  - Positive Tests (expected to PASS):
    - `examples/sft/config/behavior_pi05_vla.yaml` composes with `model_type: openpi_pytorch`, `model_path: …/pi05_base_pytorch_new`, bf16, gradient_checkpointing True, and no `full_pi05`.
    - `examples/sft/config/model/pi0_5_pytorch.yaml` exists (clean/minimal: no full_pi05 / dsrl / value-head / RL fields).
  - Negative Tests (expected to FAIL):
    - A config reintroducing full_pi05 / dsrl / value-head for SFT fails validation.

- AC-12: An SFT-saved checkpoint round-trips into the Phase-1 eval path.
  - Positive Tests (expected to PASS):
    - An SFT-saved RLinf FSDP checkpoint is exported to the eval directory format (new-format `model.safetensors` + `config.json` + norm-stats tree) and loads via the Phase-1 eval `get_model` (strict), enabling eval of an SFT-trained model.
  - Negative Tests (expected to FAIL):
    - An export missing `config.json` / norm-stats or with wrong keys fails the eval strict load.

- AC-13: The old `openpi` package and the Phase-1 eval remain regression-free.
  - Positive Tests (expected to PASS):
    - The old `openpi` SFT/eval paths still work; the Phase-1 eval (success_once ~0.34) still passes; changes are additive for `openpi_pytorch` SFT.
  - Negative Tests (expected to FAIL):
    - A change that breaks the old `openpi` SFT path or the Phase-1 eval is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A complete, self-contained SFT training path for `openpi_pytorch` on BEHAVIOR that: wires `sft_forward` (action normalize→pad→`Pi0.compute_loss`→mean) with gradient-checkpointing pass-through; adds a fp32-base training build path; relaxes the SFT guard (updating the affected tests); dispatches `OPENPI_PYTORCH` in the SFT worker to a vendored BEHAVIOR dataloader (lerobot OK, no installed `openpi`) with an explicit collate, verified per-rank sharding, direct-mode parity, and a skill-mode port (enable_gap/allow_left/allow_right) with a parity test; aligns FSDP/precision/optimizer/LR to the reference; ships the two configs; runs `run_vla_sft.sh behavior_pi05_vla` to a clear directional loss decrease; and exports SFT checkpoints back to the eval format with a round-trip test.

### Lower Bound (Minimum Acceptable Scope)
`bash examples/sft/run_vla_sft.sh behavior_pi05_vla` runs end to end on the new `openpi_pytorch` model (direct-task mode), self-contained (no installed `openpi` in the new SFT path), starting from `pi05_base_pytorch_new`, with rank-aware streaming chunk partitioning verified to give each rank distinct data, and produces a clear smoothed training-loss decrease in the same ballpark as the reference; the skill-mode dataset capability exists and passes its parity test (even if not run end to end).

### Allowed Choices
- Can use:
  - RLinf's existing SFT runner / `FSDPVlaSftWorker` / `FSDPModelManager` / FSDP wrap + optimizer + scheduler machinery.
  - The vendored Phase-1 `openpi_pytorch` package (model core incl. `Pi0.compute_loss`, transforms, tokenizer, normalize, behavior policy).
  - `lerobot` (LeRobot dataset streaming + PyAV video) and the BEHAVIOR-1K dataset at `/mnt/public/xzxuan/data/2025-challenge-demos`.
  - `pi05_base_pytorch_new` as the SFT base; bf16 training; gradient checkpointing.
- Cannot use:
  - Any `import openpi` / `from openpi …` (installed package) inside the new `openpi_pytorch` SFT code path.
  - The `full_pi05` full-VLM path / `static_kv_cache`; dsrl / value-head / RL code; non-BEHAVIOR datasets.
  - A wholesale port of `train_pytorch_new.py` that duplicates RLinf's runner/worker.
  - Edits that break the old `openpi/` package or the Phase-1 eval.

> **Note on Deterministic Designs**: The draft fixes several choices (self-contained, BEHAVIOR-only, align with the reference training + dataloader, start from `pi05_base_pytorch_new`). Where the draft/reference is prescriptive, the bounds converge to that specification; the remaining latitude is the data-parallel mechanism (resolved to RLinf-native per-rank loading with rank-aware streaming chunk partitioning, since the streaming dataset makes `DistributedSampler` ineffective) and code organization.

## Feasibility Hints and Suggestions

> **Note**: Reference-only conceptual suggestions, not prescriptive.

### Conceptual Approach

SFT data + loss flow to implement (per step, per rank):

```
LeRobot BEHAVIOR-1K (2025-challenge-demos)
   │  BehaviorSFTDataset (vendored; lerobot OK; streaming + PyAV video)
   │    - direct mode: prompt = orchestrator task text
   │    - skill mode : prompt = skill_labels[idx] via enable_gap/allow_left/allow_right windows
   │  per-sample transforms (vendored, reuse Phase-1): BehaviorInputs → resize 224 →
   │    Normalize(state, norm_stats) → tokenize(discrete normalized state) → pad state→32
   │    Normalize(actions) → pad actions→32
   │  collate_fn → batched new model.Observation + actions [B, action_horizon, 32]
   ▼
OpenPiPytorchActionModel.forward(ForwardType.SFT, data=(obs, actions))
   │   → Pi0.compute_loss(obs, actions, train=True)  → [B, action_horizon]
   │   → .mean()  → scalar loss
   ▼
FSDP (bf16 params, grad-checkpoint on) backward → AdamW(fused) clip 1.0 → warmup+cosine step
```

- Base/precision (DEC-2): load `pi05_base_pytorch_new/model.safetensors` (fp32, new-format) directly into `Pi0` (strict, no conversion), `.to(bfloat16)`, `gradient_checkpointing_enable()` — a training build path distinct from the eval bf16-strict loader.
- Data parallel (DEC-1): BEHAVIOR is a STREAMING dataset whose `__getitem__` ignores `idx` and whose chunk partition is keyed only on `(seed, worker_id)` — so `DistributedSampler` (which subsets `idx`) has NO effect and all 8 ranks would otherwise load identical data (the "256 global batch" being really 32 unique samples replicated 8×, which is why the reference switched to rank-0 fanout). The fix folds `rank` into the streaming chunk partition (alongside `seed` + `worker_id`) so each rank loads a distinct, non-overlapping shard (256 unique samples per global batch). task1 confirms the exact partition code; task9 implements + tests the rank-aware partition.
- Action contract: env actions are 23-dim; **normalize first, then pad to the 32-dim model action dim** before `compute_loss` (mirror the eval output transform inverse).
- Skill mode (AC-7): port `openpi-comet`'s `_build_skill_boundaries`/`_get_skill_label` exactly; do not reuse the repo's midpoint-split.

### Relevant References
- `rlinf/models/embodiment/openpi_pytorch/` — Phase-1 package: `utils/pi0.py` (`Pi0.compute_loss`, `embed_prefix/suffix`), `utils/model.py` (`Observation`, `preprocess_observation`), `openpi_action_model.py` (reserved `sft_forward`), `processing.py`, `normalize.py`, `tokenizer.py`, `policies/behavior_policy.py`, `__init__.py` (`get_model`), `_import_isolation.py`.
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` — `FSDPVlaSftWorker.build_dataloader()` (model_type dispatch), `get_train_model_output()` (the `forward(ForwardType.SFT, data=batch)` call).
- `rlinf/hybrid_engines/fsdp/` — `fsdp_model_manager.py` (optimizer/scheduler/checkpoint), `strategy/fsdp.py` + `utils.py` (wrap policy, mixed precision, sharding).
- `rlinf/config.py` — `validate_sft_cfg`, `_validate_openpi_pytorch_eval_cfg` (the SFT guard to relax).
- `rlinf/models/embodiment/openpi/dataconfig/{behavior_dataset,behavior_data_loader,behavior_b1k_dataconfig,behavior_vlm_data_loader}.py` — the current BEHAVIOR loaders (skill logic, lerobot usage, openpi imports) to port self-contained.
- `examples/sft/config/behavior_pi05_vla.yaml`, `examples/sft/run_vla_sft.sh`, `examples/sft/config/behavior_qwen3_vlm_sft_agentic.yaml` (skill VLM precedent).
- Sources: `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts/train_pytorch_new.py` + `src/openpi/training/data_loader.py:create_behavior_data_loader_torch` + `build_datasets`; `/mnt/public/xzxuan/repos/openpi-comet/src/behavior/learning/datas/dataset.py` (JAX skill logic).
- `/mnt/public/xzxuan/models/pi05_base_pytorch_new/` (SFT base); `/mnt/public/xzxuan/data/2025-challenge-demos` (data); reference log `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log`.

## Dependencies and Sequence

### Milestones
1. Milestone M1 — Unblock SFT for `openpi_pytorch`.
   - Step 1: Implement `sft_forward`/`forward(ForwardType.SFT)` (normalize+pad actions → `Pi0.compute_loss` → mean; tuple/dict; device) + `gradient_checkpointing_enable/disable` pass-through.
   - Step 2: Add the fp32-base training build path (load `pi05_base_pytorch_new`, cast bf16); relax the config SFT guard; update the affected tests.
   - Step 3: Add `OPENPI_PYTORCH` to `FSDPVlaSftWorker.build_dataloader()`.
2. Milestone M2 — Self-contained BEHAVIOR SFT dataloader (direct mode).
   - Step 1: Vendor/port the BEHAVIOR dataset + loader under `openpi_pytorch/dataconfig/` (lerobot OK, no installed `openpi`); explicit `collate_fn` → `(model.Observation, actions)`; extend the AST import-isolation gate.
   - Step 2: First-batch contract test + direct-mode parity vs the reference loader.
3. Milestone M3 — Rank-aware data sharding + skill mode.
   - Step 1: Confirm the streaming chunk-partition root cause (idx-ignored; `(seed, worker_id)` only) and implement rank-aware partitioning (fold `rank` in) so each rank loads distinct data (DEC-1); add the distinct-data test.
   - Step 2: Port the `openpi-comet` skill window logic (enable_gap/allow_left/allow_right) + parity test.
4. Milestone M4 — Training alignment + configs.
   - Step 1: Align FSDP (bf16, grad-checkpoint), optimizer (AdamW), and LR (warmup+cosine) to the reference; LR-schedule unit test.
   - Step 2: Retarget `behavior_pi05_vla.yaml`; add `pi0_5_pytorch.yaml` model config.
5. Milestone M5 — Runnable smoke.
   - Step 1: CPU dummy-variant `sft_forward` test.
   - Step 2: `run_vla_sft.sh behavior_pi05_vla` smoke (one fwd/bwd/optimizer step + checkpoint save).
6. Milestone M6 — Loss decrease + checkpoint round-trip.
   - Step 1: Short-loss sanity then a ≥30-min run → smoothed decrease comparable to the reference.
   - Step 2: Export an SFT checkpoint to the eval format + Phase-1 eval round-trip; regression-check old `openpi` + Phase-1 eval.

Dependency summary: M1 → M2 → M3 → M4 → M5 → M6. (M3's sharding check can run in parallel with the skill port; both gate M5/M6 quality.)

## Task Breakdown

Each task includes exactly one routing tag (`coding` = Claude; `analyze` = Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Confirm the BEHAVIOR streaming dataset's chunk-partition code (`__getitem__` ignores `idx`; partition keyed on `(seed, worker_id)`, no `rank`) and specify the precise rank-aware fix (fold `rank` into the partition) so each rank loads distinct data; document the per-rank sharding contract to match the reference. | AC-5 | analyze | - |
| task2 | Document the exact reference `create_behavior_data_loader_torch` + train-loop contract (transform order, action normalize+pad, collate, optimizer/LR formulas, deterministic-RNG option) to align to. | AC-6, AC-8 | analyze | - |
| task3 | Implement `sft_forward`/`forward(ForwardType.SFT)` in `openpi_action_model.py`: tuple/dict batch, build `model.Observation`, device move, normalize 23-dim actions then pad to 32, `Pi0.compute_loss` → `.mean()`; add `gradient_checkpointing_enable/disable` pass-through. | AC-1 | coding | task2 |
| task4 | Add the SFT training build path: load `pi05_base_pytorch_new` (fp32 new-format) strict, cast bf16, no eval bf16-strict validation; keep the eval loader unchanged. | AC-3 | coding | task3 |
| task5 | Relax the `openpi_pytorch` SFT guard in `rlinf/config.py` (reject full_pi05/dsrl/value-head/non-BEHAVIOR via data-config signals); update the SFT-rejection test + the reserved-SFT-raises test. | AC-2 | coding | task3 |
| task6 | Add `OPENPI_PYTORCH` to `FSDPVlaSftWorker.build_dataloader()`; build the self-contained BEHAVIOR SFT dataloader (direct mode) under `openpi_pytorch/dataconfig/` with explicit `collate_fn` → `(model.Observation, actions)`; extend the AST import-isolation gate over `dataconfig`. | AC-4 | coding | task4, task5 |
| task7 | Direct-mode data-parity test vs `create_behavior_data_loader_torch` (prompt/images/state/actions after transforms). | AC-6 | coding | task6 |
| task8 | Port the `openpi-comet` skill window logic (`enable_gap`/`allow_left`/`allow_right`, `_build_skill_boundaries`/`_get_skill_label`) into the dataset; skill-mode parity test vs the JAX logic. | AC-7 | coding | task6 |
| task9 | Implement rank-aware streaming chunk partitioning (fold `rank` into the `(seed, worker_id)` partition) so each rank loads distinct, non-overlapping chunks; add a torchrun distinct-data test (per-rank first-batch hashes differ; 256 global = 256 unique). | AC-5 | coding | task1, task6 |
| task10 | Align FSDP/precision/optimizer/LR to the reference (bf16, grad-checkpoint, AdamW, warmup+cosine); LR-schedule unit test; retarget `behavior_pi05_vla.yaml`; add `pi0_5_pytorch.yaml`. | AC-8, AC-11 | coding | task4 |
| task11 | CPU dummy-variant `sft_forward` test + GPU smoke run of `run_vla_sft.sh behavior_pi05_vla` (one fwd/bwd/optimizer step + checkpoint save). | AC-9 | coding | task6, task9, task10 |
| task12 | Checkpoint export (RLinf FSDP SFT → eval-format `model.safetensors`+config.json+norm-stats) + Phase-1 eval round-trip test. | AC-12 | coding | task11 |
| task13 | Short-loss sanity then ≥30-min run → smoothed loss decrease comparable to the reference bands. | AC-10 | coding | task11 |
| task14 | Regression-check: old `openpi` SFT/eval paths + Phase-1 eval (success_once ~0.34) still pass. | AC-13 | coding | task5, task6 |

## Claude-Codex Deliberation

### Agreements
- Reuse RLinf's SFT runner/worker/FSDP + the vendored `Pi0.compute_loss`; do not port `train_pytorch_new.py` wholesale.
- Add `OPENPI_PYTORCH` SFT wiring (loss path, guard relax, worker dispatch) and a self-contained BEHAVIOR dataloader (lerobot OK, no installed `openpi`).
- Direct-task mode is the runnable acceptance; skill mode is a dataset capability with a parity test.
- The data-pipeline (loader, sharding, parity) is the highest-risk area and must be early-milestone work, validated before training tuning.

### Resolved Disagreements
- "Align FSDP/precision/optimizer/LR" was too vague → made concrete (bf16 params + grad-checkpoint; AdamW betas/eps/wd/clip; warmup=1000 init=peak/(warmup+1) → cosine-to-0/30000; batch 256/32) with an LR-schedule unit test.
- SFT base dtype: the base is fp32 new-format (`pi05_base_pytorch_new`), so a dedicated training build path is required (the eval bf16-strict loader would reject it) — confirmed by investigation.
- Action dim: env actions are 23-dim; `sft_forward`/loader must normalize then pad to the 32-dim model dim before `compute_loss`.
- Collate: the `Observation` dataclass needs an explicit `collate_fn` (default PyTorch collate is unsafe).
- Checkpoint round-trip: RLinf FSDP checkpoints don't natively match the eval directory format → add an explicit export + round-trip test.
- SFT validation can't rely on `env.*` (SFT configs have none) → use data-config/`config_name` signals.
- "monotonic loss" is unrealistic for stochastic SFT → directional/smoothed decrease (DEC-3); numeric bands are advisory unless deterministic RNG + matched data order are adopted.
- Existing tests must be updated: the "SFT rejected for openpi_pytorch" test and the "reserved SFT methods raise" test now assert the supported behavior.

### Convergence Status
- Final Status: `converged`
- Codex first-pass analysis (v1) + two convergence rounds; round 2 returned only tightening required-changes (mandatory training builder, gradient-checkpoint pass-through, action transform order, explicit collate, the `no_shard` decision, test updates), all adopted, leaving no opposing positions — only the user decisions below.

## Pending User Decisions

All decisions were resolved during the discussion; none remain `PENDING`.

- DEC-1: Distributed data-parallel mechanism for SFT.
  - Claude Position: RLinf-native per-rank data loading (fits the worker/checkpoint machinery; lower risk).
  - Codex Position: rank-0 p2p fanout is closer to the reference loss curve, but invasive; whichever is chosen, the data order/RNG must be settled to claim loss comparability.
  - Tradeoff Summary: rank-0 fanout maximizes loss-curve parity but diverges from RLinf's worker; per-rank loading fits RLinf but only if the streaming chunk partition is made rank-aware so it shards distinct data per rank.
  - Decision Status: **Use rank-aware streaming chunk partitioning** — fold `rank` into the BEHAVIOR streaming dataset's existing `(seed, worker_id)` chunk partition so each rank loads distinct, non-overlapping data. `DistributedSampler` is NOT the mechanism here: the streaming dataset's `__getitem__` ignores `idx`, so `DistributedSampler` has no effect and all ranks would otherwise see identical data (the "256 global batch" being only 32 unique samples replicated 8×; the reference adopted rank-0 fanout for the same reason). This supersedes the earlier "verify `DistributedSampler` shards" framing — the streaming root cause is now identified — and makes RLinf's per-rank data-loading consistent with the reference's intent (256 unique samples per global batch).

- DEC-2: Canonical SFT start checkpoint.
  - Claude Position: `pi05_base_pytorch_new` (what the reference trainer loads).
  - Codex Position: should converge on `pi05_base_pytorch_new` unless the user rejects the reference's starting point.
  - Tradeoff Summary: using the reference's exact base maximizes loss-curve alignment.
  - Decision Status: **`/mnt/public/xzxuan/models/pi05_base_pytorch_new`** (new-format, fp32) — loaded directly by the new `Pi0`, cast to bf16 for training.

- DEC-3: Loss-acceptance strictness.
  - Claude Position: directional/smoothed decrease comparable to the reference bands.
  - Codex Position: hard numeric bands are only meaningful with deterministic per-sample RNG + matched data order.
  - Tradeoff Summary: directional avoids flaky failures from RNG/data-order; hard bands require deterministic seeding (and likely rank-0 fanout).
  - Decision Status: **Directional / smoothed decrease** in the same ballpark as the reference (e.g. ~0.25 → <~0.05 over the first ~100–200 steps and continuing down); exact per-step numbers are advisory.

- DEC-4: Skill-mode scope in this plan.
  - Claude Position: dataset capability + parity test; direct-task launch is the runnable acceptance.
  - Codex Position: decide whether skill mode must run end-to-end or only pass dataset/parity tests.
  - Tradeoff Summary: a full skill run adds a second long training; the parity test proves alignment cheaply.
  - Decision Status: **Dataset capability + parity test** (aligned with `openpi-comet`); the runnable training acceptance stays the direct-task launch.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers. These terms are for plan documentation only. Use descriptive, domain-appropriate naming in code instead.
- Follow repo conventions (Google Python style, Ruff lint + format clean on all changed files, type hints/docstrings on public APIs; use logging, not `print`).
- Keep the new SFT data path self-contained: no `import openpi` / `from openpi …` (installed package) under `openpi_pytorch/` (`lerobot` is allowed); the AST import-isolation gate must cover the new `dataconfig`.
- Do not modify the old `rlinf/models/embodiment/openpi/` package or break the Phase-1 eval; SFT changes are additive (plus the necessary guard relax + test updates + worker dispatch branch).
- Keep vendored model-core files byte-close to upstream; namespace-convert imports only.

--- Original Design Draft Start ---

# Task: Support an Updated, More Optimized PyTorch-Based OpenPI 0.5 Model

I need you to deliver a major feature: support for a newer, more optimized version of the OpenPI 0.5 model based on PyTorch.

## Repositories Involved

There are two core repositories:

1. **`/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed`** — This repository already implements the optimized OpenPI 0.5 model. The optimized PyTorch source code lives under `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new`. The original PyTorch code is under `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch`, but that version (a) is not aligned with the JAX version of OpenPI, (b) does not support mixed precision, and (c) is poorly written in places — for example, it modifies the `transformers` library's code, which makes migration difficult. For these reasons, we re-implemented a better PyTorch version of the OpenPI model, and now it needs to be migrated into the `RLinf_pi05` repository.

2. **`/mnt/public/xzxuan/repos/RLinf_pi05`** — The target repository for migration. This repository currently still uses the old PyTorch code. The PI 0.5-related code is in `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi`, with `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/openpi_action_model.py` being the core file. Note that the current code is also written by replacing/patching the `transformers` library, and it is poorly written. So I need you to carefully read the code in `openpi-comet-pytorch-mixed` and migrate it into `RLinf_pi05`.

**Additional reference — the original OpenPI repository:** `/mnt/public/xzxuan/repos/openpi`. This is the upstream OpenPI repo. Its recommended path is the JAX version, but it also provides a PyTorch version (which is consistent with `openpi-comet-pytorch-mixed`'s `models_pytorch` and with the current `RLinf_pi05`). As you can see in `/mnt/public/xzxuan/repos/openpi/README.md`, it also says to "Apply the transformers library patches." This is very undesirable, which is exactly why we re-wrote our own version.

## The migration has two phases:

### Phase 1: Support action generation with the new PyTorch version.

In `RLinf_pi05`, I want to be able to run the script `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_eval`. Its purpose is to let the PI 0.5 model actually interact with the BEHAVIOR environment: the environment provides observations, the observations are fed to the model, and the model generates actions for the environment to execute. This phase does **not** involve model training — only action sampling. The relevant functions are:
- `sample_actions` in `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new/pi0.py`
- `predict_action_batch` in `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/openpi_action_model.py`

### Phase 2: Support SFT training of the new PyTorch version on BEHAVIOR tasks.

I require you to fully migrate the SFT script from `openpi-comet-pytorch-mixed` into the `RLinf_pi05` repository, such that running `bash run.sh` works. Building on Phase 1, this involves:
- The model's `compute_loss` function (used for training).
- The FSDP distributed training code at `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts/train_pytorch_new.py`.
- **Most importantly**, the loading and processing of BEHAVIOR data. This is critical — carefully understand the full logic of `_data_loader.create_behavior_data_loader_torch` inside the `build_datasets` function.

The current `RLinf_pi05` supports FSDP-based SFT training for the *old* PI 0.5 PyTorch code, but it does not natively support the BEHAVIOR environment. I previously wrote one — its launch script is `bash examples/sft/run_vla_sft.sh behavior_pi05_vla` (run from within `RLinf_pi05`). However, it certainly has a lot of bugs, so I need you to carefully revise it on this basis, ensuring it is fully aligned with the `openpi-comet-pytorch-mixed` training code — including training hyperparameters, training configuration, and dataloader logic. In particular, ensure that the code for **streaming data from the BEHAVIOR dataset is fully aligned**.

One important point here: SFT must support **two modes**:
1. Directly inputting "turn on radio" and having the model output actions — this is what `openpi-comet-pytorch-mixed` currently implements.
2. A finer-grained, subtask-level training that I implemented (in JAX) under `/mnt/public/xzxuan/repos/openpi-comet`: the model takes four subtasks as input and then outputs actions, which is more fine-grained. Its launch script is `bash run.sh jax_skill` (run from within `openpi-comet`). The core of this is the three parameters `enable_gap`, `allow_left`, and `allow_right`.

I require that your newly fixed BEHAVIOR dataset also supports this configuration. The current `rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py` also has some skill-related code, but it may differ substantially from `openpi-comet`'s. I require it to be **fully aligned** with `openpi-comet`.

In other words, my dataset additionally implements skill-augmented training — i.e., the training input is no longer fixed to "turn on radio." Previously this was only trained and tested on the VLM (Qwen3 4B); its launch script is `bash examples/sft/run_vlm_sft.sh behavior_qwen3_vlm_sft_agentic`. Meanwhile, in the `/mnt/public/xzxuan/repos/openpi-comet` repository, I implemented full support for skill-granularity OpenPI 0.5 training, although that training is the JAX version.

---

## Finally — my coding standards and requirements:

**1.** You need to create a new directory `openpi_pytorch` under `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment`, containing the core PyTorch code you need to migrate. You can directly copy the code from `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new` into it and then modify it. The goal: after migration, the `openpi_pytorch` directory should contain a complete implementation of the new PyTorch code, so that it does **not** need to import OpenPI things from the environment (as the current `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/openpi_action_model.py` does — it imports from `/mnt/public/xzxuan/.venv_pi/lib/python3.10/site-packages/openpi/`). I want all components implemented self-contained.

The directory structure under `openpi_pytorch` should be similar to `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi`. The `__init__.py` should handle model creation and related configuration. You also need `dataconfig` and `policies`; for now, only consider the BEHAVIOR environment — nothing else is needed. You can place a `utils` subdirectory for the helper components needed to build the OpenPI model, including `gemma.py`, `pi0_config.py`, `siglip.py`, etc. But there must be an `openpi_action_model.py` as the model entry point.

**2.** Regarding your newly implemented `openpi_action_model.py`: note that the original `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/openpi_action_model.py` is very complex, because it involves many environments and many algorithms — including SFT, RL, DAgger, etc. — so it contains a lot of stuff. But I require your new `openpi_action_model.py` to focus on only **two** things:
- SFT loss computation (for model training).
- Eval action sampling.

Only these two — ignore everything else for now. However, the high-level interface must remain consistent with the current old PyTorch code, so that the interfaces I call during SFT and eval stay the same.

**3.** For now, both SFT and eval only concern BEHAVIOR-related code. The original `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/dataconfig` contains many BEHAVIOR-related files; I'm not entirely sure of the exact logic of each one, so you only need to select the ones you need and migrate those. `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/dataconfig/behavior_dataset.py` is probably the important one — ensure its implementation logic is **fully aligned** with `openpi-comet-pytorch-mixed`.

**4.** You'll notice the original `RLinf_pi05` code has a config `full_pi05 = getattr(actor_model_config, "full_pi05", False)`. If it's `True`, the code goes through `OpenPi05FullForRLActionPrediction` and uses `/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi/static_kv_cache.py`. This is the **full** version of PI 0.5: it supports standalone VLM SFT for PI 0.5, and during generation, the VLM autoregressively generates tokens first, then the action expert generates actions via flow matching. However, this is for later — your current version does **not** need to consider the full version.

**5.** You need to write **two** configs, for SFT training and BEHAVIOR eval respectively:
- `/mnt/public/xzxuan/repos/RLinf_pi05/examples/sft/config/behavior_pi05_vla.yaml` (already exists — just update its contents to be fully aligned with `openpi-comet-pytorch-mixed`).
- `/mnt/public/xzxuan/repos/RLinf_pi05/examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml` (you can model it after `behavior_ppo_openpi_pi05_eval.yaml`).

Additionally, you need to create a new model config `pi0_5_pytorch.yaml` under both `examples/sft/config/model` and `examples/embodiment/config/model`. It can retain only the needed parameters; things like `full_pi05`, `dsrl`, `value head`, and RL-related parameters can all be removed. In short, the config should be clean and minimal.

**6.** You need to carefully compare the FSDP training code to ensure that `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts/train_pytorch_new.py` and the current RLinf PyTorch code are **strictly aligned** in logic.

**7.** Both `openpi-comet` repositories use the environment `/mnt/public/xzxuan/repos/openpi-comet/.venv`; `RLinf_pi05` uses `/mnt/public/xzxuan/.venv_pi`.

---

## Acceptance Criteria

**1. Eval:**
Running `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_eval` should run successfully. To speed up the smoke test, you can set `eval_rollout_epoch=1`, then `max_episode_steps = max_steps_per_rollout_epoch = 64`.

However, simply running is not the final goal. The final goal: set `eval_rollout_epoch=8` and `max_episode_steps = max_steps_per_rollout_epoch = 4096`. After running, ensure the final `eval/success_once` recorded in the log is around **0.3**. This is because my current run of `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_eval` produced results in `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260602-12:54:50-behavior_ppo_openpi_pi05_eval`, and your measured success rate should be close to that (**0.32**) — because your eval logic should be strictly identical, and there should be no difference at the code level.

**2. SFT:**
Ensure you can run `bash examples/sft/run_vla_sft.sh behavior_pi05_vla`. Similarly, you can first use small parameters to confirm it runs. But the final acceptance criterion is alignment with the `openpi-comet-pytorch-mixed` code, and that you can observe a clear loss decrease during training — which requires training for a while (at least 30 minutes).

## Current Objective
Currently, you have completed stage 1, which is supporting eval. Now let's begin stage 2! That is, fully supporting SFT training of the new-version PyTorch openpi for the behavior scenario in the current repository! You can refer to `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log` to compare the loss!
--- Original Design Draft End ---
