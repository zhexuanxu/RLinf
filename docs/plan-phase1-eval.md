# Phase 1 — Self-Contained PyTorch OpenPI 0.5 Action Generation (BEHAVIOR Eval)

## Goal Description

Migrate the new, optimized PyTorch implementation of OpenPI 0.5 (from `openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new`) into the `RLinf_pi05` repository as a **new, fully self-contained model package** at `rlinf/models/embodiment/openpi_pytorch/`, and use it for **eval / action generation only** on the BEHAVIOR environment.

The single runnable deliverable of this plan is:

```
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_eval
```

It must let the new PyTorch PI 0.5 model interact with the BEHAVIOR environment (observation → model → action), reproducing the behaviour of the existing old-PyTorch eval (`behavior_ppo_openpi_pi05_eval`, which logged `eval/success_once ≈ 0.32`). This phase covers **action sampling only — no training**.

Key properties:
- The new package does **not** import OpenPI from the installed venv (`/mnt/public/xzxuan/.venv_pi/.../site-packages/openpi/`) and does **not** patch the `transformers` library. All components are vendored (gemma, siglip, pi0 core, BEHAVIOR transforms/tokenizer/normalization).
- A new `openpi_action_model.py` is the model entry point, focused on the two responsibilities the user named — **eval action sampling** (the runnable target of this plan) and **SFT loss** (interface reserved/stubbed here; implemented in the deferred Phase 2). Its high-level interface stays consistent with the current old PyTorch `openpi_action_model.py` so existing eval (and later SFT) call sites are unchanged.
- A new embodied model type is registered and selected via two clean, minimal configs (model config + eval config).
- The **out-of-scope** items (Phase 2 SFT training, FSDP migration, BEHAVIOR dataloader/streaming alignment, skill-granularity training, the `full_pi05` full-VLM path) are recorded under "Out of Scope / Deferred to Phase 2" so no draft requirement is lost — only deferred.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. The **primary correctness gate is AC-6** (deterministic numerical action parity, per user decision DEC-2); env-level success (AC-8) is a directional confirmation (per DEC-1).

- AC-1: The `rlinf/models/embodiment/openpi_pytorch/` package is self-contained and imports no installed `openpi`.
  - Positive Tests (expected to PASS):
    - A static (AST-based, not just grep) scan of every module under `rlinf/models/embodiment/openpi_pytorch/` finds zero `import openpi` / `from openpi …` statements (intra-package imports such as `from rlinf.models.embodiment.openpi_pytorch…` are allowed).
    - `python -c "import rlinf.models.embodiment.openpi_pytorch"` succeeds.
    - The new package's eval path still imports successfully when the installed top-level `openpi` package is hidden from `sys.path`.
  - Negative Tests (expected to FAIL):
    - Adding a `from openpi import transforms` (or any site-packages `openpi` import) anywhere in the package causes the static scan to FAIL (the check rejects it).
    - Any reliance on a `transformers` monkeypatch / source modification is rejected by the same scan / review.

- AC-2: A new embodied model type is registered and selectable; the existing `openpi` model path is untouched.
  - Positive Tests (expected to PASS):
    - After import, `SupportedModel("openpi_pytorch")` resolves and is a member of `EMBODIED_MODEL`; `get_model` with this model type instantiates the new model.
    - The existing `behavior_ppo_openpi_pi05_eval` (old `openpi`) config still runs and is byte-for-byte unchanged in code (`git diff` shows no edits under `rlinf/models/embodiment/openpi/`).
  - Negative Tests (expected to FAIL):
    - An unknown/typo model type still raises a clear error from registry lookup / `validate_embodied_cfg`.
    - A change that modifies the old `openpi` builder or files is rejected in review.

- AC-3: Minimal, clean model and eval configs exist for the new model.
  - Positive Tests (expected to PASS):
    - `examples/embodiment/config/model/pi0_5_pytorch.yaml` and `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml` exist, compose under Hydra, select `model_type: openpi_pytorch`, and reference the BEHAVIOR env.
    - The model config retains only fields needed for eval action sampling; it omits `full_pi05`, `dsrl*`, value-head, and RL-only keys.
  - Negative Tests (expected to FAIL):
    - A config that reintroduces `full_pi05`/`dsrl`/value-head fields for this model fails validation (or those paths raise per AC-10).
    - Pointing the new model at a non-BEHAVIOR env raises a clear, explicit error.

- AC-4: The `predict_action_batch` interface is a drop-in for the eval worker.
  - Positive Tests (expected to PASS):
    - The new `openpi_action_model` exposes `predict_action_batch(env_obs, mode="eval", compute_values=…, **kwargs)` returning `(actions, result_dict)` with `actions` shaped `[B, action_chunk, action_env_dim]` (e.g. `[B, 32, 23]`), float32.
    - The rollout worker reaches the new model through the **same** dispatch branch as the old `OPENPI` type (`MultiStepRolloutWorker.predict()` in `rlinf/workers/rollout/hf/huggingface_worker.py`), with **no change** to runner or eval-loop logic beyond adding the new model type to that branch.
  - Negative Tests (expected to FAIL):
    - A return shape other than `[B, action_chunk, action_env_dim]` fails a contract assertion in the parity/smoke test.
    - The eval worker requiring a bespoke (non-OPENPI) code path for this model is rejected (it must share the OPENPI eval path).

- AC-5: Checkpoint conversion and strict load.
  - Positive Tests (expected to PASS):
    - A conversion step produces a **new-format** checkpoint on disk from the old-format `jax_task0000_sft_29999/model.safetensors` using the vendored `checkpoint_format.old_to_new_state_dict` (per DEC-3); the same `physical-intelligence/behavior/norm_stats.json` is reused.
    - The new model loads the converted checkpoint with strict matching (`strict=True`, or an explicit, documented allowlist of any missing/unexpected keys with rationale), and validates each tensor's shape and dtype.
    - The load step logs parameter count and a checksum/hash of the converted weight keys and norm stats.
  - Negative Tests (expected to FAIL):
    - A checkpoint missing/renaming a required tensor (any LLM expert-0/expert-1, SigLIP image, `action_in_proj`, `action_out_proj`, or pi0.5 time-MLP key) causes a hard failure (no silent `strict=False` swallow).
    - A shape or dtype mismatch on any tensor fails fast with a clear message.

- AC-6: Deterministic numerical action parity vs. the old path (PRIMARY gate).
  - Positive Tests (expected to PASS):
    - With a single fixed BEHAVIOR observation fixture (fixed images in fixed key order `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`; fixed prompt; fixed state), both the old `OpenPi0ForRLActionPrediction` path and the new path are run with the **same injected flow-matching noise tensor**, **same `num_steps`**, same device, same eval mode, same dtype.
    - The max-absolute difference of the **full 32-dim unnormalized model action** is within `tol_model`, and of the **23-dim env-sliced unnormalized action** is within `tol_env`, evaluated separately, for **FP32** (tight tolerance) and **BF16** (looser tolerance). The exact fixture, noise, `num_steps`, tolerances, device, and seed are pinned in the parity-contract task (task9) before implementation.
  - Negative Tests (expected to FAIL):
    - Permuting image key order, dropping state/prompt normalization, or changing `num_steps` makes the parity diff exceed tolerance and the test FAILS.
    - Comparing only the env-sliced action (hiding a model-space bug behind slicing) is insufficient — both comparisons are required.

- AC-7: The smoke eval runs end-to-end.
  - Positive Tests (expected to PASS):
    - `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_eval` with `eval_rollout_epoch=1` and `max_episode_steps = max_steps_per_rollout_epoch = 64` completes without error and logs an `eval/success_once` value.
  - Negative Tests (expected to FAIL):
    - A missing checkpoint/config path produces a clear error rather than a silent hang.
    - The run does not crash during model construction or action sampling.

- AC-8: Full eval reproduces a directional success rate comparable to the old run.
  - Positive Tests (expected to PASS):
    - A full run with `eval_rollout_epoch=8` and `max_episode_steps = max_steps_per_rollout_epoch = 4096` yields `eval/success_once` clearly in the same ballpark as the old run's 0.32 (directional acceptance per DEC-1), consistent with eval logic being functionally identical.
  - Negative Tests (expected to FAIL):
    - A `success_once` collapsed near zero or grossly inconsistent with 0.32 (e.g. well under ~0.1) indicates a real defect and fails review.

- AC-9: BEHAVIOR preprocessing/postprocessing parity is verified on the fixture.
  - Positive Tests (expected to PASS):
    - On the fixed fixture, the new pipeline's tokenized prompt and mask, resized/padded images (224×224, range `[-1, 1]`), normalized state, and unnormalized actions match the old pipeline's corresponding tensors within tolerance.
    - The normalization stats are sourced from the same `norm_stats.json` as the old path.
  - Negative Tests (expected to FAIL):
    - Changing the image resize/pad behaviour or the normalization stats source breaks the match and is caught by the parity check.

- AC-10: Unsupported paths fail loudly.
  - Positive Tests (expected to PASS):
    - Invoking the SFT-loss method raises `NotImplementedError` with a pointer to Phase 2.
    - Enabling `dsrl`, value-head, `full_pi05`, a non-pi0.5 checkpoint, or a non-BEHAVIOR env raises a clear, explicit error.
  - Negative Tests (expected to FAIL):
    - Any of those unsupported configurations silently running a partial or incorrect path is rejected.

- AC-11: The SFT-loss responsibility is reserved at the Phase 1/Phase 2 boundary.
  - Positive Tests (expected to PASS):
    - The new `openpi_action_model` exposes the SFT-loss method name matching the old interface, raising `NotImplementedError` (reserved), so eval is the only runnable target in this plan.
  - Negative Tests (expected to FAIL):
    - This plan implementing training/FSDP/dataloader changes (those belong to Phase 2) is out of scope and rejected for this plan.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)
A fully self-contained `rlinf/models/embodiment/openpi_pytorch/` package for BEHAVIOR eval that includes: the vendored model core (the ten `models_pytorch_new` files: `pi0.py`, `gemma.py`, `siglip.py`, `pi0_config.py`, `model.py`, `pointnet.py`, `lora.py`, `checkpoint_format.py`, `utils.py`, `__init__.py`), vendored BEHAVIOR `policies` + `dataconfig` (image/state/prompt transforms, tokenizer, Normalize/Unnormalize), a `__init__.py` that performs model creation + config, a new `openpi_action_model.py` entry point preserving the old high-level interface (with a reserved SFT-loss stub), a checkpoint conversion script producing a new-format checkpoint, registration via `register_model`, the two minimal configs, the deterministic numerical parity test, an import-isolation gate, plus loud errors on all unsupported paths — all verified by the smoke run and the full directional eval.

### Lower Bound (Minimum Acceptable Scope)
The new package's eval path runs `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_eval` end to end using the new vendored `sample_actions`, with no installed-`openpi` import anywhere in the eval path, passing the deterministic action-parity test (AC-6) against the old path on a fixed fixture, and producing a directional `eval/success_once` comparable to the old 0.32.

### Allowed Choices
- Can use:
  - The ten-file model core copied/adapted from `openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new`, with imports namespace-converted to `rlinf.models.embodiment.openpi_pytorch.*`.
  - The vendored `checkpoint_format` converter (`old_to_new_state_dict`) and a one-off conversion script that materializes a new-format checkpoint on disk (DEC-3).
  - The current old RLinf `openpi/policies/behavior_policy.py` and `openpi/dataconfig/behavior_*` as a behavioural reference, with the minimal transform base classes (composite transform, Normalize/Unnormalize, tokenizer wrapper) vendored into the new package.
  - `torch`, `einops`, `safetensors`, `numpy` (and `torchvision` only if already used by the vendored core).
- Cannot use:
  - Any `import openpi` / `from openpi …` from the installed venv inside the new eval path.
  - `transformers` monkeypatching or source modification.
  - The `full_pi05` full-VLM-autoregressive path and `static_kv_cache.py`; `dsrl`/value-head/RL code; non-BEHAVIOR environments.
  - Edits to the old `rlinf/models/embodiment/openpi/` package, or to runner/eval-loop logic beyond adding the new model type to the existing OPENPI dispatch branch in `MultiStepRolloutWorker.predict()`.

> **Note on Deterministic Designs**: The draft fixes several choices (self-contained package, BEHAVIOR-only, reuse the existing checkpoint's weights, identical eval logic). Where the draft is prescriptive, the bounds above converge to that specification; the remaining latitude is in code organization and the exact vendoring of transform base classes.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Eval data flow to reproduce (old vs. new must match at every stage):

```
env_obs (main/wrist images [B,H,W,C], state [B,32], task strings)
   │  obs_processor            → policy-input dict (observation/image*, observation/state, prompt)
   │  input_transform          → resize+pad images to 224 in [-1,1]; inject default prompt;
   │                             tokenize prompt (PaliGemma tokenizer) → tokenized_prompt + mask;
   │                             Normalize state (norm_stats)
   │  build model.Observation  → images{base_0_rgb,left_wrist_0_rgb,right_wrist_0_rgb}, image_masks,
   │                             state, tokenized_prompt, tokenized_prompt_mask
   ▼
new Pi0.sample_actions(obs, num_steps=<old num_steps>, noise=<injected>, rng=<seeded>)
   │   → [B, action_horizon=32, action_dim=32]  (model action space)
   ▼
   │  output_transform         → Unnormalize actions; slice to action_chunk; take env action_env_dim=23
   ▼
actions [B, action_chunk, 23]  +  result_dict (forward_inputs, etc., as the old contract requires)
```

Weight path (DEC-3): `jax_task0000_sft_29999/model.safetensors` (old `paligemma_with_expert.*` layout, 812 BF16 tensors, `config.json`: action_dim=32, action_horizon=32, gemma_2b + gemma_300m, precision=bfloat16) → `old_to_new_state_dict` → new-format checkpoint on disk → strict load into `Pi0(Pi0Config(pi05=True, action_horizon=32, action_dim=32, …))`.

Precision policy: prove FP32 deterministic parity first (isolates architecture bugs from precision noise), then run BF16 to match the old run's `precision: bfloat16` for the final directional eval.

### Relevant References
- `rlinf/models/embodiment/openpi/openpi_action_model.py` — old `OpenPi0ForRLActionPrediction`; reference for `predict_action_batch`, the input/output transform pipeline, and the public interface to preserve.
- `rlinf/models/embodiment/openpi/__init__.py` — old `get_model`: checkpoint discovery, `strict=False` load, `load_norm_stats`, `setup_wrappers`.
- `rlinf/models/embodiment/openpi/policies/behavior_policy.py`, `rlinf/models/embodiment/openpi/dataconfig/behavior_*.py` — BEHAVIOR transform/normalization reference (`config_name: pi05_behavior`).
- `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new/` — the new self-contained model core (`pi0.py:sample_actions`, `pi0_config.py`, `gemma.py`, `siglip.py`, `model.py:Observation`, `checkpoint_format.py:old_to_new_state_dict`).
- `rlinf/models/__init__.py` — `register_model` / `_MODEL_REGISTRY` / `get_model` (single source of truth for `SupportedModel` + `EMBODIED_MODEL`).
- `rlinf/config.py` — `SupportedModel`, `EMBODIED_MODEL`, `validate_embodied_cfg`.
- `rlinf/workers/rollout/hf/huggingface_worker.py` — `MultiStepRolloutWorker.predict()` model-type dispatch and the `predict_action_batch` call site.
- `examples/embodiment/eval_embodiment.sh` → `examples/embodiment/eval_embodied_agent.py` → `EmbodiedEvalRunner` (`rlinf/runners/embodied_eval_runner.py`) — eval entry chain (unchanged).
- `examples/embodiment/config/behavior_ppo_openpi_pi05_eval.yaml`, `examples/embodiment/config/model/pi0_5.yaml`, `examples/embodiment/config/env/behavior_r1pro.yaml` — config templates for the new minimal configs.
- `/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999/` — checkpoint + `physical-intelligence/behavior/norm_stats.json`.
- `logs/20260602-12:54:50-behavior_ppo_openpi_pi05_eval/` — old run reference (`eval/success_once ≈ 0.32`).

## Dependencies and Sequence

### Milestones
1. Milestone M1 — Self-contained model core: stand it up and prove isolation.
   - Phase A: Create the package skeleton; copy the ten `models_pytorch_new` files into the new package; namespace-convert their imports.
   - Phase B: Add the AST-based import-isolation gate; run a synthetic `sample_actions` smoke (random `model.Observation`) to confirm the core runs without installed `openpi`.
2. Milestone M2 — BEHAVIOR eval I/O and the action-model entry point.
   - Step 1: Vendor/port the BEHAVIOR preprocessing/postprocessing (image resize+pad, state/prompt handling, tokenizer, Normalize/Unnormalize) and the minimal transform base classes, producing a `model.Observation`.
   - Step 2: Implement the new `openpi_action_model.py` wrapping `Pi0.sample_actions`, preserving the old `predict_action_batch` contract and reserving the SFT-loss method (`NotImplementedError`); raise clear errors on unsupported paths.
3. Milestone M3 — Checkpoint conversion + load.
   - Step 1: Add a one-off conversion script (`old_to_new_state_dict`) that writes a new-format checkpoint; reuse the existing `norm_stats.json`.
   - Step 2: Strict load with per-tensor shape/dtype validation and a checksum/key report; define the FP32-then-BF16 precision policy.
4. Milestone M4 — Registration + configs.
   - Step 1: Register the new type via `register_model` (populating `SupportedModel`/`EMBODIED_MODEL`); add the new model type to the OPENPI dispatch branch in the rollout worker.
   - Step 2: Author the minimal `pi0_5_pytorch.yaml` model config and `behavior_ppo_openpi_pi05_pytorch_eval.yaml` eval config.
5. Milestone M5 — Deterministic numerical parity (the key gate).
   - Step 1: Pin the parity contract (fixture, prompt, image order, noise, `num_steps`, dtype/autocast, device, seed, FP32/BF16 tolerances).
   - Step 2: Implement and pass the old-vs-new action-parity test (32-dim unnormalized and 23-dim env slice compared separately).
6. Milestone M6 — Runnable eval.
   - Step 1: Smoke eval (`eval_rollout_epoch=1`, steps=64).
   - Step 2: Full eval (`eval_rollout_epoch=8`, steps=4096); confirm directional `eval/success_once` comparable to 0.32.

Dependency summary: M1 → M2 → (M3, M4) → M5 → M6. M3 (checkpoint) and M4 (registration/config) both depend on M2 but are independent of each other; M5 needs the action model (M2), the loaded checkpoint (M3), and the pinned contract; M6 needs configs (M4) plus a passing parity test (M5).

## Task Breakdown

Each task includes exactly one routing tag (`coding` = implemented by Claude; `analyze` = executed via Codex `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Document the exact old eval contract (predict_action_batch I/O, transform order, tokenizer asset, norm_stats usage, precision, num_steps) and the new `models_pytorch_new` dependency graph for `sample_actions`. | AC-4, AC-9 | analyze | - |
| task2 | Create the `openpi_pytorch` package skeleton; copy the ten model-core files; namespace-convert imports to `rlinf.models.embodiment.openpi_pytorch.*`. | AC-1 | coding | task1 |
| task3 | Add the AST-based import-isolation gate and a synthetic `sample_actions` smoke (random `model.Observation`). | AC-1 | coding | task2 |
| task4 | Vendor/port BEHAVIOR preprocessing/postprocessing (image resize+pad, state, prompt inject, tokenize+mask, Normalize/Unnormalize) and minimal transform base classes; build `model.Observation`. | AC-9 | coding | task2 |
| task5 | Implement `openpi_action_model.py`: constructor, `predict_action_batch` preserving the old contract, reserved SFT-loss `NotImplementedError`, loud errors on unsupported paths. | AC-4, AC-10, AC-11 | coding | task4 |
| task6 | Add the checkpoint conversion script (old→new format) and strict load + per-tensor shape/dtype validation + checksum/key report in the package `__init__`/`get_model`. | AC-5 | coding | task2 |
| task7 | Register the new model type via `register_model` (SupportedModel/EMBODIED_MODEL/validation) and add it to the OPENPI dispatch branch in `MultiStepRolloutWorker.predict()`. | AC-2 | coding | task5 |
| task8 | Author `examples/embodiment/config/model/pi0_5_pytorch.yaml` and `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml` (minimal; no full_pi05/dsrl/value-head/RL fields). | AC-3 | coding | task7 |
| task9 | Pin the numerical parity contract: fixture spec, prompt, image order, injected noise, num_steps, device, seed, dtype/autocast, and concrete FP32/BF16 tolerances. | AC-6 | analyze | task1 |
| task10 | Implement the deterministic old-vs-new action-parity test (32-dim unnormalized and 23-dim env slice compared separately, FP32 then BF16). | AC-6 | coding | task5, task6, task9 |
| task11 | Run the smoke eval (epoch=1, steps=64) and fix any wiring/runtime issues. | AC-7 | coding | task8, task10 |
| task12 | Run the full eval (epoch=8, steps=4096); confirm directional `eval/success_once` comparable to 0.32. | AC-8 | coding | task11 |

## Claude-Codex Deliberation

### Agreements
- `openpi_pytorch` as a new, isolated model type is the right boundary; the old `openpi` path stays unchanged.
- Preserving `predict_action_batch(env_obs, mode, compute_values, **kwargs) -> (actions, result_dict)` is the correct integration target; the rollout worker should reach the new model via the existing OPENPI eval branch.
- Eval-only scope is correct; SFT loss is a reserved/stubbed interface in this plan.
- Copying the ten self-contained model-core files into RLinf (with namespace conversion and an import-isolation gate) is the right migration mechanic.
- Separate model/eval YAMLs (not mutating the working OpenPI eval config) are preferable.
- Preprocessing/postprocessing parity is a first-class acceptance concern.

### Resolved Disagreements
- Checkpoint nature: Codex flagged that the converter must not be assumed to consume a raw JAX checkpoint. Investigation showed `jax_task0000_sft_29999/model.safetensors` is **old PyTorch layout** (`paligemma_with_expert.*`, 812 BF16 tensors), which `old_to_new_state_dict` is purpose-built to convert. Resolution: reuse those weights; per user decision DEC-3, **pre-convert** to a new-format checkpoint on disk and load it directly.
- Parity gate primacy: env `success_once` is noisy and can hide preprocessing/checkpoint bugs. Resolution (DEC-2): a deterministic numerical action-parity test is the **primary** gate (AC-6); env success (AC-8) is a directional confirmation (DEC-1).
- Checkpoint load strictness: group-level "required keys exist" is too coarse. Resolution: tensor-level strict load (`strict=True` or a documented allowlist) with per-tensor shape/dtype validation (AC-5).
- Parity contract precision: "within tolerance" is underspecified. Resolution: pin fixture, prompt, image order, noise, `num_steps`, dtype/autocast, device, seed, and concrete FP32/BF16 tolerances in task9 before implementation (AC-6).
- Import-isolation method: a runtime import test is insufficient because `openpi` is installed. Resolution: AST-based static scan as the gate (AC-1).
- Precision policy: prove FP32 numerical parity first, then run BF16 to match the old run's `precision: bfloat16` for the final eval.
- Registration mechanics: use `register_model` as the single source of truth (it populates `SupportedModel` and `EMBODIED_MODEL`) rather than editing separate registries.
- Worker dispatch: the new type takes the same eval path as old `OPENPI`; no runner/eval-loop changes.

### Convergence Status
- Final Status: `converged`
- Codex first-pass analysis (v1) + two convergence rounds with a second Codex pass; round 2 returned only tightening required-changes (AC-5 strictness, AC-6 contract pinning), both adopted, leaving no opposing positions.

## Pending User Decisions

All decisions were resolved during the discussion; none remain `PENDING`.

- DEC-1: Final acceptance strictness for `eval/success_once`.
  - Claude Position: hard band (±0.05 of 0.32) for a crisp pass/fail.
  - Codex Position: env success is noisy; treat it as secondary to deterministic parity.
  - Tradeoff Summary: a hard band risks flaky failures from environment stochasticity; a directional target keeps env success as a sanity check while the deterministic test carries correctness.
  - Decision Status: `Directional / close enough` — `success_once` clearly comparable to 0.32 is acceptable; not a hard numeric fail.

- DEC-2: Whether to require a deterministic numerical action-parity test as the primary gate.
  - Claude Position: yes — most robust signal for catching preprocessing/checkpoint bugs.
  - Codex Position: yes — promote it above env success.
  - Tradeoff Summary: adds a fixture/harness cost but yields a deterministic, reproducible correctness gate independent of env noise.
  - Decision Status: `Yes — primary gate`; env `success_once` is the final confirmation.

- DEC-3: Checkpoint loading approach for the new self-contained model.
  - Claude Position: reuse `jax_task0000_sft_29999` weights + convert via `old_to_new_state_dict`.
  - Codex Position: pin the path precisely; do not assume raw-JAX conversion.
  - Tradeoff Summary: converting at load time avoids an extra artifact but mixes conversion with runtime; pre-converting to disk yields a clean strict-load path and a reproducible artifact.
  - Decision Status: `Pre-convert to a new-format checkpoint` on disk via a one-off script, then load it directly (strict); reuse the same `norm_stats.json`.

## Out of Scope / Deferred to Phase 2

Recorded here so no draft requirement is lost — these are deferred, not dropped. They belong to the SFT migration covered by a later plan (the draft's Phase 2):
- Full SFT training of the new PyTorch model on BEHAVIOR, such that `bash examples/sft/run_vla_sft.sh behavior_pi05_vla` (and the source repo's `bash run.sh`) work with a clear training loss decrease (the draft's SFT acceptance criterion, ≥30 min training).
- The model's `compute_loss` (training) — implemented behind the SFT-loss interface reserved in this plan.
- The FSDP distributed training code, strictly aligned with `openpi-comet-pytorch-mixed/scripts/train_pytorch_new.py`.
- BEHAVIOR data loading/streaming fully aligned with `_data_loader.create_behavior_data_loader_torch` (inside `build_datasets`), including the `behavior_dataset.py` alignment with `openpi-comet-pytorch-mixed` and `openpi-comet`.
- The two SFT modes: (1) direct task string ("turn on radio") → actions; (2) finer-grained skill/subtask-level training aligned with `openpi-comet` (`bash run.sh jax_skill`), centered on `enable_gap`, `allow_left`, `allow_right`; ensure the BEHAVIOR dataset supports this configuration (the skill-augmented training previously exercised on the Qwen3 4B VLM via `bash examples/sft/run_vlm_sft.sh behavior_qwen3_vlm_sft_agentic`).
- Updating `examples/sft/config/behavior_pi05_vla.yaml` and adding `pi0_5_pytorch.yaml` under `examples/sft/config/model` for SFT, aligned to the source training config/hyperparameters.
- The `full_pi05` full version (`OpenPi05FullForRLActionPrediction`, `static_kv_cache.py`): standalone VLM SFT and think-then-act autoregressive generation — explicitly later.

Environment note (reference): the `openpi-comet` repositories use `/mnt/public/xzxuan/repos/openpi-comet/.venv`; `RLinf_pi05` uses `/mnt/public/xzxuan/.venv_pi`.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers. These terms are for plan documentation only, not for the resulting codebase. Use descriptive, domain-appropriate naming in code instead.
- Follow the repo conventions (Google Python style, Ruff, type hints/docstrings on public APIs; use logging, not `print`).
- Do not modify the old `rlinf/models/embodiment/openpi/` package or the runner/eval-loop logic, beyond adding the new model type to the existing OPENPI dispatch branch in the rollout worker.
- Keep vendored core file names close to the source package to ease future syncs; namespace-convert imports only.

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
--- Original Design Draft End ---
