# Phase 3 — OpenPI PyTorch Refactor & SFT Alignment

## Goal Description

Refactor the self-contained PyTorch OpenPI 0.5 package (`rlinf/models/embodiment/openpi_pytorch/`) into a clean, layered layout and make its BEHAVIOR SFT training strictly aligned with the external reference run, while preserving every verified behavior from the prior migration phases.

Concretely:

1. **Reorganize the package** so that only `openpi_action_model.py` remains at the top level. Rename the current `utils/` (the Pi0 model internals) to `pi0_model/`; create a NEW `utils/` holding the tooling scripts (the three directional converters `jax_to_new_pytorch.py`/`old_to_new.py`/`new_to_old.py`, the `export_sft_checkpoint.py` export, and `image_tools.py`); move `normalize.py`, `processing.py`, `tokenizer.py` into `pi0_model/`; relocate the bundled tokenizer `.model` out of the repo. Remove dead code and reduce assertions to those that guard genuine external/runtime boundaries.
2. **Mirror the functionality** of the in-repo `openpi/` package (`__init__.py`, `dataconfig/`, `policies/`) for the BEHAVIOR path, but drive ALL configuration from YAML — no hard-coded `TrainConfig`-style registry — so the YAML no longer needs `openpi.config_name`.
3. **Relocate the BEHAVIOR data-loading code** out of the model package into `rlinf/data/datasets/behavior/`, and move the dataloader-builder out of the SFT worker, dispatched the same way DreamZero is.
4. **Consolidate the three checkpoint-conversion concerns** (JAX→PyTorch-new, PyTorch-old→PyTorch-new, PyTorch-new→PyTorch-old) into documented CLI scripts with a uniform four-parameter interface, verified by value-level comparison against the three reference base models.
5. **Unify norm-stats** on the BEHAVIOR task-0000 distribution for BOTH eval and SFT, sourced via YAML asset configuration (not hard-coded paths).
6. **Refactor the YAML configs** so model-shape fields live in the model template and all filesystem paths live in the experiment configs, removing hard-coded paths and the unused `config.json` load from package code.
7. **Align SFT training** (`use_skill: false` first) with the reference loss curve across model, norm stats, hyperparameters, training method, dataloader, and distributed setup; and make the `use_skill: true` skill-text path functional (dataloader loads) without requiring a full loss-matching run.

The whole effort is **additive and self-contained**: the package must remain free of any dependency on the in-repo `openpi` package, the relocated data modules must likewise avoid that dependency, the in-repo `openpi/` package must not change in behavior, and the existing CPU test suite must remain green at every milestone (with import paths updated).

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. "The suite" refers to the existing `tests/unit_tests/test_openpi_pytorch_*` CPU test set (≈55 tests today), kept green (with updated import paths) throughout. CPU-verifiable gates are hard-blocking; gates that require GPU/FSDP/bf16 are evidenced per DEC-5.

- AC-1: **Package layout end-state.** After the refactor, `openpi_pytorch/` contains only `openpi_action_model.py` as a top-level Python module (plus `__init__.py` and the subpackages `pi0_model/`, `utils/`, `dataconfig/`, `policies/`). The former `utils/` model internals live under `pi0_model/`; `normalize.py`, `processing.py`, `tokenizer.py` live under `pi0_model/`; the new `utils/` holds the three directionally-named converters `jax_to_new_pytorch.py`, `old_to_new.py`, `new_to_old.py` (per DEC-4), the SFT-FSDP→eval export module `export_sft_checkpoint.py`, and `image_tools.py`. The bundled `assets/paligemma_tokenizer.model` is relocated out of the repository (per DEC-3), so `assets/` is removed once empty.
  - Positive Tests (expected to PASS):
    - A layout test asserts the set of top-level `*.py` files equals exactly `{openpi_action_model.py, __init__.py}`.
    - Import smoke test imports `pi0_model.pi0`, `pi0_model.tokenizer`, `pi0_model.normalize`, `pi0_model.processing`, `utils.jax_to_new_pytorch`, `utils.old_to_new`, `utils.new_to_old`, `utils.export_sft_checkpoint`, `utils.image_tools` from the new paths and succeeds.
  - Negative Tests (expected to FAIL):
    - Importing any moved module from its OLD path (e.g. `openpi_pytorch.tokenizer`, `openpi_pytorch.utils.pi0`) raises `ModuleNotFoundError`.
    - A repo-wide grep/AST scan test finds ZERO references to old import paths in `rlinf/`, `tests/`, `examples/`; any leftover old path fails the test.

- AC-2: **Dead code removed and assertions minimized (concrete inventory).** The following are deleted: `pi0_model/lora.py` (imported nowhere). The following assertions are removed as redundant internal invariants: the config-consistency `assert`s in `pi0_model/gemma.py` (the ~5 `assert all(c.X == configs[0].X ...)` checks) and any dtype/None internal-invariant asserts in `pi0_model/{gemma,siglip,utils}.py` that re-check values already guaranteed at construction. The following assertions are RETAINED because they guard external/runtime contracts: config validation in `__init__.py`/`get_model`, and the boundary shape/type guard in `utils/image_tools.py`. (The implementer may extend this inventory during task10's audit, but every removal must be justified as a redundant internal invariant, and every retained guard must keep a test that proves it still rejects bad input.)
  - Positive Tests:
    - Importing the deleted dead module raises `ModuleNotFoundError`.
    - The full suite passes after assertion reduction (behavior unchanged).
    - A test confirms each RETAINED boundary guard still raises on its bad-input case (e.g. an out-of-contract shape into `image_tools`, an invalid config into `get_model`).
  - Negative Tests:
    - Removing a boundary-guarding validation that changes observable behavior is caught by a validation/regression test turning red.
    - Deleting a module still imported anywhere makes an import/smoke test fail.

- AC-3: **Self-containment invariant preserved.** The "no real `import openpi` / `from openpi import`" AST guarantee still holds over the FINAL package layout AND over the relocated BEHAVIOR data modules under `rlinf/data/datasets/behavior/`. The standalone `_import_isolation.py` module may be removed only if its AST-checking logic is relocated (into the isolation test or a test helper) so the guarantee is still enforced.
  - Positive Tests:
    - The isolation test scans the post-refactor `openpi_pytorch/` tree and `rlinf/data/datasets/behavior/` and reports zero external-`openpi` imports.
  - Negative Tests:
    - Injecting `from openpi import x` (or `import openpi`) into any scanned module makes the isolation test fail.
    - A comment/string containing the token `openpi` does NOT trigger a failure (AST-based, not text-based).

- AC-4: **Mirror `openpi/` BEHAVIOR functionality without a hard-coded TrainConfig.** `openpi_pytorch/__init__.py`, `dataconfig/` (relocated per AC-6), and `policies/behavior_policy.py` provide BEHAVIOR functionality equivalent to their `openpi/` counterparts (the model factory entry, the BEHAVIOR data-config/dataset/loader surface, and the BEHAVIOR input/output policy transforms), but every configuration value is supplied from YAML; there is no hard-coded `TrainConfig`-style registry inside the package.
  - Positive Tests:
    - The model and the BEHAVIOR dataloader can be constructed from a YAML config alone (no `config_name` lookup into an in-code registry).
    - A fixed-sample parity test: a single raw BEHAVIOR frame through the mirrored policy/dataconfig path yields the same state/action/image/prompt tensors (within the established tolerance) as before the refactor — i.e. the relocation/mirroring preserves the verified per-sample behavior.
    - A grep/AST test confirms the package defines no `TrainConfig`-style registry dict keyed by config name.
  - Negative Tests:
    - Constructing the model fails clearly if a required YAML field is absent (rather than silently falling back to an in-code hard-coded default registry).
    - A change to the mirrored policy/dataconfig that alters the per-sample output beyond tolerance fails the fixed-sample parity test.

- AC-5: **Remove `config_name` and the `config.json` reliance.** `get_model` builds the Pi0 config from YAML fields (`action_horizon`/`action_dim`/`paligemma_variant`/`action_expert_variant`); the `config.json`-loading block and the hard-coded `norm_stats_path = model_path / "physical-intelligence" / "behavior" / "norm_stats.json"` are removed; config validation no longer requires `openpi.config_name` and instead confirms the BEHAVIOR env via env-type/explicit fields. Per DEC-2, `openpi.config_name` is removed ENTIRELY from the openpi_pytorch path (no compatibility alias); the two test configs are updated to omit it.
  - Positive Tests:
    - `get_model` builds a working model from a YAML config that contains NO `openpi.config_name`.
    - Validation accepts a BEHAVIOR openpi_pytorch eval/SFT config that omits `config_name`.
  - Negative Tests:
    - A YAML missing a required model-shape field (e.g. `action_horizon`) raises a loud validation/build error.
    - A `config.json` present in the checkpoint directory does not change the built model config (it is ignored).

- AC-6: **Relocate BEHAVIOR data code and worker dispatch.** `behavior_sft_dataset.py`, `behavior_sft_data_loader.py`, and `behavior_sft_transform.py` move to `rlinf/data/datasets/behavior/`; a builder `build_behavior_sft_dataloader(cfg, world_size, rank, data_paths, eval_dataset)` lives there; the SFT worker dispatches to it the same way it dispatches DreamZero, and no longer defines `_build_openpi_pytorch_dataloader` inline. The rank-aware streaming chunk partition behavior is preserved exactly. (The AC gates BEHAVIOR, not exact branch syntax.)
  - Positive Tests:
    - The sharding, skill-window, and dataloader tests pass against the new `rlinf.data.datasets.behavior` import paths.
    - A worker-dispatch test confirms that, for an OPENPI_PYTORCH config, the worker invokes `build_behavior_sft_dataloader` with the expected arguments and returns a working loader.
  - Negative Tests:
    - The SFT worker still defining the inline helper, or importing from the old `openpi_pytorch.dataconfig` data path, fails an import/grep test.
    - A rank-aware partition test fails if relocation reverts to rank-independent (duplicated) sharding.

- AC-7: **Three documented checkpoint converters with a uniform interface, verified at value level.** Under `openpi_pytorch/utils/`, three CLI-invocable converters implement, one concern per file, per the DEC-4 binding: `jax_to_new_pytorch.py` (JAX→PyTorch-new), `old_to_new.py` (PyTorch-old→PyTorch-new), and `new_to_old.py` (PyTorch-new→PyTorch-old). Each has top-of-file usage documentation and a four-parameter interface: input model path, input norm-stats path, output model path, output norm-stats path (the two norm-stats paths are copied across verbatim). The shared key-remap helpers live with the converters. The existing SFT-FSDP→eval export capability is preserved as a separate documented module `utils/export_sft_checkpoint.py` (per DEC-4).
  - Positive Tests:
    - Each converter, run on the corresponding reference base model, produces an output whose state-dict (a) has the expected target-format key set, (b) has matching per-key shapes, (c) has the expected dtype (e.g. bf16 where required), AND (d) matches representative-tensor values: for a FIXED set of representative keys and numeric tolerances pinned by task10 BEFORE implementation (not chosen after seeing results), the converted tensor value-hashes/statistics equal the reference target model's within tolerance (catching wrong transpose/split/remap, not just key/shape).
    - A round-trip (old→new→old or new→old→new) over the reference model preserves per-key values within tolerance.
    - The norm-stats file is copied unchanged (byte/hash match).
  - Negative Tests:
    - Invoking a converter with a missing required parameter exits with a clear usage error.
    - A converter with an injected wrong transpose/split produces values outside tolerance and fails the value-parity test (i.e. the test is sensitive to value errors, not just structural ones).

- AC-8: **Norm-stats unified on task-0000 for eval and SFT.** Both the eval path and the SFT path resolve the SAME BEHAVIOR task-0000 `norm_stats.json` (the `pi05_b1k-task0000_sft_pytorch_mixed` distribution), via YAML `assets_dir` + `asset_id` rather than a hard-coded path. The canonical artifact is made available to the new-format base checkpoint per the draft.
  - Positive Tests:
    - A test asserts the norm-stats resolved by the SFT dataloader path and by the eval/model path hash-match the canonical task-0000 file.
  - Negative Tests:
    - Pointing one path at a different norm-stats file makes the hash-equality test fail.
    - A missing norm-stats artifact raises a clear resolution error (not a silent fallback to wrong stats).

- AC-9: **YAML config refactor (paths out of the model template / package code).** `examples/sft/config/model/pi0_5_pytorch.yaml` and `examples/embodiment/config/model/pi0_5_pytorch.yaml` carry the model-shape fields (`action_horizon`/`action_dim`/`paligemma_variant`/`action_expert_variant`) and NOT filesystem paths; `model_path`/`assets_dir`/`asset_id` live in the experiment configs `examples/sft/config/behavior_pi05_vla.yaml` and `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml`; the embodiment model template carries `assets_dir`/`asset_id` placeholders; no path is hard-coded in `openpi_pytorch/__init__.py`.
  - Positive Tests:
    - A config-composition snapshot test composes both eval and SFT configs and asserts the resolved model+data settings (shape fields, resolved paths) match an expected snapshot, catching template/path drift.
    - A grep/AST test confirms `openpi_pytorch/__init__.py` contains no hard-coded `norm_stats.json` path and no `config.json` read.
  - Negative Tests:
    - A leftover hard-coded path in package code fails the grep/AST test.
    - The model template still carrying `model_path` fails the config-shape snapshot test.

- AC-10: **SFT data config additions and `use_skill` plumbing.** The `data` block of `examples/sft/config/behavior_pi05_vla.yaml` includes `tasks: ["turning_on_radio"]` and `use_skill: false`. The `use_skill` flag is plumbed end-to-end: `false` selects the main-task text as the training prompt; `true` selects the skill text, honoring `enable_gap=True`, `allow_left=100`, `allow_right=100` per the reference skill TrainConfig.
  - Positive Tests:
    - With `use_skill: false`, a built batch's prompt derives from the main-task text.
    - With `use_skill: true`, a built batch's prompt derives from the per-frame skill label, and the existing skill-window test (gap absorption, allow_left/allow_right extension, random overlap assignment) still passes.
  - Negative Tests:
    - A config that ignores `use_skill` (always task text, or always skill text) fails the corresponding prompt-source assertion.

- AC-11: **`use_skill: false` SFT alignment with the reference.** Under the matched recipe (global batch 256 = 32×8, AdamW β=(0.9,0.95)/eps=1e-8/wd=1e-10/clip=1.0, `openpi_cosine` peak=2.5e-5 / warmup=1000 / decay=30000 / min_lr=0, seed=42, fp32 weights + bf16 compute under FSDP, no EMA, prefetch disabled, episodes range(200), task-0000 norm stats, `turning_on_radio`), the repo's SFT loss matches the reference per the operational protocol in DEC-1: (a) HARD — fixed-batch forward-loss parity at identical weights/seed/data within tolerance; (b) HARD — the first-N logged training-loss values fall within the agreed tolerance band of the reference log (which starts ≈0.246 at step 0 with grad-norm ≈2.0, log_interval=1, rank-0 per-step loss); (c) ADVISORY — a longer (~1-hour) run's loss trend tracks the reference curve. CPU-verifiable parts are CI-hard; GPU/FSDP parts are evidenced per DEC-5.
  - Positive Tests:
    - A fixed-batch loss-parity check (identical weights, seed, and input batch) matches the reference loss within the DEC-1 tolerance.
    - The first-N step training losses lie within the agreed band of the reference log under the matched hyperparameters.
  - Negative Tests:
    - Introducing a hyperparameter mismatch (e.g. warmup starting at 0 instead of peak/(warmup+1), wrong global batch, EMA enabled, wrong norm stats) pushes the early-step loss outside the band.

- AC-12: **`use_skill: true` is functional.** The BEHAVIOR dataloader, with `use_skill: true` and `enable_gap/allow_left/allow_right` set, loads and yields valid `(Observation, actions)` batches whose prompts are skill labels selected by the reference window logic. A full loss-matching training run is NOT required.
  - Positive Tests:
    - Iterating the `use_skill: true` loader yields a well-formed batch (correct shapes, skill-text prompt) without error.
  - Negative Tests:
    - A crash, a wrong-shaped batch, or a task-text (non-skill) prompt under `use_skill: true` fails the loader test.

- AC-13: **Suite green at every milestone; imports migrated; old `openpi` unaffected.** Every existing `test_openpi_pytorch_*` test is updated to the new import paths and passes after EACH milestone (M1…M5, including M4); per DEC-3 the bundled tokenizer asset is moved out of the repo and resolved via a default/override path, so the byte-exact tokenizer parity test is skip-gated when the external asset is unavailable and runs (and must pass) when it is present. The in-repo `openpi/` (old) package is unaffected, proven concretely: its own tests still pass, and a grep/AST check confirms the Phase-3 changes touch no file under `rlinf/models/embodiment/openpi/` (only the `openpi_pytorch` package, the relocated `rlinf/data/datasets/behavior/`, the worker dispatch, `rlinf/config.py`, configs, and tests).
  - Positive Tests:
    - The full CPU suite passes after each milestone, including M4.
    - The tokenizer byte-exact parity test passes when the external asset is present and is skip-gated (not failed) when it is absent.
    - The old `openpi` package's tests still pass and the no-touch grep/AST check is clean.
  - Negative Tests:
    - A milestone that leaves any test red is not "done".
    - A change under `rlinf/models/embodiment/openpi/` (old package) or a broken old-package test fails the no-touch check.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The full reorganization is complete and the package exposes only `openpi_action_model.py` at the top level with `pi0_model/`, `utils/`, `dataconfig/`, `policies/` subpackages; dead code is deleted and assertions are minimized per the AC-2 inventory; the BEHAVIOR data code is relocated and the worker dispatches DreamZero-style; all three checkpoint converters exist with documented four-parameter interfaces and pass value-level reference verification; norm stats are unified on task-0000 for eval and SFT; all paths are YAML-driven with no hard-coded paths or `config.json` reads in package code; `use_skill` is wired for both modes; the `use_skill: false` SFT loss matches the reference to the agreed hard tolerance with an advisory long-run trend check; and the full CPU suite (with migrated imports) is green at every milestone.

### Lower Bound (Minimum Acceptable Scope)
The package layout end-state is achieved (AC-1) with the suite green; `config_name`/`config.json` reliance is removed and config is YAML-driven (AC-4/AC-5); the BEHAVIOR data code is relocated with rank-aware sharding preserved (AC-6); norm stats are unified on task-0000 (AC-8); the YAML configs are refactored (AC-9); self-containment is preserved (AC-3); the three checkpoint converters exist, are documented, and pass value-level verification for the conversions whose reference artifacts are available (AC-7); the `use_skill: false` SFT loss is aligned to the agreed CPU-hard gates (fixed-batch parity + first-N band, AC-11); and `use_skill: true` is functional at the dataloader level (AC-12). The ~1-hour long-run loss-trend match is ADVISORY and is NOT part of the minimum scope (it is GPU-evidenced per DEC-5, not CI-gated).

### Allowed Choices
- Can use: the existing `openpi_pytorch` building blocks and test patterns; the `openpi-comet-pytorch-mixed` and `openpi-comet` repos and their listed reference model/checkpoint/log/norm-stats artifacts for verification; the reference Python-3.11 venv for cross-checks; `python -m` module CLIs (with top-of-file docs) for the converters; tolerance bands and fixed-seed/fixed-batch parity for loss verification; committed run-evidence artifacts for GPU/FSDP gates (per DEC-5).
- Cannot use: any real `import openpi` / `from openpi import` in the `openpi_pytorch` package or the relocated BEHAVIOR data modules; a hard-coded `TrainConfig`-style registry inside the package; hard-coded filesystem paths or `config.json` reads in package code; modifications that change the in-repo `openpi/` (old) package's behavior; exact bitwise loss-curve equality as a hard gate (run-to-run nondeterminism makes this infeasible — see DEC-1).

> **Note on Deterministic Designs**: The draft is highly prescriptive about the target file layout, the four-parameter converter interface, the norm-stats source, the reference hyperparameters, and the `use_skill` semantics; for those, the bounds converge and "Allowed Choices" is narrow. The genuinely open decisions are isolated in `## Pending User Decisions`.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach
Sequence the work so the test suite never goes red for long:
1. **Layout move first (mechanical).** Rename `utils/`→`pi0_model/`; create the new `utils/`; move `normalize/processing/tokenizer` into `pi0_model/`; move `image_tools` and the converters (`jax_to_new_pytorch`/`old_to_new`/`new_to_old` + `export_sft_checkpoint`) into the new `utils/`; delete dead `lora.py`; relocate `_import_isolation` logic into the isolation test; move the bundled tokenizer `.model` out of the repo and resolve it via a default/override path. Update every import (package code, the ≈13 test files, and `rlinf/models/__init__.py`). Run the suite green.
2. **Relocate BEHAVIOR data code.** Move the three `behavior_sft_*` modules to `rlinf/data/datasets/behavior/`, add `build_behavior_sft_dataloader(...)`, switch the worker to a DreamZero-style dispatch branch, and update test imports. Keep the rank-aware partition intact.
3. **YAML + config_name + config.json + norm-stats.** Build `Pi0Config` from YAML; delete the `config.json` block and hard-coded norm-stats path; relax validation off `config_name`; move paths into experiment configs; unify task-0000 norm stats via `assets_dir`/`asset_id`.
4. **Checkpoint converters.** Implement/curate the three converters under `utils/` with the four-parameter CLI and docs; verify value-level against the reference base models.
5. **SFT alignment (`use_skill: false`).** Walk the reference recipe knob-by-knob (norm stats, episode set, batch, optimizer, schedule, precision, FSDP, prefetch, seed, tokenizer length); add a fixed-batch loss-parity check vs the reference; compare first-N logged losses to the reference band; then a longer advisory run.
6. **`use_skill: true` functional.** Plumb the flag to select skill vs task text and exercise the loader.

A small **external-artifact manifest** (model paths, norm-stats path + expected hash, tokenizer path + expected hash, reference log path) is a useful aid for AC-7/AC-8 and the DEC-5 evidence.

### Relevant References
- `rlinf/models/embodiment/openpi_pytorch/` — package being refactored (current `utils/` holds the Pi0 model internals; `dataconfig/` holds the BEHAVIOR SFT data code; `assets/paligemma_tokenizer.model` is the bundled tokenizer).
- `rlinf/models/embodiment/openpi/` — in-repo old package whose BEHAVIOR `__init__`/`dataconfig`/`policies` functionality is mirrored (but whose hard-coded `TrainConfig` style is explicitly NOT copied) and whose behavior must NOT change.
- `rlinf/data/datasets/dreamzero/` — the dispatch/builder pattern to follow (`build_dreamzero_sft_dataloader(cfg, world_size, rank, data_paths, eval_dataset)`); `rlinf/data/datasets/behavior/` — pre-created empty relocation target.
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` — SFT worker with the current inline `_build_openpi_pytorch_dataloader` and the DreamZero dispatch branch to mirror.
- `rlinf/config.py` — `SupportedModel.OPENPI_PYTORCH`, `_validate_openpi_pytorch_eval_cfg`, `validate_sft_cfg` (the two `config_name` validation sites).
- `rlinf/hybrid_engines/fsdp/utils.py` — `get_lr_scheduler` with the reference-exact `openpi_cosine` mode.
- `examples/sft/config/{behavior_pi05_vla.yaml, model/pi0_5_pytorch.yaml, model/pi0_5.yaml}` and `examples/embodiment/config/{behavior_ppo_openpi_pi05_pytorch_eval.yaml, model/pi0_5_pytorch.yaml}` — configs to refactor.
- External: `openpi-comet-pytorch-mixed` (`run.sh`; `pi05_b1k-task0000_sft_pytorch_mixed` TrainConfig; `scripts/convert_jax_model_to_pytorch_new.py`; `src/openpi/models_pytorch_new/checkpoint_format.py`; the reference loss log; the task-0000 `norm_stats.json`) and `openpi-comet` (`pi05_b1k-task0000_sft_local_skill` TrainConfig; `run.sh jax_skill`). Reference base models: `/mnt/public/xzxuan/models/pi05_base{,_pytorch,_pytorch_new}`.

## Dependencies and Sequence

### Milestones
1. **M1 — Package reorganization (AC-1, AC-2, AC-3, AC-13).**
   - Phase A: Rename `utils/`→`pi0_model/`; create new `utils/`; move modules per the end-state; delete dead code; fix tokenizer asset path.
   - Phase B: Migrate all imports (package, tests, `rlinf/models/__init__.py`); relocate the AST isolation logic into the test; minimize assertions per the AC-2 inventory; suite green.
2. **M2 — BEHAVIOR data relocation + worker dispatch (AC-6, AC-3, AC-13).**
   - Step 1: Move the three `behavior_sft_*` modules to `rlinf/data/datasets/behavior/` and add `build_behavior_sft_dataloader`.
   - Step 2: Switch the worker to DreamZero-style dispatch; migrate test imports; preserve rank-aware sharding; suite green.
3. **M3 — YAML-driven config, `config_name`/`config.json` removal, norm-stats unification (AC-4, AC-5, AC-8, AC-9, AC-13).**
   - Step 1: Build `Pi0Config` from YAML; remove `config.json` block and hard-coded paths; relax validation.
   - Step 2: Refactor the YAML configs (paths into experiment configs); unify task-0000 norm stats; add config-composition snapshot tests; suite green.
4. **M4 — Checkpoint converter consolidation (AC-7, AC-13).**
   - Depends on M1 (final `utils/` location) and DEC-4 (file→conversion binding). Implement/curate the three converters; value-level verify against reference models; suite green (M4 carries the AC-13 gate too).
5. **M5 — SFT alignment (AC-10, AC-11, AC-12).**
   - Depends on M2 + M3 (relocated data + unified norm stats + YAML recipe). Plumb `use_skill`; build the fixed-batch parity harness; run the first-N comparison; the advisory long run; then the `use_skill: true` functional loader.

Dependency summary: M1 precedes everything (it fixes the layout and imports). M2 and M3 are largely independent of each other but both follow M1. M4 follows M1 and DEC-4. M5 follows M2 and M3. AC-13 (suite-green + import migration + old-package no-touch) is a cross-cutting gate satisfied at the end of each milestone.

## Task Breakdown

Each task includes exactly one routing tag: `coding` (Claude) or `analyze` (Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Rename `utils/`→`pi0_model/`, create new `utils/`, move `normalize/processing/tokenizer` into `pi0_model/` and `image_tools` + the converters into `utils/`; relocate the bundled `paligemma_tokenizer.model` out of the repo (DEC-3) and point the tokenizer at a default/override external path; remove `assets/` once empty | AC-1 | coding | - |
| task2 | Delete dead `lora.py`; minimize redundant asserts per the AC-2 inventory while keeping (and testing) boundary guards | AC-2 | coding | task1 |
| task3 | Relocate the AST import-isolation logic into the isolation test (remove standalone `_import_isolation.py`); extend its scan to cover `rlinf/data/datasets/behavior/` | AC-3 | coding | task1, task5 |
| task4 | Migrate all import paths (package modules, ≈13 test files, `rlinf/models/__init__.py`); add the old-import-path grep/AST scan test; run the full CPU suite green | AC-1, AC-13 | coding | task1 |
| task5 | Move `behavior_sft_dataset/transform/data_loader` to `rlinf/data/datasets/behavior/`; add `build_behavior_sft_dataloader(cfg, world_size, rank, data_paths, eval_dataset)`; preserve rank-aware streaming partition | AC-6 | coding | task1 |
| task6 | Switch the SFT worker to dispatch the BEHAVIOR loader the DreamZero way; remove the inline `_build_openpi_pytorch_dataloader`; add a dispatch-behavior test | AC-6 | coding | task5 |
| task7 | Build `Pi0Config` from YAML shape fields; delete the `config.json` block and the hard-coded `norm_stats_path`; remove `config_name` entirely (DEC-2) and switch validation to env/explicit-field checks; update the two test configs to omit `config_name` | AC-4, AC-5 | coding | task4 |
| task8 | Refactor YAML configs: move `model_path`/`assets_dir`/`asset_id` into experiment configs; placeholders in the embodiment model template; shape fields in both `pi0_5_pytorch.yaml`; add `data.tasks`/`data.use_skill`; add config-composition snapshot tests | AC-9, AC-10 | coding | task7 |
| task9 | Unify task-0000 norm stats for eval+SFT via `assets_dir`/`asset_id`; make the canonical artifact available to the new-format base; add a hash-equality test | AC-8 | coding | task8 |
| task10 | Audit the repo's assertions (for the AC-2 inventory) AND the three checkpoint-conversion concerns vs the reference scripts and the three reference base models; produce the exact file→conversion binding (resolving DEC-4 detail) and the value-level verification spec, naming the FIXED representative tensor keys and numeric tolerances up front | AC-2, AC-7 | analyze | task1 |
| task11 | Implement/curate the three documented converters `utils/jax_to_new_pytorch.py`, `utils/old_to_new.py`, `utils/new_to_old.py` (DEC-4) with the four-parameter interface; value-level verify key-set/shape/dtype/representative-values against reference models; preserve the SFT-FSDP→eval export as `utils/export_sft_checkpoint.py` | AC-7 | coding | task10 |
| task12 | Audit the repo SFT path against the reference recipe knob-by-knob (norm stats, episode set, batch, optimizer, schedule, precision, FSDP, prefetch, seed, tokenizer length) and list every divergence | AC-11 | analyze | task9 |
| task13 | Plumb `use_skill` to select task vs skill text; wire `enable_gap/allow_left/allow_right` | AC-10 | coding | task12 |
| task14 | Build the fixed-batch loss-parity harness (identical weights/seed/batch) and assert parity vs the reference within the DEC-1 tolerance | AC-11 | coding | task13 |
| task15 | Run the first-N-step `use_skill: false` training and assert the logged losses fall within the DEC-1 band of the reference log; capture run-evidence per DEC-5 | AC-11 | coding | task14 |
| task16 | Run the advisory ~1-hour long run and record the loss-trend comparison as evidence (ADVISORY; not a CI gate) | AC-11 | coding | task15 |
| task17 | Make `use_skill: true` functional: loader yields skill-text batches via the window logic; add/keep the loader + skill-window tests | AC-12 | coding | task13 |
| task18 | Final pass: full CPU suite green with migrated imports; old `openpi` no-touch check; tokenizer byte-exact parity preserved | AC-13 | coding | task16, task17 |

## Claude-Codex Deliberation

### Agreements
- Couple-then-isolate risk is real: the package move and the SFT alignment must be sequenced and individually verified so failures are attributable. Keep the suite green per milestone (M1…M5 including M4).
- Deleting the standalone import-isolation module is acceptable ONLY if the AST guarantee is relocated; the invariant must survive and extend to the relocated data modules.
- "Loss curves align exactly" cannot be a bitwise hard gate; alignment is fixed-batch/seed parity plus a tolerance band over an agreed step window, with the long-run trend advisory and GPU-evidenced.
- Removing `config_name` is low-risk in practice (validation-only today; data pipeline already `config_name`-agnostic) but touches validation, the model factory, the two test configs, and YAML conventions, so it is its own milestone.
- Norm-stats unification on task-0000 for both eval and SFT may shift eval numerics relative to the prior eval baseline; this is accepted as a consequence of the draft's explicit instruction and is re-baselined.
- AC verifiability must be concrete: AC-2 carries an explicit assertion inventory; AC-4 adds a fixed-sample parity gate; AC-7 adds value-level (not just structural) checks; AC-9 adds config-composition snapshots; AC-13 adds an explicit old-package no-touch proof.

### Resolved Disagreements
- **Checkpoint script location (draft section 1 vs section 4):** The stated end-state ("only `openpi_action_model.py` at the top level") is authoritative — all three converters live under the new `utils/`; discoverability is preserved via top-of-file docs and `python -m ...utils.<name>` CLIs. (The file→conversion binding and SFT-export home are resolved by DEC-4: `jax_to_new_pytorch.py`/`old_to_new.py`/`new_to_old.py` plus a separate `export_sft_checkpoint.py`.)
- **Tokenizer asset (DEC-3, user decision):** The user chose to honor the draft and move the bundled tokenizer `.model` out of the repository, accepting that CPU byte-exact tokenizer parity becomes skip-gated when the external asset is absent (resolved via a default/override path). This overrides the bundled-keep recommendation in favor of the draft's explicit "no model files in the repo" instruction.
- **`checkpoint_format.py` double-listing:** It is tooling, so it lives under `utils/` with the other converters, not under `pi0_model/`.
- **"Mirror `pi0_5.yaml`" vs "no `model_path`":** `pi0_5.yaml` carries a placeholder `model_path`, yet the draft says the pytorch model template should not. Resolution by intent: model templates carry model-shape fields only; all filesystem paths (including `model_path`) live in the experiment configs.
- **AC-2/AC-4/AC-7 verifiability (round-1 Codex):** strengthened from subjective/structural to concrete inventory + fixed-sample parity + value-level checks, respectively.
- **AC-6 over-specification (round-1 Codex):** the AC gates dispatch BEHAVIOR (invokes `build_behavior_sft_dataloader` with expected args), not exact branch syntax.
- **M4 missing suite gate (round-1 Codex):** AC-13 now explicitly covers M4.
- **task13 overload (round-1 Codex):** split into `use_skill` plumbing (task13), fixed-batch parity harness (task14), first-N comparison (task15), advisory long run (task16).

### Convergence Status
- Final Status: `converged` (the technical plan converged over two Codex rounds with no remaining required changes; all five user decisions below are now resolved).

## Pending User Decisions

- DEC-1: **SFT loss-alignment acceptance protocol (operational).**
  - Claude Position: Tiered. (a) HARD — fixed-batch forward-loss parity at identical weights/seed/data within **relative tolerance 5%** (or abs 0.01, whichever larger). (b) HARD — the **first N=50** logged per-step training-loss values lie within a band of the reference log (per-step **|Δ| ≤ 0.03** on the absolute loss, OR within ±2σ of the reference's own step-to-step variation). Loss metric = the **mean flow-matching MSE over the batch, as logged at rank 0, log_interval=1**, recorded **before** each optimizer step (matching the reference: step-0 LR = peak/(warmup+1) = 2.5e-8, step-0 loss ≈ 0.246); no gradient accumulation (global batch = per-step batch). Required evidence: committed log + the norm-stats/tokenizer/weight hashes. (c) ADVISORY — the ~1-hour long-run trend tracks the reference curve direction.
  - Codex Position: Concurs exact equality is infeasible; requires N, tolerance band, exact loss numerator/denominator, rank aggregation, accumulation timing, step-0 before/after logging, and artifact hashes to be fixed before execution (as above).
  - Tradeoff Summary: A hard exact-match gate is unachievable under FSDP/bf16/dataloader nondeterminism; a purely advisory gate is too weak. The tiered, fully-specified split makes the deterministic parts hard and the nondeterministic long-run part advisory.
  - Decision Status: `RESOLVED` — Tiered protocol adopted as written (fixed-batch parity + first-50-step band HARD; ~1h trend ADVISORY; rank-0 per-step pre-optimizer-step loss; rel 5%/|Δ|≤0.03).

- DEC-2: **`config_name` removal vs temporary compatibility alias.**
  - Claude Position: Remove `openpi.config_name` from the openpi_pytorch path entirely (the draft says the YAML should no longer need it; it is validation-only today and the package is internal on a feature branch with no external consumers). AC-5 tests then assert configs WITHOUT `config_name`.
  - Codex Position: Acceptable; alternatively keep `config_name` as a deprecation alias until validation/tests/configs all pass, removing it afterward — but AC-5 must match whichever is chosen.
  - Tradeoff Summary: Clean removal is simpler and matches the draft's intent but is a one-shot break of existing configs/tests; a compat alias is safer but leaves dead surface and contradicts "YAML should no longer need it".
  - Decision Status: `RESOLVED` — Remove `config_name` ENTIRELY (no alias); AC-5 tests assert configs without it.

- DEC-3: **Tokenizer asset: keep CI-deterministic parity, or move out of the repo per the draft.**
  - Claude Position (recommended): Keep `paligemma_tokenizer.model` bundled in the package for CI/offline byte-exact tokenizer parity, and ADD an optional external override path (e.g. `/mnt/public/xzxuan/models/paligemma_tokenizer/`) for deployments. This preserves the already-landed byte-exact parity contract (AC-13) deterministically in CPU CI. (Codex flags that moving it out and skip-gating CPU parity weakens that contract.)
  - Codex Position: Resolve in favor of CI-deterministic parity (bundled + override) unless the user explicitly accepts weaker skip-gated tests.
  - Tradeoff Summary: The draft explicitly says "model files should not live in the repo" (move it out), but moving it out makes byte-exact CPU parity depend on a private absolute path (skip-gated, weaker). Keeping it bundled with an override honors the parity contract while still enabling external assets.
  - Decision Status: `RESOLVED` — MOVE the tokenizer `.model` out of the repo per the draft; resolve via default/override path; CPU byte-exact parity test is skip-gated when the external asset is absent.

- DEC-4: **Checkpoint file→conversion binding and where the SFT-FSDP→eval export lives.**
  - Claude Position: Bind `utils/convert_checkpoint.py`→JAX→PyTorch-new, `utils/checkpoint_format.py`→PyTorch-old→PyTorch-new (plus the shared key-remap helpers), `utils/export_checkpoint.py`→PyTorch-new→PyTorch-old; keep the existing SFT-FSDP→eval export as an additional documented function in `export_checkpoint.py`.
  - Codex Position: Acceptable, but the SFT-FSDP→eval export is a distinct concern from the three format conversions and must be explicitly preserved; entrypoints should stay clearly user-facing/documented. DEC-4 must be resolved before AC-7 implementation (task11).
  - Tradeoff Summary: The draft says "three conversions, one per file" but does not bind file→conversion and overloads `export_checkpoint.py` (currently the SFT-export). A clear binding plus an explicitly preserved SFT-export avoids losing required functionality.
  - Decision Status: `RESOLVED` — User-specified directional names: `utils/jax_to_new_pytorch.py` (JAX→new), `utils/old_to_new.py` (old→new), `utils/new_to_old.py` (new→old); the SFT-FSDP→eval export lives in its own module `utils/export_sft_checkpoint.py`.

- DEC-5: **Evidence policy for GPU/FSDP gates not runnable in CPU CI.**
  - Claude Position: CPU-verifiable gates (the full unit suite, fixed-sample transform parity, LR schedule, rank-aware sharding logic, tokenizer parity, config snapshots, converter value-parity on CPU) are HARD and CI-blocking. Gates that require GPU/FSDP/bf16 (the fixed-batch loss-parity run, the first-N-step training comparison, the advisory long run) are verified by COMMITTED run-evidence artifacts: exact command, environment, log path, the loss series, and output/checkpoint paths + hashes — following the prior phase's precedent. The advisory long run is evidence-only, never a CI gate.
  - Codex Position: Requires an explicit decision on which checks are CPU-only vs GPU, and what artifact/log evidence is mandatory when hardware runs are not CI-gated (as above).
  - Tradeoff Summary: Without this split, GPU-only alignment claims could be asserted in prose without verifiable evidence; with it, every non-CPU gate has a concrete committed artifact.
  - Decision Status: `RESOLVED` — CPU-verifiable gates are HARD/CI-blocking; GPU/FSDP gates are verified by committed run-evidence artifacts (command, env, log path, loss series, output/checkpoint paths+hashes); the ~1h long run is evidence-only, never a CI gate.

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.
- All new behavior must be covered by tests (the CPU suite must stay green at each milestone); changes are additive and must preserve the self-containment invariant and the in-repo `openpi/` (old) package's behavior.

--- Original Design Draft Start ---

# Code Refactoring & SFT Alignment Task

This document specifies a refactoring of the OpenPI PyTorch codebase and an SFT (supervised fine-tuning) alignment effort. The current code is somewhat disorganized and has several incorrect spots. Please address the items below.

---

## Part I — Code Refactoring

### 1. Reorganize the `openpi_pytorch` module

Clean up all code under:

```
/mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi_pytorch
```

Specifically:

- **Remove dead code.** Delete anything that is unused.
- **Reduce assertions.** Keep `assert` checks to a minimum. They add safety but create clutter — assume the current configuration is guaranteed to run.
- **Rename `utils/` → `pi0_model/`.**
- **Create a new `utils/` directory** containing:
  - `rlinf/models/embodiment/openpi_pytorch/utils/checkpoint_format.py`
  - `rlinf/models/embodiment/openpi_pytorch/convert_checkpoint.py`
  - `rlinf/models/embodiment/openpi_pytorch/export_checkpoint.py`
  - `rlinf/models/embodiment/openpi_pytorch/image_tools.py`
- **Move into `pi0_model/`:**
  - `rlinf/models/embodiment/openpi_pytorch/normalize.py`
  - `rlinf/models/embodiment/openpi_pytorch/processing.py`
  - `rlinf/models/embodiment/openpi_pytorch/tokenizer.py`
- **Delete `_import_isolation.py`** — it appears to be used only for testing.
- **Move the tokenizer model out of the repo.** Move:
  ```
  /mnt/public/xzxuan/repos/RLinf_pi05/rlinf/models/embodiment/openpi_pytorch/assets/paligemma_tokenizer.model
  ```
  to:
  ```
  /mnt/public/xzxuan/models/paligemma_tokenizer/
  ```
  Model files should **not** live inside the code repository.

**End state:** the `openpi_pytorch/` directory should contain only `openpi_action_model.py` at the top level; everything else lives in subdirectories.

---

### 2. Mirror the functionality of the `openpi` module

Make the following pairs functionally equivalent (functional parity is the goal — the code does **not** need to match line for line):

- `openpi_pytorch/__init__.py` ↔ `openpi/__init__.py`
- `openpi_pytorch/dataconfig/` ↔ `openpi/dataconfig/` (including `behavior_dataconfig.py` and `__init__.py`)
- `openpi_pytorch/policies/` ↔ `openpi/policies/` (i.e. `behavior_policy.py`)

**Important exception:** Do **not** copy the `TrainConfig` style from `openpi/dataconfig/__init__.py`. It is not clean — its settings are hard-coded in Python. All configuration should come from YAML, not be written into the code.

As a result, the YAML should no longer need to specify `openpi.config_name`.

---

### 3. Relocate the Behavior data-loading code

The following files are all related to loading data from Behavior and should **not** stay under `openpi_pytorch/dataconfig/`:

- `openpi_pytorch/dataconfig/behavior_sft_data_loader.py`
- `openpi_pytorch/dataconfig/behavior_sft_dataset.py`

Move them to:

```
rlinf/data/datasets/behavior/
```

Also move the `_build_openpi_pytorch_dataloader` function out of `rlinf/workers/sft/fsdp_vla_sft_worker.py` and into `rlinf/data/datasets/behavior/`.

The worker should then load from that location, following the same pattern already used for DreamZero. For example, replace:

```python
if model_type == SupportedModel.OPENPI_PYTORCH:
    return self._build_openpi_pytorch_dataloader(
        data_paths, eval_dataset=eval_dataset
    )
```

with the DreamZero-style approach:

```python
elif SupportedModel(self.cfg.actor.model.model_type) in [
    SupportedModel.DREAMZERO
]:
    from rlinf.data.datasets.dreamzero import (
        build_dreamzero_sft_dataloader,
    )

    return build_dreamzero_sft_dataloader(
        self.cfg, self._world_size, self._rank, data_paths, eval_dataset
    )
```

This keeps `fsdp_vla_sft_worker.py` concise and readable.

---

### 4. Carefully consolidate the three checkpoint-conversion scripts

The three files to organize are:

- `rlinf/models/embodiment/openpi_pytorch/export_checkpoint.py`
- `rlinf/models/embodiment/openpi_pytorch/convert_checkpoint.py`
- `rlinf/models/embodiment/openpi_pytorch/utils/checkpoint_format.py`

They must implement **three conversions**, one per file:

1. **JAX → PyTorch (new)** — reference:
   `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts/convert_jax_model_to_pytorch_new.py`
2. **PyTorch (old) → PyTorch (new)** — reference:
   `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/models_pytorch_new/checkpoint_format.py`
3. **PyTorch (new) → PyTorch (old)** — reference: same file as above.

Requirements for each file:

- **Top-of-file documentation** explaining how to use it.
- **Four parameters each:** `input model path`, `input norm stats`, `output model path`, `output norm stats`. The two norm-stats parameters are simply copied across (a straight `cp`).

**Reference model paths** (use these to verify your conversions are correct):

| Format | Path |
| --- | --- |
| JAX | `/mnt/public/xzxuan/models/pi05_base` |
| PyTorch (old) | `/mnt/public/xzxuan/models/pi05_base_pytorch` |
| PyTorch (new) | `/mnt/public/xzxuan/models/pi05_base_pytorch_new` |

**Norm stats:** The norm stats currently in use come from:

```
/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json
```

This is the distribution for **task 0000** on the Behavior benchmark, and it must be used for **both eval and SFT training**. Copy this file into:

```
/mnt/public/xzxuan/models/pi05_base_pytorch_new/
```

---

### 5. Fix the SFT norm-stats alignment

It follows from the above that the current SFT code is **not** strictly aligned — at minimum the norm stats are wrong. Align it with the `pi05_b1k-task0000_sft_pytorch_mixed` config in:

```
/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/src/openpi/training/config.py
```

That config does **not** set an `asset_dir` explicitly; it relies on a default that points to:

```
/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json
```

The current SFT config's `asset_dir` is therefore incorrect and must be fixed.

---

### 6. Refactor the configs

Make `examples/sft/config/model/pi0_5_pytorch.yaml` mirror `examples/sft/config/model/pi0_5.yaml` — i.e. it should **not** contain `model_path`, `assets_dir`, or `asset_id`.

Instead:

- `assets_dir` and `asset_id` should be written in:
  - `examples/sft/config/behavior_pi05_vla.yaml`
  - `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml`
- `examples/embodiment/config/model/pi0_5_pytorch.yaml` should also **add** these two fields, but as placeholders.

This removes the need to hard-code paths inside `openpi_pytorch/__init__.py`, such as:

```python
norm_stats_path = (
    model_path / "physical-intelligence" / "behavior" / "norm_stats.json"
)
```

Additionally, **delete** this block:

```python
config_json = {}
if (model_path / "config.json").exists():
    config_json = json.loads((model_path / "config.json").read_text())
```

All config should be written in YAML. The `config.json` inside both checkpoints is **never** used. The fields `action_horizon`, `action_dim`, `paligemma_variant`, and `action_expert_variant` should all be written in the two `pi0_5_pytorch.yaml` files.

---

## Part II — SFT Alignment

**Current status:** eval is correct, and SFT loss does decrease — but the training is not strictly aligned.

Please do a complete pass over the SFT code in the current repo (as modified above) and compare it against running:

```
/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/run.sh
```

Verify **full consistency** across: the model, norm stats, hyperparameters, training method, dataloader, distributed training, and so on. The goal is for the two loss curves to align exactly.

**Reference loss log:**

```
/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log
```

Then launch your own training script and observe for **one hour** to confirm the loss matches.

### Training data & config

SFT training uses the `turning_on_radio` task from:

```
/mnt/public/xzxuan/data/2025-challenge-demos
```

In the `data` field of `examples/sft/config/behavior_pi05_vla.yaml`, add:

```yaml
tasks: ["turning_on_radio"]
use_skill: false
```

The `use_skill` flag controls whether the text input for training is the **main task** (`false`) or the **skill** (`true`).

When `use_skill: true`, you must also account for these three parameters:

```python
enable_gap=True,
allow_left=100,
allow_right=100,
```

For this logic, carefully reference the `pi05_b1k-task0000_sft_local_skill` TrainConfig in:

```
/mnt/public/xzxuan/repos/openpi-comet/src/openpi/training/config.py
```

Its launch command is, from within `/mnt/public/xzxuan/repos/openpi-comet`:

```bash
bash run.sh jax_skill
```

### Priorities

- **Focus first on `use_skill: false`.** Get its loss curve aligned.
- **For `use_skill: true`,** get the functionality supported first (e.g. dataloader loading), and align it where possible — but a full training run to check the loss is **not** required at this stage.
--- Original Design Draft End ---
