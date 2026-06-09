# Phase 4 — RLinf SFT + Conversion vs Reference: Eval-Gap Root-Cause & Numerical Alignment

## Goal Description

Root-cause and resolve the remaining divergence between RLinf's BEHAVIOR `use_skill:false` SFT + eval
pipeline and the reference (`openpi-comet-pytorch-mixed`), so that RLinf's SFT-trained-then-converted model
is numerically equivalent to a reference-trained model under identical inputs and matched eval settings.

The investigation is **scoped to the genuinely-new surfaces** and must NOT reopen the SFT training-step
alignment that a prior effort already proved: on identical inputs RLinf's 8-GPU FSDP training matches the
reference model per-step (a pinned first-50 run scored 50/50 within `|Δ|≤0.03`, max `0.0024`; a fixed-batch
loss-parity test matches within tolerance; an unpinned ~1-hour run tracks the reference long-run trend at
Pearson `r=0.975`). Forward / gradient / optimizer / LR-schedule / FSDP are therefore aligned for identical
inputs; per-step differences between two *production* runs are benign independent flow-matching noise/time
RNG + independent data shuffle. That training path is re-confirmed here only via a pinned harness, and a
broader training investigation opens **only if** a pinned-input parity test fails under the corrected config.

The new surfaces, in priority order, are:

1. **Norm-stats alignment.** The OLD and NEW `openpi.assets_dir` values resolve to **different**
   `norm_stats.json` files (`d66ed168…` / 6368 B vs `ff7e1ff0…` / 6346 B). **The reference's canonical source
   is confirmed to be the NEW file**, established by reading the reference `TrainConfig`
   (`pi05_b1k-task0000_sft_pytorch_mixed`): it sets no explicit `assets_dir`, so it uses the default
   `assets_base_dir = "./outputs/assets/train"`; its `assets_dirs` property resolves
   `{assets_base_dir}/{config.name}`, and `_load_norm_stats` loads `{assets_dirs}/{asset_id}/norm_stats.json`
   with `asset_id = repo_id = behavior-1k/2025-challenge-demos` — i.e.
   `outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json`
   = the NEW file (`ff7e1ff0…`). So the user's `assets_dir` switch aligned RLinf to the reference; **all prior
   Phase-3 work used the OLD file (`d66ed168…`) — a real production norm-stats misalignment that the pinned
   proofs (pre-normalized dumped batches) could not see.** The remaining work is to verify RLinf's train AND
   eval paths now resolve this exact file (byte/sha256-identity) and to quantify the OLD-vs-NEW stat diff that
   prior runs were exposed to.
2. **Converter correctness.** A new, never-value-verified converter (`sft_to_new_pytorch.py`) turns the
   trained SFT FSDP checkpoint into the eval ("new pytorch") format (strip wrapper/FSDP prefixes → bare Pi0
   keys, cast float weights to bf16, write `model.safetensors` + `config.json`, copy norm-stats). It must be
   proven value-lossless *independent of training* before any success-rate reasoning.
3. **Eval-gap statistical significance.** The reported gap (RLinf-trained `0.2578` vs reference-trained
   `0.3203` success_once over 128 episodes) comes from a **stochastic, unpaired** eval (flow-matching noise
   sampled with no fixed RNG; the two runs are independent). `8/128` is ≈ 1.5 standard errors, so the gap may
   be within run-to-run noise and must be established as real (paired noise seeds and/or more episodes/seeds
   with confidence intervals) before it is treated as a defect. A control eval already shows the
   reference-trained checkpoint run through RLinf's *own* eval pipeline reproduces `0.3203` — so the eval
   pipeline itself is not implicated; suspicion is isolated to the RLinf-trained weights and/or the converter.
4. **Eval `num_steps` resolution (verified — already the tuned 5).** The eval model's flow-matching denoising
   step count is read by the model factory's `_select("num_steps")` from `actor.model.num_steps`, which
   resolves to the tuned `5` (the `model/pi0_5_pytorch.yaml` template value, also propagated to
   `actor.model.openpi.num_steps: ${actor.model.num_steps}`). The actual eval run's resolved config confirms
   `actor.model.num_steps = 5` and `actor.model.openpi.num_steps = 5`; the `num_steps: 10` present in the eval
   YAML is a STRAY, unused `actor.num_steps` key at the wrong nesting level (the model factory never reads it).
   So eval already runs at the tuned `num_steps = 5` — **no correction is required**; the only residual is a
   minor config-cleanliness item (drop the unused `actor.num_steps: 10`). `num_steps` is recorded with every
   eval to keep the operating point explicit.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests. "Within tolerance" for bf16
forward comparisons means the agreed numeric band recorded with the test (not bitwise equality). Every eval
record must carry: `num_steps`, model dtype, the resolved `model_path` / `assets_dir` / `asset_id`, the
norm-stats sha256, the action-denormalization path, seeds, episode/task set, and the config/source revision.

- AC-1: **Norm-stats alignment across all four touchpoints.** The `norm_stats.json` actually used by (a) RLinf
  SFT *training* data normalization, (b) the converter's copied output, (c) the RLinf *eval* model-load, and
  (d) the reference model's train/eval path are the SAME canonical file, established empirically (the
  reference loader's resolved path is read, not assumed), with byte/sha256-identity and shape/stat identity.
  - Positive Tests (PASS):
    - The reference loader's resolved norm-stats path (read from the reference run/config) and RLinf's
      resolved norm-stats path (read from the RLinf train + eval config dumps/logs) hash-match the same file.
    - The OLD-vs-NEW `assets_dir` discrepancy is resolved into a single documented canonical source, and a
      shape/quantile-stat diff of OLD vs NEW is recorded (quantifying what the switch changed).
  - Negative Tests (FAIL):
    - A train or eval path resolving a norm-stats file whose sha256 differs from the canonical one fails.
    - Asserting alignment from the configured `assets_dir` *string* alone (without reading the resolved file)
      is rejected — the gate is file-content identity, not config text.

- AC-2: **Converter value-parity, proven independent of training.** `sft_to_new_pytorch.py` is value-lossless
  up to the documented bf16 cast: the converted checkpoint reproduces the pre-conversion wrapper model's
  behavior on fixed inputs, and its state dict structurally matches the reference new-format checkpoint.
  - Positive Tests (PASS):
    - On a fixed observation + fixed *injected* noise/time, the converted model and the pre-conversion
      `OpenPiPytorchActionModel` wrapper (same checkpoint) produce the same per-timestep flow-matching loss
      and the same sampled action chunk (normalized model output AND post-denormalization actions) within the
      recorded bf16 tolerance.
    - The converted state dict's key-set, per-key shape, and per-key dtype equal the reference new-format
      checkpoint's (`pi05_base_pytorch_new` namespace) — no missing buffers/keys, no stray wrapper prefixes,
      every float tensor bf16.
  - Negative Tests (FAIL):
    - A converter variant that drops a key/buffer, leaves a wrapper prefix, mis-transposes, or casts to a
      wrong dtype produces actions/loss outside tolerance or a key-set mismatch and fails.
    - Relying on end-to-end success rate to judge converter correctness (instead of fixed-input parity) is
      rejected as a proxy.

- AC-3: **Eval-gap significance under a paired / confidence-interval protocol.** The `0.2578`-vs-`0.3203` gap
  is assessed with a reproducible protocol that controls eval stochasticity, and a material threshold (DEC-1)
  is applied; acceptance requires the gap be below the threshold OR statistically explained.
  - Positive Tests (PASS):
    - Both converted models are evaluated under a paired protocol (same fixed per-episode initial states AND
      the same injected flow-matching noise sequence per episode) and/or over enough episodes/seeds to report
      a confidence interval; the recorded gap and its CI are committed.
    - The control direction holds and is recorded: the reference-trained checkpoint through RLinf's eval path
      reproduces its expected success within the eval variance.
  - Negative Tests (FAIL):
    - Treating a single unpaired 128-episode aggregate difference as a defect (without pairing or CIs) fails
      the protocol.
    - Comparing the two models at *different* `num_steps`, dtype, norm-stats, or episode sets fails (the eval
      must be matched on every recorded knob).

- AC-4 (CORE): **First-step training-loss divergence root-caused on identical inputs, including the
  norm-stats application.** The step-1 loss difference — RLinf `0.349` (its run, which used the NEW norm-stats)
  vs reference `0.246` / `0.235` (its steps 0 / 1) — is reproduced and root-caused via a SAME-BATCH comparison
  that drives each repo's FULL normalization + forward path (raw actions → its own quantile-normalized actions
  → loss), NOT the pre-normalized dumped batches the prior pinned proofs used. This deliberately exercises the
  one surface those proofs bypassed — the norm-stats *application* — and is treated as a first-class
  root-cause objective, not merely diagnostic: a confirmed same-input divergence is a real misalignment to be
  fixed; a confirmed same-input match means the unpinned `0.349`-vs-`0.246` log gap is the different first
  batch / shuffle / RNG (benign), and that explanation must itself be evidenced.
  - Positive Tests (PASS):
    - Feed BOTH repos the IDENTICAL raw first batch + identical noise/time, each applying ITS OWN norm-stats
      normalization through its real loader/forward path, and compare the step-1 loss: it matches within the
      recorded tolerance (→ the log gap is benign batch/RNG variation) OR it diverges (→ a real
      norm-stats-application or recipe misalignment, which is then localized and fixed).
    - A norm-stats matrix (OLD vs NEW vs the reference-resolved `norm_stats.json`, on the same raw batch +
      noise/time) attributes any step-1 delta to the specific norm-stats file and/or its application, with
      measured numbers recorded.
  - Negative Tests (FAIL):
    - Treating the LOGGED `0.349`-vs-`0.246` values (computed on DIFFERENT first batches) as proof of a
      training bug — without the same-batch + same-normalization control — fails; the delta must be reproduced
      on identical raw inputs through each repo's full normalization + forward path before any conclusion.
    - A same-batch pipeline that injects PRE-normalized actions (bypassing the raw→normalized step) fails to
      satisfy this AC, because it cannot exercise the norm-stats-application surface under test.

- AC-5: **Training-step parity re-confirmed under the corrected config.** With the canonical norm-stats and
  matched recipe, the pinned-input training-step parity (the prior 50/50 first-50 result) still holds.
  - Positive Tests (PASS):
    - A pinned-input first-N run through the real FSDP stack under the corrected norm-stats/config reproduces
      the reference model per-step within the prior gate (`|Δ|≤0.03`).
  - Negative Tests (FAIL):
    - A pinned-input parity failure under the corrected config re-opens (and only then) a scoped training
      investigation; a pass means no broad FSDP/optimizer rework is warranted.

- AC-6: **Model divergence quantified by behavior, not raw weights.** The RLinf-trained vs reference-trained
  models are compared by their action-chunk outputs on fixed eval observations + fixed injected noise, not by
  raw parameter-tensor deltas (which are uninterpretable across independently-shuffled production runs).
  - Positive Tests (PASS):
    - On a fixed set of real eval observations with fixed injected noise/time, the per-chunk action deltas
      (post-denormalization) between the two trained-and-converted models are measured and reported, with an
      interpretation tied to AC-3's significance result.
  - Negative Tests (FAIL):
    - Reporting raw per-parameter tensor deltas between the two independently-trained models as a "divergence
      bug" indicator fails — that quantity is not interpretable given known RNG/shuffle divergence.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
Every new surface is closed with committed, reproducible evidence: the canonical norm-stats source is
established and RLinf train+eval are aligned to it (AC-1); the converter passes a fixed-input value-parity
test and a structural state-dict check (AC-2); the eval gap is measured under a paired/CI protocol at the
tuned `num_steps` (verified `5`) and judged against the DEC-1 threshold (AC-3); the step-1 loss divergence is
root-caused on identical inputs through each repo's full normalization + forward path, exercising the
norm-stats application (AC-4); pinned training parity is re-confirmed under the corrected config (AC-5); and
behavior divergence is quantified on fixed inputs (AC-6). Any genuine bug found (a converter key drop, a
norm-stats misresolution or misapplication, a stray config key) is fixed, and the corrected end-to-end eval
is re-run.

### Lower Bound (Minimum Acceptable Scope)
The canonical norm-stats source is established and the RLinf train+eval norm-stats are proven to match it
(AC-1); the converter is proven value-lossless on fixed inputs with a structural state-dict check (AC-2); the
eval `num_steps` is confirmed to be the tuned `5` and recorded with each eval; the step-1 loss divergence is
root-caused on identical inputs through each repo's full normalization + forward path (AC-4); and the eval
gap is re-measured under a paired or CI protocol so it is either shown to be within eval noise OR localized
to a specific, named cause (AC-3). AC-5/AC-6 are evidence that scope the remaining investigation but do not
require new training runs if AC-1–AC-4 already explain the gap.

### Allowed Choices
- Can use: the existing RLinf SFT/eval/converter code paths and configs; the reference repo and its committed
  logs/checkpoints/norm-stats as ground truth; the reference py-3.11 venv for cross-checks; pinned-input
  harnesses (fixed batch + injected noise/time) and fixed-seed/paired eval to control stochasticity; committed
  CSV/JSON run-evidence artifacts with commands, resolved configs, source revisions, and asset hashes;
  injecting a fixed `noise`/`time`/`rng` into the eval action-sampling path for a paired/deterministic eval.
- Cannot use: reopening the already-proven training-step alignment (forward/grad/optimizer/LR/FSDP) unless a
  pinned-input parity test fails; treating an unpinned per-step or single-unpaired-eval delta as decisive
  evidence; judging converter correctness by success rate instead of fixed-input parity; comparing the two
  models at mismatched eval knobs (`num_steps`, dtype, norm-stats, episode set); asserting norm-stats
  alignment from config strings rather than resolved file content; raw cross-run weight deltas as a bug signal.

> **Note on Deterministic Designs**: The eval-pairing, converter value-parity, and norm-stats file-identity
> checks are prescriptive (fixed inputs, hash identity, recorded knobs); the genuinely open choices (the
> material eval-gap threshold, the canonical norm-stats source if the empirical check is ambiguous) are
> isolated in `## Pending User Decisions`.

## Feasibility Hints and Suggestions

> **Note**: For reference and understanding only — conceptual, not prescriptive.

### Conceptual Approach
1. **Norm-stats first (cheap, high-yield).** Read the reference loader's resolved norm-stats path from the
   reference config/run; hash it; compare to the OLD and NEW RLinf `assets_dir` files. Diff the quantile
   stats of OLD vs NEW to quantify what the `assets_dir` switch changed. Decide the canonical source
   empirically, then confirm RLinf's train-config and eval-config both resolve it (read the resolved Hydra
   config / loader logs, not the YAML string).
2. **Converter value-parity (independent of training).** Take one saved RLinf SFT checkpoint
   (`.../checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt`), build the pre-conversion
   `OpenPiPytorchActionModel` wrapper from it, and also run `sft_to_new_pytorch.py` on it. On a fixed
   observation + fixed injected noise/time, compare the wrapper's and the converted model's per-timestep loss
   and sampled action chunk (normalized AND denormalized). Separately, meta-load both the converted and the
   reference new-format checkpoints and diff key-set/shape/dtype. This attacks the highest-risk new component
   without any training run.
3. **Eval-gap significance.** Inject a fixed per-episode noise sequence (and fixed reset/initial states) so
   the two models see paired inputs, and/or run many more episodes/seeds; report the gap with a confidence
   interval. Re-run at the tuned `num_steps` (the prior lesson: `num_steps` materially changes success). Use
   the existing control (reference checkpoint through RLinf eval) as the calibration point.
4. **Step-1 delta + pinned re-confirmation.** Reuse the established pinned-input harness; add an OLD-vs-NEW
   norm-stats axis to attribute the step-1 delta; confirm the 50/50 first-N parity still holds under the
   canonical stats.
5. **Behavior divergence.** Compare the two trained-and-converted models by action chunks on fixed eval
   observations + fixed noise (not raw weight deltas).

### Relevant References
- `rlinf/models/embodiment/openpi_pytorch/utils/sft_to_new_pytorch.py` — the new SFT→new-format converter
  (prefix-strip + bf16 cast + norm-stats copy); `utils/export_sft_checkpoint.py` provides `_as_state_dict` /
  `_strip_wrapper_prefix`.
- `rlinf/models/embodiment/openpi_pytorch/__init__.py` — the eval model factory (`get_model` /
  `_build_openpi_pytorch`): bf16 eval load, `num_steps` resolution, norm-stats resolution.
- `rlinf/models/embodiment/openpi_pytorch/openpi_action_model.py` + `pi0_model/pi0.py` — `predict_action_batch`
  / `sample_actions` (the flow-matching Euler loop; `noise=None` → stochastic `torch.randn`), and
  `compute_loss` (already accepts injected `noise`/`time`).
- `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml` (+ `..._eval2.yaml`) — eval configs;
  both set `num_steps: 10` (the model template `model/pi0_5_pytorch.yaml` sets the tuned `num_steps: 5`).
- `examples/sft/config/behavior_pi05_vla.yaml` — SFT config (the `openpi.assets_dir` switched in the prior
  config commit); `rlinf/data/datasets/behavior/` — the BEHAVIOR SFT loader that applies norm-stats.
- The reference: `openpi-comet-pytorch-mixed` `pi05_b1k-task0000_sft_pytorch_mixed` config + its
  `outputs/assets/train/.../norm_stats.json` + the committed reference loss log; the reference-trained eval
  checkpoint `/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew`.
- Prior evidence (do not re-derive): `docs/evidence/r36_pinned_residual.json`, `r37_pinned_first50.{csv,json}`,
  `r38_advisory_1h_trend.{csv,json}`, `docs/sft-first50-step-evidence.md`.

## Dependencies and Sequence

### Milestones
1. **M1 — Norm-stats canonicalization (AC-1).**
   - Phase A: read the reference loader's resolved norm-stats path; hash OLD vs NEW vs reference; diff stats.
   - Phase B: establish the canonical source; confirm RLinf train + eval both resolve it from their dumped
     configs/logs; record the alignment + the OLD-vs-NEW stat diff.
2. **M2 — Converter value-parity (AC-2).** Depends only on a saved SFT checkpoint (not M1).
   - Step 1: fixed-input loss/action parity, pre- vs post-conversion (normalized + denormalized).
   - Step 2: structural state-dict key/shape/dtype check vs the reference new-format checkpoint; fix any gap.
3. **M3 — Eval protocol + significance (AC-3, `num_steps`).** Depends on M1 (canonical stats) + M2 (trusted
   converter).
   - Step 1: confirm the eval `num_steps` is the tuned `5` (verified; optionally drop the stray
     `actor.num_steps: 10`); add a paired (fixed-noise + fixed-initial-state) and/or
     multi-seed CI eval protocol; record all knobs.
   - Step 2: re-measure the RLinf-vs-reference gap with CIs; apply the DEC-1 threshold; calibrate with the
     reference-through-RLinf-eval control.
4. **M4 — Pinned re-confirmation + step-1 isolation (AC-5, AC-4).** Depends on M1.
   - Reuse the pinned harness under the canonical stats; add the OLD-vs-NEW norm-stats axis for the step-1
     delta; confirm 50/50 first-N parity holds (or open a scoped investigation only if it fails).
5. **M5 — Behavior divergence + write-up (AC-6).** Depends on M2, M3.
   - Action-chunk deltas on fixed observations; synthesize root cause(s) and committed fixes/evidence.

Dependency summary: M1 and M2 are independent and come first (cheapest, highest-yield). M3 needs M1+M2. M4
needs M1. M5 needs M2+M3. A confirmed bug in any milestone (norm-stats misresolution, converter key drop,
`num_steps` misconfig) is fixed and the affected downstream eval re-run.

## Task Breakdown

Each task has exactly one routing tag: `coding` (Claude) or `analyze` (Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Record + verify the confirmed canonical norm-stats source — the reference `TrainConfig` default `assets_base_dir="./outputs/assets/train"` → `assets_dirs/{name}/{asset_id}/norm_stats.json` resolves to the NEW file (`ff7e1ff0…`); hash OLD vs NEW vs the reference-resolved file and diff their quantile stats to quantify what prior runs (on the OLD `d66ed168…` file) were exposed to | AC-1 | coding | - |
| task2 | Confirm RLinf SFT-train and eval both resolve the canonical norm-stats (from dumped/resolved configs + loader logs, not YAML strings); align if not; record the OLD-vs-NEW stat-diff evidence | AC-1 | coding | task1 |
| task3 | Build a converter value-parity harness: pre-conversion wrapper vs `sft_to_new_pytorch.py` output on a fixed observation + injected noise/time; compare per-timestep loss + sampled action chunk (normalized + denormalized) within bf16 tolerance | AC-2 | coding | - |
| task4 | Structural state-dict check: meta-load the converted checkpoint vs the reference new-format checkpoint; assert key-set/shape/dtype identity (no missing buffers, no wrapper prefixes, bf16 floats); fix any converter gap found | AC-2 | coding | task3 |
| task5 | Add a paired/deterministic eval mode (inject a fixed per-episode noise sequence + fixed initial states) and/or a multi-seed CI eval; confirm `num_steps` is the tuned `5` (verified) and record it + all eval knobs + asset hashes per run; optionally drop the stray unused `actor.num_steps: 10` | AC-3 | coding | task2, task4 |
| task6 | Re-measure the RLinf-trained vs reference-trained eval gap under the paired/CI protocol at the tuned `num_steps`; apply the DEC-1 material threshold; calibrate with the reference-through-RLinf-eval control; report gap + CI | AC-3 | coding | task5 |
| task7 | Pinned-input first-N training-step parity re-confirmation under the canonical norm-stats/config; assert the prior `|Δ|≤0.03` gate still holds | AC-5 | coding | task2 |
| task8 | Same-batch step-1 root-cause (CORE): drive BOTH repos' FULL normalization + forward path on the IDENTICAL raw first batch + noise/time (raw actions → each repo's own quantile normalization → loss), plus a norm-stats matrix (OLD vs NEW vs reference-resolved), to reproduce + attribute the `0.349`-vs-`0.246` step-1 delta to the norm-stats file/application vs batch/RNG; fix any same-input divergence | AC-4 | coding | task1, task2, task7 |
| task9 | Behavior divergence: action-chunk deltas (post-denormalization) between the two trained-and-converted models on fixed eval observations + fixed noise; interpret against the AC-3 significance result | AC-6 | coding | task4, task6 |
| task10 | Independent audit of the converter + norm-stats resolution + eval-knob matching for missed correctness gaps (e.g. EMA/non-EMA, dtype, denormalization path), cross-checked against the reference | AC-2, AC-1, AC-3 | analyze | task4, task5 |

## Claude-Codex Deliberation

### Agreements
- The AC boundaries are reasonable and correctly avoid reopening the already-proven FSDP/optimizer/training
  alignment unless a pinned-input parity test fails. AC-1, AC-2, AC-3, AC-5, AC-6 are the right core gates.
- The converter is the highest-risk new component and must be value-verified independent of training (fixed
  inputs), before any success-rate reasoning.
- The reported eval gap (8/128 ≈ 1.5 SE, stochastic + unpaired) must be established as statistically real
  (paired seeds and/or CIs) before being treated as a defect.
- Norm-stats alignment must be verified by resolved file content across train-load, converter output, eval
  model-load, and the reference path — not by config strings.
- Production per-step / cross-run weight differences are benign (independent noise/time RNG + shuffle);
  divergence should be measured by behavior on fixed inputs, not raw tensors.

### Resolved Disagreements
- **AC-4 status (Codex DISAGREE → user override → reconciled as CORE):** Codex first argued the step-1-loss
  delta should be merely *diagnostic* (an unpinned step-1 log delta is weak evidence given proven pinned
  parity + RNG/shuffle divergence). The user (annotation) deems it important and asked to dive deep. These are
  reconciled: AC-4 is elevated to a **CORE** root-cause objective (the user is right it matters), while the
  METHOD is constrained to a SAME-BATCH comparison through each repo's full normalization + forward path (Codex
  is right that comparing logged values on *different* batches is not valid evidence). The key justification:
  the prior pinned proofs used PRE-normalized dumped batches, so the norm-stats *application* (raw→normalized)
  was never validated — AC-4 deliberately exercises that one untested surface, which is the most likely source
  of a real `0.349`-vs-`0.246` step-1 difference if one exists.
- **Converter-before-success ordering (Codex REQUIRED_CHANGE):** AC-2 is sequenced before any success-rate
  reasoning (M2 before M3), and converter correctness is judged by fixed-input parity, not success rate.
- **Norm-stats breadth (Codex REQUIRED_CHANGE):** AC-1 explicitly spans all four touchpoints (train-load,
  converter output, eval-load, reference path), including the eval-norm-stats source used for the
  reference-model control eval (a possible confound).
- **Eval-protocol up front (Codex REQUIRED_CHANGE):** AC-3 fixes the paired-seed/CI protocol and the material
  threshold (DEC-1) before running more evals; every eval records `num_steps`, dtype, denormalization path,
  asset hashes, and config hashes.
- **`num_steps` (corrected after the user's annotation + code re-check):** the eval model factory reads
  `_select("num_steps")` → `actor.model.num_steps`, which resolves to the tuned `5` (model template); the
  actual eval run's resolved config confirms `actor.model.num_steps = 5` and `actor.model.openpi.num_steps =
  5`. The `num_steps: 10` in the eval YAML is a STRAY, unused `actor.num_steps` key (wrong nesting level), not
  read by the model. So eval already runs at the tuned `5` — no correction is required (an earlier draft
  wrongly claimed `10` is used); the only residual is dropping the unused `actor.num_steps: 10`.

### Convergence Status
- Final Status: `converged` (one Codex first-pass + one convergence round; all REQUIRED_CHANGES incorporated;
  the only remaining open items are the user decisions below, which are recorded rather than blocking).

## Pending User Decisions

- DEC-1: **Material eval-gap threshold + hard-gate-vs-trend.**
  - Claude Position: Treat the eval gap as a TREND to confirm first (the eval is stochastic + unpaired;
    `0.2578` vs `0.3203` ≈ 1.5 SE). Establish significance via paired noise seeds and/or more episodes/seeds
    with confidence intervals; propose a default material threshold of a CI-significant gap `> 5%` success.
  - Codex Position: Concurs the gap may be noise; requires the material threshold + paired-seed protocol be
    fixed before running more evals.
  - Tradeoff Summary: A hard absolute gate risks chasing eval noise; a pure trend is too weak. The
    confirm-significance-first approach with a stated threshold balances both.
  - Decision Status: **RESOLVED — Trend; confirm significance first (paired seeds / CIs).** The exact numeric
    material threshold (e.g. `> 5%` CI-significant) is to be finalized with the AC-3 evidence.

- DEC-2: **Eval `num_steps` — verified to already be the tuned 5.**
  - Claude Position: Re-checked at the user's request: the eval model factory's `_select("num_steps")` reads
    `actor.model.num_steps`, which resolves to the tuned `5` (model template), confirmed in the actual eval
    run's resolved config (`actor.model.num_steps = 5`, `actor.model.openpi.num_steps = 5`). The `num_steps:
    10` in the eval YAML is a STRAY, unused `actor.num_steps` key (wrong nesting level). No eval correction is
    needed; optionally drop the stray key for cleanliness.
  - Codex Position: Record `num_steps` with every eval; do not compare models across mismatched `num_steps`.
  - Tradeoff Summary: Since eval already runs at the tuned `5`, the prior "uses 10" concern is void; recording
    `num_steps` per run keeps the operating point explicit and prevents future drift.
  - Decision Status: **RESOLVED — eval uses the tuned `num_steps = 5` (the user's belief was correct; an
    earlier draft wrongly claimed `10`). No correction required; record `num_steps` per eval and optionally
    remove the unused `actor.num_steps: 10`.** (Open sub-item: confirm which `num_steps` the reference's
    *reported* success used, so the absolute calibration matches — expected to also be `5`.)

- DEC-3: **Canonical norm-stats source — confirmed: the NEW file.**
  - Claude Position: Resolved empirically this refinement by reading the reference `TrainConfig`
    (`pi05_b1k-task0000_sft_pytorch_mixed`): it sets no explicit `assets_dir`, so the default
    `assets_base_dir = "./outputs/assets/train"` → `assets_dirs/{name}` → `_load_norm_stats({assets_dirs}/
    {asset_id})` with `asset_id = repo_id` resolves to the NEW file
    (`outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json`,
    sha256 `ff7e1ff0…`). RLinf's `assets_dir` switch already points there; prior Phase-3 work used the OLD
    `d66ed168…` file (the misalignment).
  - Codex Position: Verify by resolved file content, not config text.
  - Tradeoff Summary: The canonical file is now established from the reference's own config resolution; the
    only remaining action is the byte-identity check on RLinf's train+eval paths (AC-1 / task1, task2).
  - Decision Status: **RESOLVED — the canonical norm-stats is the NEW file (`ff7e1ff0…`), confirmed via the
    reference config's default `assets_base_dir` resolution. RLinf is aligned (the user's `assets_dir`
    switch); task1/task2 formally verify byte-identity across RLinf train+eval and quantify the OLD-vs-NEW
    diff prior runs used.**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone",
  "Step", "Phase", "task<N>", or similar workflow markers. These belong in this plan document only.
- Use descriptive, domain-appropriate naming in code instead.
- New behavior must be covered by tests/evidence; experiment/eval/run outputs go under the project scratch
  location, never `/tmp`; preserve self-containment (no real `import openpi` in the package or relocated data
  modules) and do not change the in-repo old `openpi/` package's behavior.


--- Original Design Draft Start ---

# Task: Verify SFT Alignment & Conversion Correctness

## Objective
Carefully verify whether the SFT training code in **RLinf** is *fully aligned* with the
**reference repo** (openpi), and whether the conversion script (`sft_to_new_pytorch.py`)
is correct. Given identical inputs, the two repos should be numerically equivalent, but a
measurable gap remains — so something is likely still misaligned. This round's goal is to
find and root-cause that misalignment.

## Repositories
- **RLinf (repo under test):** `/mnt/public/xzxuan/repos/RLinf_pi05/`
- **Reference (ground truth, openpi):** `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/`

## What I've Done So Far

### 1. SFT training (RLinf)
- Command: `bash examples/sft/run_vla_sft.sh behavior_pi05_vla`
- Output: `logs/20260605-12:39:44-behavior_pi05_vla/`

### 2. Checkpoint conversion for eval
- I wrote a new conversion script:
  `rlinf/models/embodiment/openpi_pytorch/utils/sft_to_new_pytorch.py`
- Running it produced the eval-ready checkpoint:
  `logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new`

### 3. Evaluation (RLinf-trained model)
- Command: `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_eval`
- Log: `logs/20260607-08:13:16-behavior_ppo_openpi_pi05_pytorch_eval/`
- Model used: the trained + converted RLinf checkpoint
  `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/pi05_sft_pytorch_new`
- Result over **128 episodes**: `'eval/success_once': array(0.2578125, dtype=float32)`

### 4. Control eval (reference-trained model)
- I copied a config:
  `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval2.yaml`
- **Only difference:** `model` and `asset` swapped to
  `/mnt/public/xzxuan/models/ckpt/jax_task0000_sft_29999_ptnew`
  (a model trained on the **reference repo**)
- Log: `logs/20260607-08:50:42-behavior_ppo_openpi_pi05_pytorch_eval2/`
- Result over 128 episodes: `'eval/success_once': array(0.3203125, dtype=float32)`

### Core problem
Swapping **only the model** yields a sizable gap:
**0.2578 (RLinf-trained) vs 0.3203 (reference-trained)**.
This strongly suggests the SFT training and/or the conversion is not yet fully aligned.

## Loss & Grad-Norm Comparison

To check whether RLinf's and the reference's loss / grad-norm are aligned, I wrote
`/mnt/public/xzxuan/plot_loss_curves.py`; its output (4 plots) is in
`/mnt/public/xzxuan/loss_curves/`. **Examine all four plots carefully.** My observations:

1. **Overall loss & grad-norm look aligned, but RLinf's loss is systematically lower.**
   In `loss_log.png`, RLinf's loss appears consistently below openpi's (reference).
   Investigate why.

2. **The first-step loss differs significantly — likely a real misalignment at step 0.**
   - RLinf
     (`/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla/run_embodiment.log`):
     step 1 `train/loss=0.349`, step 2 `train/loss=0.266`
   - Reference
     (`/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/logs/pi05_b1k-pt-2k-8gpu-fmp-wo_prefetch-xzx.log`):
     step 1 `train/loss=0.246`
   - This gap is large. **Feed the *same input* to both repos, determine the exact
     first-step loss in each, and explain precisely why they differ.**

3. **Validate the loss / grad-norm computation itself.** Deeply analyze whether each repo
   computes loss and grad-norm correctly — this may require digging into the FSDP source.
   Confirm the plots are correct and faithfully reflect the true model loss.

## Debug Directions

1. **Initial config alignment — model path, asset, norm-stats, hyperparameters.**
   - The `asset` path I used previously was **wrong**
     (`/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets`).
   - In the commit *"chore(config): point BEHAVIOR norm-stats assets_dir at the reference
     task-0000 outputs"* I updated it to
     `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed`.
   - **Re-confirm both repos use this same norm-stats so they are aligned.**

2. **SFT training code — bit-for-bit reproducibility.** Given identical input through the
   same model, verify both repos produce the same forward output, the same gradients, the
   same parameter updates, an identical LR schedule/logic, etc.

3. **Conversion script `sft_to_new_pytorch.py` — correctness.** Verify the conversion logic
   is right, that converting this way introduces no bug and no precision loss, etc.

4. **Weight-change comparison.** Compare how the weights of the two trained models differ.
   If the two repos are strictly aligned, the post-training models should be very close —
   so quantify the divergence.

## Deliverable
For each direction above, report findings, root-cause any divergence, and propose fixes so
that RLinf's SFT + conversion become numerically equivalent to the reference repo.
--- Original Design Draft End ---
