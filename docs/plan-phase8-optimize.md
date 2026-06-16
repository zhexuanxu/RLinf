# Implementation Plan — Optimize pi0.5 VLM_VLA SFT to BEHAVIOR `success_once` ≥ 0.25

## Goal Description

Find the ROOT CAUSE of why four-subtask `vlm_vla` SFT trains cleanly (loss down, language accuracy up, correct subtask emitted at eval, posture visibly learned) yet evaluates at ~0 `success_once` (the broken run scored 1/64 = 0.0156), while the byte-compatible pure-VLA baseline reaches 0.25 — then fix it so that the **four-subtask `vlm_vla` mode**, with the VLM **freely generating** the subtask autoregressively at eval, reaches **`success_once` ≥ 0.25** after `sft2new` conversion and an env evaluation through the unmodified eval path.

This is a debugging-and-optimization effort, not a feature build: the `vlm_vla` mode already exists (delivered by the prior plan `docs/plan-phase8-vlm-vla-mode.md`, ported from `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05`). The single hard gate is the performance number; "by whatever method" is allowed, subject to the boundaries below (pure-`vla` numerics must stay untouched, and the final eval must use the real four-subtask free-generation path — the constant-subtask control is a diagnostic only).

Operating constraints carried from the draft:
- Interpreter `/mnt/public/xzxuan/.venv_pi/bin/python`; scratch/test outputs under `/mnt/public/xzxuan/tmp`.
- Hardware: local 8 GPUs plus an identical 8-GPU box via `ssh -p 40431 root@183.233.148.6` with shared storage — effectively 16 GPUs, two full experiments in parallel. A full SFT run takes ~30h. First confirm you can log in, load the env, see the GPUs, and run scripts on both boxes.
- The actual training/eval config for any run is recorded at `tensorboard/config.yaml` inside its log directory — read it to inspect what truly ran.
- Commands: train `bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_vla`; eval `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`.
- Required hook: when a training run finishes, auto-convert its checkpoint with `python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new` (with the path arguments the converter actually requires), then run eval.
- Reference papers to study: OpenPI / pi0.5 (`https://arxiv.org/abs/2504.16054`) and Knowledge Insulating VLAs (`https://arxiv.org/abs/2505.23705`).
- Prior context: `docs/plan-phase8-vlm-vla-mode.md` and the last RLCR loop under `.humanize/rlcr/2026-06-11_10-03-58`. Work on the current git branch.

### Decisive diagnostic already established

The user's control experiment (`examples/sft/config/behavior_pi05_vlm_vla_test.yaml`) is byte-identical to the broken config except that all four `task_subtasks` collapse to the constant string "turning on radio". It reached **training accuracy 1.0**, **emitted the correct text at eval**, and still scored `success_once` ≈ 0.0156. Making the language task trivial does NOT recover action success — so the failure lives in the **action path** (training dynamics and/or the action-expert's eval context), not in subtask-language difficulty. The plan is built around this fact.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification. Per the user, **AC-1 is the single hard acceptance gate**; AC-2 through AC-4 are process guardrails and verification requirements that make AC-1 reachable and trustworthy, not additional product features.

- AC-1: Four-subtask `vlm_vla` SFT, evaluated through the unmodified env-eval path with the VLM freely generating the subtask, reaches `success_once` ≥ 0.25.
  - Positive Tests (expected to PASS):
    - A `vlm_vla` SFT run that trains on the four real `turning_on_radio` subtasks, is auto-converted via `--mode sft2new`, and is evaluated by `eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval` logs `eval/success_once` ≥ 0.25 over the iteration gate of 64 trajectories.
    - The result is re-confirmed at the final gate of 256 trajectories with `eval/success_once` ≥ 0.25.
    - The eval used greedy free generation (the standard `vlm_vla` path); per the user, the generated subtask text is NOT required to match any of the four reference strings — `success_once` is the sole measure.
  - Negative Tests (expected to FAIL / be rejected):
    - A `success_once` ≥ 0.25 obtained on the constant-subtask control config does NOT satisfy AC-1.
    - A number obtained only via a teacher-forced or subtask-constrained decoding path (diagnostic paths) does NOT satisfy AC-1; AC-1 requires the standard free-generation eval.
    - Reducing the eval trajectory count below the agreed gate to inflate the average is rejected.
    - Any run that reaches the number by altering pure-`vla`-mode numerics is rejected (see AC-4 / DEC-3).

- AC-2: The diagnostic and automation guardrails exist and are exercised BEFORE any full ~30h run.
  - Positive Tests (expected to PASS):
    - The train→convert(`sft2new`)→eval hook runs end-to-end unattended and emits an `eval/success_once` number.
    - Gradient-norm instrumentation reports, on a fixed batch, the pre-clip total norm and the CE-only / flow-only / combined contributions bucketed by VLM vs action-expert parameters, for `stop_gradient_to_vlm` both False and True.
    - An offline parity harness compares the joint training forward against the eval static-cache denoise on one batch under inference preprocessing, fixed noise and time, and teacher-forced non-EOS subtask tokens; it prints a max-absolute-velocity difference and a PASS/FAIL against a tolerance of 1e-3 (bf16).
    - A held-out, teacher-forced normalized action-MSE proxy is computed for both the working baseline-VLA checkpoint and the broken `vlm_vla` checkpoint on identical frames.
    - 16-GPU control (login, env, GPU visibility, script execution) is verified on both boxes.
  - Negative Tests (expected to FAIL / be rejected):
    - Launching a full ~30h run before the gradient-norm audit and the baseline-vs-broken action-MSE control have produced numbers is rejected.
    - A hook that converts or evaluates the wrong checkpoint path (mismatched `--output-norm-stats` / model dir) is rejected.

- AC-3: The root cause is identified and documented with before/after evidence that ties the applied fix to the `success_once` gain.
  - Positive Tests (expected to PASS):
    - A short writeup names the failing stage, cites the diagnostic numbers that localize it, and shows the metric moving from ~0.0156 toward ≥ 0.25 after the fix.
  - Negative Tests (expected to FAIL / be rejected):
    - "It works now" with no diagnostic evidence linking cause to fix is rejected.

- AC-4: Changes are contained, pure-`vla` numerics are preserved, and the result is reproducible.
  - Positive Tests (expected to PASS):
    - With `mode: vla` (default), the tokenizer string, the SFT loss, and sampled actions are unchanged; a pinned-behavior `vla` regression check passes.
    - The winning run records its config, git commit, and converted-checkpoint path so the eval number can be reproduced.
  - Negative Tests (expected to FAIL / be rejected):
    - Any diff that changes `vla`-mode numerics or the `vla` tokenizer string is rejected.
    - A winning number that cannot be reproduced from the recorded config/commit/checkpoint is rejected.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

A full systematic investigation: all hypotheses below are instrumented and tested (gradient-norm audit, `stop_gradient` gradient-probe plus VLM weight-drift-from-base, offline train/eval action parity, baseline-vs-broken teacher-forced action-MSE control, dataset frame-distribution audit); multiple hypothesis-driven fixes are tried as short probes and the survivors are run to full length on both boxes; the four-subtask `vlm_vla` mode reaches ≥ 0.25 robustly (confirmed at 256 trajectories); pure-`vla` numerics are preserved; the change set is contained with regression tests (a `vla` pinned-behavior test, mask/gradient probes) and a documented root cause; the train→convert→eval hook and the diagnostic scripts are reusable.

### Lower Bound (Minimum Acceptable Scope)

The minimal diagnostic set that pinpoints the root cause, plus the smallest fix that moves four-subtask `vlm_vla` to `success_once` ≥ 0.25 through the standard free-generation eval, plus the train→convert→eval hook so the result is produced automatically. Pure-`vla` numerics remain untouched and the winning configuration is reproducible.

### Allowed Choices

- Can use: any method that reaches the target while keeping `vla` numerics intact — tuning `language_loss_weight` / `action_loss_weight`, `stop_gradient_to_vlm`, learning rate, and gradient clipping (global or per-expert); training-recipe changes (action-expert warmup, a two-stage CE schedule, decoupled per-expert clipping or optimizers); eval/generation-path fixes; and dataset frame-selection changes, including (per DEC-7) training the action loss on the level-0 / all-frames distribution or a reweighted mixture while still using the four subtask labels for the CE loss. Both 8-GPU boxes may be used continuously, iterating until the target is met (DEC-4). Instrumentation may be added to the model and the SFT worker/strategy. `stop_gradient_to_vlm=True` is the preferred working default unless the gradient audit disproves it (DEC-6, Knowledge-Insulation aligned).
- Cannot use: any change to `vla`-mode numerics or the `vla` tokenizer string (DEC-3); the constant-subtask control as evidence of success; a teacher-forced or subtask-constrained eval as the AC-1 path; silently shrinking the eval trajectory count to inflate the gate.

> **Note on Deterministic Designs**: The goal is fixed and singular (four-subtask `vlm_vla` reaching `success_once` ≥ 0.25 via free generation, `vla` preserved), but the *route* is deliberately open ("by whatever method"). The boundaries are therefore narrow on the END STATE and the protected `vla` path, and wide on the investigative and fix choices.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

Work economics-first: every full run costs ~30h, so spend cheap, high-information diagnostics (hours) before committing GPU-days.

Ranked root-cause hypotheses, each with its cheapest decisive check:

1. **H1a — Global gradient-clip coupling (lead).** Training uses one optimizer over all trainable parameters and a single global gradient-norm clip at 1.0. The CE loss back-propagates through the shared VLM backbone, so its gradient inflates the global norm and the single clip coefficient scales DOWN the action-expert gradients too. The baseline `vla` run has no CE term and therefore no such coupling. The *mechanism* is verified from the code; the *magnitude* is not — confirm with the gradient-norm audit. Fix family: a larger or per-expert clip.
2. **H1b — CE/action loss-scale imbalance (lead).** Even with both logged losses falling, the CE term may dominate the update direction. Same audit distinguishes this from H1a (compare CE-only vs flow-only contributions). Fix family: loss reweighting or a CE schedule.
3. **H2 — VLM pollution / objective conflict (Knowledge Insulation).** With `stop_gradient_to_vlm=False`, the flow loss also reshapes the VLM representations the action expert reads. The KI paper argues the action gradient should not pollute the VLM. Check via a gradient probe (action-expert norms with the flag True vs False) and VLM weight-drift-from-base after a short probe. Caveat: even with the flag True, CE still trains the VLM and still contributes to the global clip norm — so H2 and H1 may need to be fixed together.
4. **H3 — Prompt/template shift.** `vlm_vla` conditions the action expert on a different prefix (the "Subtask:" template plus subtask tokens in the cache) than the baseline action-only prompt. Check by measuring the broken checkpoint's offline action MSE under the `vlm_vla` prompt vs a baseline-style prompt on identical observations.
5. **H4 — Eval generation / static-cache numeric mismatch (cheap rule-out).** Code reading indicates train and eval are consistent by design (the training position cumsum excludes EOS specifically so the suffix RoPE matches eval, where EOS is never cached); the divergence is only versus the `vla_lib` reference and is intentional. Confirm empirically with the offline parity harness (expected PASS) and a teacher-forced env eval of the broken checkpoint.
6. **H5 — Level-1 action-frame distribution.** `vlm_vla` (fine_grained_level 1) selects action-training frames by subtask window with gap reassignment and `valid_duration` filtering, unlike the level-0 baseline. Audit and compare the (frame→action) distributions at level 0 vs level 1; run an `enable_gap=false` probe.

Two mandatory pre-full-run gates:
- **Gradient-norm audit**: on one fixed batch, log the pre-clip total norm and the CE-only / flow-only / combined contributions, bucketed by VLM vs action-expert parameters, for `stop_gradient_to_vlm` both False and True.
- **Baseline-vs-broken action-MSE control**: held-out, teacher-forced normalized action MSE for the working baseline-VLA checkpoint (`M_base`) and the broken `vlm_vla` checkpoint on identical frames. This directly tests the action-path-failure thesis without a rollout, and is the gating proxy for short probes: a probe is promoted to a full run only if its teacher-forced action MSE falls toward `M_base` (predefined threshold: within 1.5 × `M_base`).

Offline parity acceptance (pin): joint training forward vs eval static-cache denoise on the same batch, under inference preprocessing (no train augmentation), fixed noise and time, teacher-forced non-EOS subtask tokens (EOS excluded from the cache-validity mask), identical masks and identical denoise steps; PASS iff the max absolute velocity difference ≤ 1e-3 (bf16). A PASS exonerates the eval mechanics; a FAIL localizes the bug to the eval path.

Hook design: the SFT and eval entrypoints are separate processes/configs, so the cleanest hook is a shell wrapper around `run_vla_sft.sh` that, on training completion, resolves the latest checkpoint, runs the `sft2new` converter with explicit `--ckpt` / `--input-norm-stats` / `--output-model` / `--output-norm-stats`, then invokes `eval_embodiment.sh`. (An in-process alternative exists in the SFT runner's checkpoint-save path if a wrapper proves insufficient.)

### Relevant References

- `rlinf/models/embodiment/openpi_pytorch/pi0_model/pi0.py` — `compute_loss` (the `action_loss_weight * action_loss + language_loss_weight * ce_loss` combine and the returned `loss`/`action_loss`/`language_loss`/`language_acc` dict), `_compute_language_loss` (tied-embedder decode, next-token shift, loss-mask normalization, `language_acc`), the training position-cumsum that excludes EOS, `make_attn_mask` / `block_suffix_from_excluded_prefix` (suffix-from-EOS blocking), `generate_language` and `_denoise_actions` / `reason_and_sample_actions` (eval generation + denoise), and the `detach_prefix_kv` setup for `stop_gradient_to_vlm`.
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/gemma.py` — the split-attention detach implementation for `stop_gradient_to_vlm`; the frozen-cache prefix+suffix K/V concatenation.
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/static_kv_cache.py` — `StaticKVCache` writes/freeze and `left_to_right_align`.
- `rlinf/models/embodiment/openpi_pytorch/openpi_action_model.py` — `predict_action_batch` `vlm_vla` branch and the (currently effectively unbounded) per-batch generation logging constant.
- `rlinf/models/embodiment/openpi_pytorch/utils/tokenizer.py` — the `vla` prompt string (must stay byte-identical) and the `vlm_vla` "Subtask:" template.
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` and `rlinf/workers/sft/fsdp_sft_worker.py` — the generic scalar-metric extraction, the micro-batch/grad-accum loop, and the `.backward()` call.
- `rlinf/hybrid_engines/fsdp/fsdp_model_manager.py` and `rlinf/hybrid_engines/fsdp/strategy/fsdp.py` — the single AdamW construction and the single global `clip_grad_norm_` (the coupling site); the only place a per-expert clip or per-module gradient norm would be added.
- `rlinf/data/datasets/openpi_pytorch/behavior/behavior_sft_dataset.py`, `behavior_sft_data_loader.py`, `skill_segments.py` — level-0 vs level-1 frame selection, gap reassignment, and `valid_duration` filtering.
- `rlinf/envs/behavior/behavior_env.py` and `rlinf/utils/metric_utils.py` — `success_once` is a per-trajectory boolean averaged to a fraction in [0,1]; trajectory counting.
- `rlinf/utils/ckpt_convertor/openpi/convert.py` and `sft2new.py` — `--mode sft2new` (strip wrapper prefixes, cast bf16, write `model.safetensors` + `config.json`, copy norm stats); required path arguments.
- `examples/sft/config/behavior_pi05_vlm_vla.yaml`, `behavior_pi05_vlm_vla_test.yaml`, `behavior_pi05_vla.yaml` — the broken, control, and baseline configs.
- `examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval.yaml` — the eval config (8 envs × 8 epochs = 64 trajectories at the iteration gate; raise the rollout breadth for the 256-trajectory final gate).
- `examples/sft/run_vla_sft.sh`, `examples/embodiment/eval_embodiment.sh` — train/eval launch wrappers (hook attach points).
- `docs/plan-phase8-vlm-vla-mode.md`, `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05/` — the prior plan and the porting reference.

## Dependencies and Sequence

### Milestones

1. Milestone M0 — Instrumentation and confirmation:
   - Phase A: verify 16-GPU control on both boxes (login, env, GPUs, script execution).
   - Phase B: build the train→convert(`sft2new`)→eval shell hook with explicit converter path arguments; keep the iteration eval at 64 trajectories and provide a 256-trajectory final-confirmation variant.
   - Phase C: add per-module and pre/post-clip gradient-norm logging (CE-only / flow-only / combined); add the offline parity harness; add a teacher-forced eval path plus the held-out teacher-forced action-MSE proxy and the baseline-vs-broken control script; (lower priority) replace the unbounded generation text log with a structured summary (EOS rate, length histogram, unique strings, per-subtask confusion).
2. Milestone M1 — Cheap offline diagnostics (no full run):
   - Phase A: gradient-norm audit (H1a / H1b) and the `stop_gradient` gradient-probe plus VLM weight-drift (H2).
   - Phase B: baseline-vs-broken action-MSE control and a teacher-forced env eval of the broken checkpoint (H3 / H4); offline parity (H4).
   - Phase C: dataset frame-distribution audit level 0 vs level 1 and an `enable_gap=false` check (H5). Output: an evidence-backed, ranked root cause.
3. Milestone M2 — Short hypothesis probes (few-hundred steps, gated by the action-MSE proxy): loss reweighting; `stop_gradient_to_vlm=True`; per-expert or larger gradient clip; action-expert warmup or two-stage CE; (DEC-7) level-0 action-frame mixture. Keep only probes that move the proxy toward `M_base`.
4. Milestone M3 — Full runs of the surviving candidates (up to two in parallel on the 16 GPUs), each with the auto convert+eval hook; iterate freely until `eval/success_once` ≥ 0.25 at the iteration gate, then confirm at the 256-trajectory gate.
5. Milestone M4 — Confirmation and closeout: document the root cause with before/after evidence; finalize the contained change set with a `vla` regression test and the mask/gradient probes; record the reproducibility triple (config, commit, converted-checkpoint path).

Dependencies: M1 depends on M0's instrumentation; M2 depends on M1's ranking; M3 depends on M2's survivors; M4 depends on a passing M3. The mandatory gates (gradient-norm audit and baseline-vs-broken action-MSE control) must complete before any M3 run.

## Task Breakdown

Each task includes exactly one routing tag.

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Verify 16-GPU control on both boxes (login, env, GPU visibility, run scripts) | AC-2 | coding | - |
| task2 | Train→convert(`sft2new`)→eval shell hook with explicit converter path args; 64-trajectory iteration eval + 256-trajectory final-confirmation variant | AC-1, AC-2 | coding | - |
| task3 | Per-module + pre/post-clip gradient-norm logging (CE-only/flow-only/combined, VLM vs action-expert) in the SFT worker/strategy | AC-2 | coding | - |
| task4 | Offline train-vs-eval action parity harness (inference preprocessing, fixed noise/time, teacher-forced non-EOS tokens, 1e-3 tolerance) | AC-2 | coding | - |
| task5 | Teacher-forced eval path + held-out teacher-forced action-MSE proxy + baseline-vs-broken action-MSE control script | AC-2, AC-3 | coding | - |
| task6 | Structured generation summary (EOS rate, length histogram, unique strings, per-subtask confusion) replacing the unbounded text log (lower priority) | AC-2 | coding | - |
| task7 | Dataset frame-distribution audit level 0 vs level 1 + `enable_gap=false` check | AC-3 | coding | - |
| task8 | Run + interpret the gradient-norm audit and the `stop_gradient` gradient-probe/weight-drift; align findings with the KI and pi0.5 papers | AC-3 | analyze | task3, task5 |
| task9 | Run the cheap offline diagnostic suite (parity, action-MSE control, teacher-forced eval, frame audit) and produce a ranked root cause | AC-3 | analyze | task1, task2, task4, task5, task7 |
| task10 | Short hypothesis probes (loss reweight, `stop_gradient` True, per-expert/larger clip, action warmup, DEC-7 mixture) gated by the action-MSE proxy | AC-1, AC-3 | coding | task8, task9 |
| task11 | Full ~30h runs of survivors with auto convert+eval; iterate to ≥ 0.25 at 64 then confirm at 256 | AC-1 | coding | task10 |
| task12 | Final root-cause writeup + minimality / `vla`-regression review of the complete diff | AC-3, AC-4 | analyze | task11 |

## Claude-Codex Deliberation

### Agreements

- The constant-subtask control isolates the failure to the action path (trivial-but-perfect language still fails), so language difficulty is not the cause.
- `success_once` is a fraction in [0,1]; the broken run's number corresponds to 1/64 trajectories; the target 0.25 is baseline parity.
- The loss combine is `action_loss_weight * action_loss + language_loss_weight * ce_loss`; training uses one optimizer and one global gradient-norm clip; no per-component or pre-clip gradient norms are logged today.
- `stop_gradient_to_vlm` is wired correctly and is False in all three runs; even when True, CE still trains the VLM and still contributes to the global clip norm.
- The cheapest decisive diagnostics are the gradient-norm audit, the baseline-vs-broken teacher-forced action-MSE control, the offline train/eval parity check, and the teacher-forced env eval — all before any ~30h run.
- The hook is best implemented as a shell wrapper around the separate train and eval entrypoints; the converter needs explicit path arguments beyond the draft's bare command.
- Level-1 selects action frames differently than level-0, a real confound worth auditing.

### Resolved Disagreements

- Primary root-cause ranking: first-pass Codex ranked a TRAIN/EVAL position-id mismatch as the top suspect (RLinf excludes EOS from the training position cumsum, unlike the `vla_lib` reference). Claude's code reading showed RLinf's train and eval are consistent **by design** — EOS is excluded from the training cumsum precisely so the suffix RoPE matches eval (where EOS is never cached); the divergence is only versus the reference and is intentional. Resolution (second-pass Codex agreed): the position-id mismatch is largely refuted as the root cause; the lead shifts to global gradient-clip coupling (H1a) and CE/action loss-scale imbalance (H1b), with the offline parity harness retained as the cheap empirical confirmation that eval mechanics are sound.
- H1 strength of claim: Codex required H1 be stated as a verified MECHANISM with UNVERIFIED MAGNITUDE, and split into H1a (clip coupling) and H1b (loss-scale imbalance) because the fixes differ. Resolution: adopted, with the gradient-norm audit mandated before any full run.
- Eval gate sizing: the eval is currently fixed at 64 trajectories (not "variable"). Resolution: iterate on 64 and confirm at 256 (DEC-5).
- Offline parity precision: Codex required an explicit harness contract and tolerance. Resolution: inference preprocessing, fixed noise/time, teacher-forced non-EOS tokens (EOS excluded from the cache-validity mask), and a pinned 1e-3 (bf16) tolerance.

### Convergence Status

- Final Status: `converged` — one first-pass analysis plus two convergence rounds; the second round returned zero required changes and zero high-impact disagreements (two optional polish items — the pinned parity tolerance and a predefined action-MSE proxy threshold — were folded in).

## Pending User Decisions

All decisions were resolved with the user during plan review; none remain `PENDING`.

- DEC-1: Meaning of the "25" target.
  - Claude Position: 0.25 as a fraction (code-confirmed scale; baseline parity).
  - Codex Position: flagged 0.25-vs-25.0 as needing explicit confirmation.
  - Tradeoff Summary: the units define the entire acceptance gate.
  - Decision Status: **0.25 fraction (baseline parity); hard requirement.**
- DEC-2: Eval generation policy.
  - Claude Position: free autoregressive generation is the shipping path; teacher-forcing/constraining are diagnostics.
  - Codex Position: asked whether constrained/teacher-forced output is acceptable.
  - Tradeoff Summary: free generation reflects deployment; constrained/teacher-forced hides failure modes.
  - Decision Status: **Train forces the four subtask labels; at eval the model generates freely and the generated subtask need NOT be checked for consistency — `success_once` is the sole gate. Teacher-forcing remains a diagnostic only.**
- DEC-3: `vla` numerics.
  - Claude Position: preserve `vla` byte-compatibility; guard all changes behind `vlm_vla`.
  - Codex Position: asked whether `vla` regression is forbidden.
  - Tradeoff Summary: protecting `vla` keeps the working baseline safe.
  - Decision Status: **Preserve `vla` numerics.**
- DEC-4: Compute budget.
  - Claude Position: economics-first; approve each full run.
  - Codex Position: asked for an explicit compute ceiling.
  - Tradeoff Summary: more autonomy is faster but spends more GPU-time.
  - Decision Status: **Iterate freely until target on both boxes.**
- DEC-5: Eval trajectory count.
  - Claude Position: keep the current count; raise for final confidence.
  - Codex Position: 64 is supported by current logs; more increases confidence at cost.
  - Tradeoff Summary: small counts iterate fast but have wide error bars near 0.25.
  - Decision Status: **Iterate on 64; confirm the final result at 256.**
- DEC-6: `stop_gradient_to_vlm` default.
  - Claude Position: prefer True (Knowledge Insulation) unless the audit disproves it.
  - Codex Position: True is low-risk and KI-aligned; verify with probes.
  - Tradeoff Summary: True may slow VLM↔subtask adaptation but protects representations the action expert reads.
  - Decision Status: **Prefer True unless the gradient audit disproves it; ablate False to confirm.**
- DEC-7: Action-frame distribution as a fix.
  - Claude Position: allow decoupling action-frame selection from the CE subtask labels if level-1 selection is harmful.
  - Codex Position: raised it as a new decision (DEC-7).
  - Tradeoff Summary: matching the level-0 action distribution may restore action quality while keeping subtask CE supervision.
  - Decision Status: **Allowed as a fix (train action loss on level-0/all frames or a reweighted mixture while using subtask labels for CE).**

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Phase", "Step", or similar workflow markers; these belong to this plan document only.
- Use descriptive, domain-appropriate naming in code instead.

### Repository Conventions

- Google Python style; Ruff lint/format; docstrings and type hints on public APIs; new files need the repository's standard license/copyright header (CI enforces this via `ruff --preview` CPY001 — reproduce locally with `pre-commit run --all-files`).
- Logging via `rlinf.utils.logging.get_logger()` or workers' `self.log_*`; no `print` in library code (standalone diagnostic/inspection scripts whose purpose is printing are exempt).
- Conventional Commits with `Signed-off-by` (`git commit -s`); config YAMLs hold static values only and do not overwrite user-facing fields in code.
- Use `/mnt/public/xzxuan/.venv_pi/bin/python` for all runs; write temporary files and test outputs to `/mnt/public/xzxuan/tmp`. Work on the current git branch.

### Acceptance-Path Clarification (per DEC-2)

The shipping eval is the standard `vlm_vla` free-generation path. The four subtask strings are used as forced CE labels during training only; at eval there is no check that the generated subtask matches any reference string. AC-1 is met purely by `eval/success_once` ≥ 0.25 through that path (iterate at 64 trajectories, confirm at 256). Teacher-forced and subtask-constrained decoders exist only to bisect the root cause.

--- Original Design Draft Start ---

# Optimize VLM_VLA SFT Training Performance on the Behavior Task

In this round, I need you to work carefully, dig deep into the code, perform an in-depth analysis, and optimize the performance of my SFT training.

## Current Situation

### Baseline (pure VLA SFT — works)

A pure VLA SFT run, trained **only** on the Behavior `task 0000` training data, corresponds to the config `examples/sft/config/behavior_pi05_vla.yaml`. The model produced by this run reaches an **eval accuracy of 0.25** in the actual eval.

- Full training log: `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla`
- Full eval log: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-16:08:02-behavior_ppo_openpi_pi05_pytorch_eval`

> Note: this training was run in a **different repository**. The code in that repo and the code in the current repo are **aligned under this VLA training config**.

### The key difference: VLM_VLA mode

Compared with VLM_VLA training, the difference is that the latter:
1. Uses **subtasks**.
2. Splits the SFT loss into **two parts**.
3. At eval time, lets the model **first autoregressively generate the subtask, then generate the action**.

### The problem (VLM_VLA — broken)

The model obtained from SFT training in **VLM_VLA mode** evals very poorly. The config is `examples/sft/config/behavior_pi05_vlm_vla.yaml`.

- My SFT training log: `/mnt/public/xzxuan/repos/RLinf/logs/20260611-13:45:13-behavior_pi05_vlm_vla`
- My eval log: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-15:54:57-behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`

Observed behavior:
- During training, the **SFT loss is decreasing**, and **language accuracy keeps rising**.
- In the eval log, the model **does output a subtask**, and the **output content is at least correct**. Watching the video, the **overall posture is learned** as well.
- However, **`success_once` is very poor (0%)** — i.e., the training result is bad.

### Control experiment

I also trained a **control experiment**: `/mnt/public/xzxuan/repos/RLinf/logs/20260613-05:42:48-behavior_pi05_vlm_vla_test`. The config is `examples/sft/config/behavior_pi05_vlm_vla_test.yaml`

- The **only** modification is using `examples/sft/config/behavior_pi05_vlm_vla_test.yaml` (the run above used `examples/sft/config/behavior_pi05_vla.yaml`). The only difference is that in the `test` yaml, **`task_subtasks` is entirely changed to output `"turning on radio"`** instead of the normal four subtasks.
- Eval log for this control: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-16:15:06-behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`

Results of the control:
- At eval time, the model outputs **only** `'turning on radio.' [EOS]`.
- The **training accuracy stays at 1** throughout.
- The eval **`success_once` is 0.01**.

## Summary

The SFT training performance under **VLM_VLA mode is very poor**. I need you to deeply analyze **which stage is the root cause**. Here are a few directions:

1. **Code correctness (training and eval).** Is the attention handling correct? Is the KV cache handling correct? Etc.
2. **Hyperparameters.** Could a hyperparameter be causing such a stark difference — e.g., the **ratio between the CE loss and the action loss**?
3. **`stop_gradient_to_vlm`.** Should this be turned on? Verify that the logic for setting this config to `true` vs `false` is correct — i.e., if `true`, the VLM portion's weights should **not change** after training. (You can train a few steps, dump a checkpoint, and verify against the base model.)

## Required reference papers

You must carefully study the ideas in these papers (they are official OpenPI papers related to Pi05):
- **OpenPI05**: https://arxiv.org/abs/2504.16054
- **Knowledge Insulating**: https://arxiv.org/abs/2505.23705

## Notes / Constraints

1. My three eval runs used **exactly the same code** (i.e., the current eval in the current repo), to ensure the differences truly come from the **training**.
2. Environment: `/mnt/public/xzxuan/.venv_pi`
3. `/mnt/public/xzxuan/tmp` is for intermediate results and test outputs, etc.
4. For both training and eval, the **full config** can be found in `tensorboard/config.yaml` inside the log directory. Use this to inspect the **actual** training and eval configs.
5. Commands:
   - Training: `bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_vla`
   - Eval: `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`
6. **Hardware.** This machine has **8 GPUs**. You can also access another 8-GPU machine via:
   ```
   ssh -p 40431 root@183.233.148.6
   ```
   The two machines have **identical configurations** and **share storage**, so you effectively have **16 GPUs** and can run two sets of experiments in parallel. Use the same code, environment, and data on both machines — I've already verified it runs. First validate that you can: log in, load the environment, see the GPUs, and execute scripts — make sure you can fully control all 16 GPUs.
7. Refer to `docs/plan-phase8-vlm-vla-mode.md` and `.humanize/rlcr/2026-06-11_10-03-58` for last rlcr loop, where we achieve the full vlm_vla mode code(reference to /mnt/public/xzxuan/repos/vla_lib)
8. Using the current branch for coding.

## Acceptance Criterion

There is **only one** acceptance criterion: by **whatever method**, the final **VLM_VLA mode**, based on the **four different subtask outputs**, must reach a **`success_once` of 25 or above** in eval.

A single training run may take ~30h. Add a hook so that when training finishes, convert the checkpoint into an eval-ready ckpt with:
```
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new
```
then run eval. The final result must reach **25 or above**.
--- Original Design Draft End ---
