# Phase-4 AC-3 — eval-gap significance evidence (expanded matched protocol)

**Verdict: under an EXPANDED deterministic, knob-matched eval (5 seed pairs, 640 episodes/model) the
reference-trained model SIGNIFICANTLY outperforms the RLinf-trained model — pooled `success_once`
`160/640 (25.0%)` vs `219/640 (34.2%)`, a `9.2%` gap (two-proportion `z = 3.61`, `p < 0.001`; 95% CI of the
difference `[4.2%, 14.2%]`, excludes 0). This is a MATERIAL, statistically significant gap (DEC-1).**

This corrects the Round-3 single-pair conclusion. A single matched 128-episode pair (pair 0) gave a 1-episode
gap and was reported as "decisive / no defect" — that was a lucky seed. Pooling over 5 seed pairs reveals a
consistent, significant gap.

## Per-seed results (matched env seed + injected flow-noise seed)

| pair | (env.eval.seed, eval_noise_seed) | RLinf `success_once` | reference `success_once` | gap (ref − RLinf) |
|---|---|---|---|---|
| 0 | (0, 1234) | 34/128 (0.266) | 35/128 (0.273) | +0.8% |
| 1 | (1, 1235) | 34/128 (0.266) | 39/128 (0.305) | +3.9% |
| 2 | (2, 1236) | 31/128 (0.242) | 49/128 (0.383) | +14.1% |
| 3 | (3, 1237) | 26/128 (0.203) | 44/128 (0.344) | +14.1% |
| 4 | (4, 1238) | 35/128 (0.273) | 52/128 (0.406) | +13.3% |
| **pooled** | — | **160/640 (0.250)** | **219/640 (0.342)** | **+9.2%** |

The reference model wins every seed pair; the gap is small only on pair 0. Across 640 episodes the difference
is statistically robust.

## Significance (pooled, two-proportion)

- Gap (reference − RLinf) = **0.0922**.
- Pooled two-proportion z-test: `z = 3.61`, two-sided `p < 0.001` → rejects equality.
- 95% CI of the difference (unpooled): **`[0.042, 0.142]`** — excludes 0.
- Wilson 95% CIs: RLinf ≈ `[21.8%, 28.5%]`, reference ≈ `[30.7%, 37.9%]` — disjoint.
- DEC-1 (a CI-significant gap `> 5%`): **met** — the gap estimate (9.2%) is significant and exceeds 5%.

## What this means

AC-1 proved the norm-stats are byte-identical across train/convert/eval/reference, and AC-2 proved the
converter is value-lossless (the converted model is bitwise-equal to the consolidated SFT weights). So the
significant eval gap is **not** a norm-stats or converter artifact — **it is a real difference in the trained
weights**: the RLinf SFT run produced a model that performs significantly worse on BEHAVIOR than the
reference-trained model.

This is precisely the trigger the plan reserves for the training investigation: the gap being significant
means **AC-5 (pinned first-N training-step parity under the canonical config) and AC-4 (same-batch step-1
root-cause through each repo's full normalization + forward path) are now required** to localize the
divergence. The prior Phase-3 pinned-input parity (50/50) was on identical pre-normalized inputs; the
production SFT run differs (data shuffle, the norm-stats *application* on raw actions, etc.), and one of those
surfaces must account for the ~9% behavior gap.

## The matched protocol + run records

Both checkpoints were evaluated under the deterministic protocol (`rollout.eval_deterministic_noise: True`)
with identical `(env.eval.seed, rollout.eval_noise_seed)` per pair, `num_steps = 5`, dtype bf16, and the
canonical norm-stats (`ff7e1ff0…`). Each run is summarized into a full run record parsed from its dumped
config + log, recording the complete seed/task schedule — `actor_seed` (1234), `env_eval_seed` (the env seed,
0–4, per-rank `env.eval.seed + rank*stage_num`), `flow_noise_seed` (1234–1238), `use_fixed_reset_state_ids`
(False), the task (`turning_on_radio`, `online_object_sampling: False`), `num_env_subprocess`,
`eval_rollout_epoch`, `total_num_envs` — plus the model knobs, norm-stats sha256, denormalization path, and the
config revision (the run dir + dumped config), plus the evidence-generation git revision (the generator's
`--git-rev`, not a per-run log-derived field). `all_pairs_protocol_knobs_matched: true`.

## Artifacts

- Seed-pair manifest (committed, drives the generator): `docs/evidence/phase4_eval_seed_pairs.json`.
- Evidence JSON: `docs/evidence/phase4_eval_gap_significance.json` (per-seed records + pooled CI).
- Generator (reproducible from the manifest): `tests/unit_tests/_eval_gap_significance_dump.py`.
- Production plumbing: `rlinf/workers/rollout/hf/huggingface_worker.py`
  (`deterministic_eval_seed` + `_next_eval_noise_generator` + `predict()` injection).
- Gates: `tests/unit_tests/test_openpi_pytorch_eval_gap_significance.py` (full matched-knob run records +
  pooled-CI consistency + the generator rebuilds the committed evidence from the manifest),
  `tests/unit_tests/test_openpi_pytorch_eval_plumbing.py` (seed derivation + the rollout-worker passthrough
  gate cases),
  `tests/unit_tests/test_openpi_pytorch_eval_determinism.py` (the injected-noise hook is reproducible).
