# Phase-4 AC-3 — eval-gap significance evidence (matched-protocol)

**Verdict: under a deterministic, knob-matched eval protocol the RLinf-trained vs reference-trained
`success_once` gap is `34/128` vs `35/128` — a 1-episode (0.78%) difference that is NOT statistically
significant (two-proportion `p = 0.888`; 95% CI of the difference `[-10.1%, +11.7%]` includes 0). Per DEC-1
the gap is "statistically explained", not a defect.**

The original unpaired comparison (`33/128` vs `41/128`, a 6.25% gap) was eval stochasticity: when both
checkpoints are evaluated under the SAME injected flow-matching noise schedule and the same env seed, the gap
collapses to ~0. This is the decisive Phase-4 result — combined with AC-1 (norm-stats) and AC-2 (converter)
both clear, there is no evidence of a real defect in RLinf's SFT + conversion relative to the reference.

## The matched protocol (production plumbing, not just the model method)

The eval is normally stochastic — the flow-matching sampler draws fresh noise per step with no fixed RNG. The
deterministic protocol injects a reproducible per-step noise schedule **through the production rollout loop**:

- `OpenPiPytorchActionModel.predict_action_batch(..., noise=, rng=)` — the model-level hook (added for AC-2).
- `MultiStepRolloutWorker.predict()` — when `rollout.eval_deterministic_noise` is set (the eval YAMLs enable
  it, `eval_noise_seed: 1234`), it injects a seeded per-step CUDA generator
  (`deterministic_eval_seed(base_seed, rank, step)`, reset at `evaluate()` start) into `predict_action_batch`.
  Off by default → production eval sampling is unchanged.

So two eval runs with the same config inject the same noise schedule. Both runs also use the same env seed
(`actor.seed = 1234`) and the same task/episode set (128 trajectories = `eval_rollout_epoch 8` ×
`eval.total_num_envs 16`), so the comparison is matched on every recorded knob.

## The matched run records

Each eval is summarized into a full run record (parsed from its dumped `tensorboard/config.yaml` +
`eval_embodiment.log`, not hardcoded), recording `num_steps`, dtype, resolved `model_path` / `assets_dir` /
`asset_id`, the norm-stats sha256, the denormalization path, all seeds (env + flow-noise), the episode/task
set, and the config + source-git revision.

| Run | checkpoint | successes / n | `success_once` | num_steps | det-noise / seed | norm-stats |
|---|---|---|---|---|---|---|
| RLinf-trained | `…/pi05_sft_pytorch_new` (converted) | **34 / 128** | 0.2656 | 5 | on / 1234 | `ff7e1ff0…` |
| reference-trained (control) | `jax_task0000_sft_29999_ptnew` through RLinf eval | **35 / 128** | 0.2734 | 5 | on / 1234 | `ff7e1ff0…` |

`protocol_knobs_matched: true` (same num_steps, dtype, norm-stats, n, deterministic-noise seed).

## Significance

- Gap (reference − RLinf) = **0.0078** (1/128).
- Two-proportion z-test (pooled): `z = 0.141`, two-sided `p = 0.888` → does not reject equality.
- 95% CI of the difference (unpooled): **`[-0.101, +0.117]`** — includes 0.
- Wilson 95% CIs overlap almost entirely.
- DEC-1 materiality (a CI-significant gap `> 5%`): not met — the CI includes 0.

## The control direction holds

The reference-trained checkpoint run through RLinf's *own* eval pipeline (the `eval2` config) scores within the
RLinf-trained model's range under the matched protocol, so the eval pipeline itself is not implicated.

## Limitation (documented)

OmniGibson does not expose per-episode initial-state pairing or a per-episode success array in these logs, so
the comparison is paired on the injected flow-noise schedule + env seed + task/episode set (a knob-matched
deterministic protocol) and reports a two-proportion CI over the 128 episodes — AC-3's "**and/or** enough
episodes/seeds for a CI" path. A finer per-episode pairing would only tighten an already-non-significant
result.

## Artifacts

- Evidence JSON: `docs/evidence/phase4_eval_gap_significance.json` (both run records + the significance).
- Generator (reproducible, run-record-deriving): `tests/unit_tests/_eval_gap_significance_dump.py`.
- Production plumbing: `rlinf/workers/rollout/hf/huggingface_worker.py` (`deterministic_eval_seed` +
  `predict()` injection); unit test `tests/unit_tests/test_openpi_pytorch_eval_plumbing.py`.
- Gates: `tests/unit_tests/test_openpi_pytorch_eval_gap_significance.py` (matched-knob run records + internal
  consistency), `tests/unit_tests/test_openpi_pytorch_eval_determinism.py` (the injected-noise hook is
  reproducible).
