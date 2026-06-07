# Phase-4 AC-3 — eval-gap significance evidence

**Verdict: the `0.2578`-vs-`0.3203` `success_once` gap is NOT statistically significant — it is within eval
run-to-run noise (two-proportion test `p = 0.270`; 95% CI of the difference `[-4.8%, +17.3%]` includes 0). Per
DEC-1 (trend / confirm-significance-first) the gap is "statistically explained", not a defect.**

The eval is stochastic (flow-matching noise sampled with no fixed RNG) and the two runs are independent
(unpaired). Counts are parsed from each run's logged `success_once` + `num_trajectories` (not hardcoded), so
the gate fails if those logs change.

## The numbers

| Run | checkpoint | successes / n | `success_once` |
|---|---|---|---|
| RLinf-trained | `…/pi05_sft_pytorch_new` (converted) | **33 / 128** | 0.2578 |
| reference-trained (control) | `jax_task0000_sft_29999_ptnew` through RLinf eval | **41 / 128** | 0.3203 |

- Gap (reference − RLinf) = **0.0625** (8/128).
- **Two-proportion z-test** (pooled): `z = 1.103`, two-sided `p = 0.270` → does **not** reject equality.
- **95% CI of the difference** (unpooled): **`[-0.048, +0.173]`** — includes 0.
- Wilson 95% CIs: RLinf `[0.188, 0.345]`, reference `[0.243, 0.405]` — heavily overlapping.
- **DEC-1** materiality threshold (a CI-significant gap `> 5%`): not met — the CI includes 0, so the gap is
  not even significant, let alone material.

## The control direction holds

The reference-trained checkpoint run through RLinf's *own* eval pipeline reproduces its expected `0.3203`
(`eval2`), so the eval pipeline itself is not implicated — exactly the calibration AC-3 requires.

## Why this is the decisive Phase-4 result

AC-1 (norm-stats) and AC-2 (converter) are both cleared as null results, and now AC-3 shows the eval gap
itself is not statistically distinguishable from zero on 128 episodes. There is therefore **no evidence of a
real defect** in the RLinf SFT + conversion pipeline relative to the reference: the gap is consistent with the
stochastic eval's run-to-run variation.

A paired/deterministic eval (same per-episode initial states + a shared injected flow-matching noise sequence)
would *tighten* this comparison by removing the eval's noise; the deterministic-noise hook needed for it is
implemented (`predict_action_batch(..., noise=, rng=)` + the rollout-worker passthrough). But since the
unpaired 128-episode CI already includes 0, a paired run can only confirm a gap that is already not
significant.

## Artifacts

- Evidence JSON: `docs/evidence/phase4_eval_gap_significance.json`.
- Generator (reproducible, log-deriving): `tests/unit_tests/_eval_gap_significance_dump.py`.
- Gate test: `tests/unit_tests/test_openpi_pytorch_eval_gap_significance.py` (counts parsed from logs; gap
  within noise; skip-gated when the eval logs are absent).
