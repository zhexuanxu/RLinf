# Phase-4 AC-6 — model divergence quantified by BEHAVIOR, not raw weights

**Verdict: on a fixed batch of REAL BEHAVIOR eval observations + fixed injected flow-matching noise, with
every eval knob matched (num_steps=5, bf16, canonical `ff7e1ff0…` norm-stats for BOTH input-normalization
and output-denormalization, so only the trained WEIGHTS differ), the RLinf-trained-and-converted model and
the reference-trained model produce measurably different action chunks: post-denormalization |Δ| mean
`0.0088`, p95 `0.024`, max `0.306` — `3.9%` of the reference action magnitude (`0.225`). Paired
determinism holds exactly (same model + same noise ⇒ identical actions, max|Δ| `0.0`). The divergence is
reported as post-denormalization action-chunk deltas (the behavior signal), NOT raw per-parameter tensor
deltas. The behavior-level divergence is real and non-trivial, consistent with the significant +9.2%
reference-minus-RLinf eval-success gap (AC-3).**

This closes the last new surface: AC-1 (norm-stats), AC-2 (converter), AC-3 (eval-gap significance),
AC-4 (same-batch raw→loss), AC-5 (pinned parity) are all met; AC-6 gives the eval gap a behavior-level
magnitude on the model side.

## Protocol (real eval obs, not synthetic)

1. **Real eval observation capture** (`_behavior_eval_obs_dump.py`): construct the SAME `BehaviorEnv` that
   `get_env_cls` returns for the BEHAVIOR `use_skill:false` eval path (OmniGibson-backed,
   `env/behavior_r1pro`, task `turning_on_radio`), `reset()` once, and dump the real `env_obs`
   (`main_images` / `wrist_images` / `states` / `task_descriptions`) + a sha256 over each field + the env
   knobs (seed, num_envs, task). These REAL observations are the fixed set — recorded by content hash, not
   synthetic random tensors.
2. **Two-model behavior comparison** (`_behavior_divergence_dump.py`): load BOTH checkpoints through the
   identical production `get_model` factory, each pinned to the canonical `ff7e1ff0…` norm-stats (so input
   normalization AND output denormalization are held constant; only the trained WEIGHTS differ). Feed the
   IDENTICAL real `env_obs` + the SAME fixed flow-matching noise (seed 1234) + seeded rng at num_steps=5 /
   bf16 through each model's production `predict_action_batch`. Record the denormalized action chunk
   (`actions` [B, action_chunk, action_env_dim]) and the normalized model action
   (`forward_inputs.model_action`).
3. **Deltas**: post-denormalization action-chunk |Δ| (mean / max / p95 overall, per action-dim, per chunk
   position) + normalized model-action |Δ|; paired-determinism self-check (same model + same noise ⇒
   identical actions). Interpret the magnitude against AC-3's significant +9.2% reference-minus-RLinf gap.

## Results

From `docs/evidence/phase4_behavior_divergence.json` (generated under git `ae9e774a`).

| quantity | value |
|---|---|
| real eval obs (env-sourced, hashed) | **true** (`main_images`/`wrist_images`/`states`/`task_descriptions` each sha256'd) |
| num envs / task | 4 / `turning_on_radio` (env_seed 0) |
| RLinf checkpoint `model.safetensors` / `config.json` sha | `9a31478e…` / `a5110e38…` |
| reference checkpoint `model.safetensors` / `config.json` sha | `aa151960…` / `a4ae2082…` |
| norm-stats sha (both models) | `ff7e1ff0…` (held constant; rlinf == reference == canonical) |
| num_steps / dtype | 5 / bfloat16 |
| fixed noise (seed / sha) | 1234 / `52728f95…` (shared by both models) |
| paired determinism max\|Δ\| | **0.0** (same model + same noise ⇒ identical actions) |
| reference action mean abs (scale) | 0.2252 |
| denorm action-chunk \|Δ\| mean / p95 / max | **0.00879 / 0.02445 / 0.30570** |
| denorm \|Δ\| relative to action scale | **0.0390** (3.9%) |
| normalized model-action \|Δ\| mean / p95 / max | 0.01123 / 0.03125 / 0.89453 |
| per action-dim mean \|Δ\| | 23 dims; max-dim 0.0475, min-dim 0.0 (divergence concentrates in a subset of action dims) |
| per chunk-position mean \|Δ\| | 32 positions recorded |

The two independently-trained policies, given the IDENTICAL real observations + IDENTICAL fixed noise and
all knobs matched, disagree on the action chunk by ~3.9% of the action magnitude on average (p95 2.4%, with
a worst-case single-element 0.31). That is a real behavioral difference — not a gross divergence, but large
enough to plausibly drive the significant +9.2% eval-success gap. Because the input normalization, the
denormalization stats, the noise, num_steps, and dtype are all held constant, the difference is attributable
solely to the trained weights, expressed as BEHAVIOR.

## Why this is the AC-6 signal (not raw weights)

The negative test is explicit: raw per-parameter tensor deltas between two independently-shuffled
production SFT runs are **not** an interpretable divergence-bug signal (known RNG/shuffle divergence). AC-6
measures BEHAVIOR — action chunks on fixed real observations + fixed noise — which is what actually drives
the eval success gap. The evidence records `divergence_signal = action_chunk_delta_post_denormalization`
and `raw_parameter_delta_rejected = true`; the gate fails if the signal is reported as raw parameter
deltas, if the observations are synthetic, if the two model loads used mismatched knobs, or if paired
determinism is violated.

## Artifacts

- Evidence: `docs/evidence/phase4_behavior_divergence.{json,csv}`.
- Real-obs dumper: `tests/unit_tests/_behavior_eval_obs_dump.py` (standalone `BehaviorEnv`, embodied venv).
- Two-model comparison generator: `tests/unit_tests/_behavior_divergence_dump.py`.
- Gate test: `tests/unit_tests/test_openpi_pytorch_behavior_divergence.py` (real-obs-not-synthetic;
  knobs-matched; paired-determinism; behavior-signal-not-raw-weights; divergence measured + tied to AC-3).
