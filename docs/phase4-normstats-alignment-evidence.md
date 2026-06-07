# Phase-4 AC-1 — BEHAVIOR norm-stats alignment evidence

**Verdict: all four norm-stats touchpoints of the current SFT-train → convert → eval pipeline resolve the SAME
canonical file (sha256 `ff7e1ff0…`). Norm-stats are NOT the cause of the `0.2578`-vs-`0.3203` eval gap.**

The canonical source is the reference's own resolution: the reference `TrainConfig`
`pi05_b1k-task0000_sft_pytorch_mixed` sets no explicit `asset_dir`, so the default
`assets_base_dir = "./outputs/assets/train"` → `assets_dirs/{name}` → `_load_norm_stats({assets_dirs}/
{asset_id})` with `asset_id = repo_id = behavior-1k/2025-challenge-demos` resolves to
`outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json`
= the **NEW** file (`ff7e1ff0…`, 6346 B).

## Touchpoint hashes (gate = resolved file content, not config string)

| Touchpoint | Source | Resolved `norm_stats.json` | sha256 |
|---|---|---|---|
| (a) SFT **training** data normalization | the ACTUAL run's dumped config + run log | `…/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/` | `ff7e1ff0…` |
| (b) converter copied output | `sft_to_new_pytorch.py` | `…/pi05_sft_pytorch_new/physical-intelligence/behavior/` | `ff7e1ff0…` |
| (c) RLinf **eval** model-load | eval YAML `assets_dir`+`asset_id` via `resolve_norm_stats_dir` | `…/pi05_sft_pytorch_new/physical-intelligence/behavior/` | `ff7e1ff0…` |
| (d) **reference** train/eval path | reference `TrainConfig` default `assets_base_dir` | `…/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/` | `ff7e1ff0…` |
| reference checkpoint's own (eval2 control) | `jax_task0000_sft_29999_ptnew/physical-intelligence/behavior/` | same | `ff7e1ff0…` |
| OLD pre-switch (prior Phase-3 only — NOT a current touchpoint) | `pi05-b1kpt50-cs32/assets/behavior-1k/2025-challenge-demos/` | — | `d66ed168…` (6368 B) |

`resolve_norm_stats_dir(assets_dir, asset_id)` (`rlinf/models/embodiment/openpi_pytorch/pi0_model/
normalize.py:89`) is the SINGLE shared resolver used by BOTH the SFT data loader
(`behavior_sft_data_loader.py:184`) and the eval model factory (`__init__.py:244`), so train-load and
eval-load resolve `{assets_dir}/{asset_id}/norm_stats.json` identically.

## The actual SFT run trained on the NEW (canonical) file — no train/eval mismatch

The eval'd checkpoint comes from `logs/20260605-12:39:44-behavior_pi05_vla`. That run's evidence (read, not
assumed — BL-`ac-exact-gate-not-proxy`, BL-`verify-actual-import-path`):

- Dumped `tensorboard/config.yaml` (lines 20-21):
  `assets_dir: …/pi05_b1k-task0000_sft_pytorch_mixed`, `asset_id: behavior-1k/2025-challenge-demos`.
- `run_embodiment.log` (lines 272-279): all **8** FSDP workers log
  `Loaded BEHAVIOR norm stats from …/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos`.

Both point at the NEW canonical file. So the checkpoint was **trained on the same norm-stats eval uses** —
there is no train-on-OLD / eval-on-NEW normalization mismatch. (At training time the model factory logs
`norm_stats_digest=deferred` because training defers norm-stats to the data loader, which is what logs the
resolved path above; this is by design, `__init__.py:234-236`.)

## OLD vs NEW: the switch was material (so prior Phase-3 work on OLD was genuinely misaligned)

The OLD (`d66ed168…`) and NEW (`ff7e1ff0…`) files differ substantially (per-key, dim 32):

| Key | field | max abs diff | mean abs diff |
|---|---|---|---|
| actions | q01 | 1.414 | 0.254 |
| actions | q99 | 1.037 | 0.124 |
| actions | mean | 0.336 | 0.061 |
| actions | std | 0.410 | 0.074 |
| state | q01 | 1.408 | 0.254 |
| state | q99 | 1.033 | 0.116 |
| state | mean | 0.334 | 0.056 |
| state | std | 0.410 | 0.076 |

Since BEHAVIOR uses quantile normalization keyed on `q01`/`q99`, a >1.0 max shift in those quantiles is a
real change to the normalized inputs/targets. This quantifies why the user's `assets_dir` switch mattered —
and confirms that any *prior* Phase-3 run on the OLD file was on a different normalization. The current run is
not affected.

## Self-verifying derivation (each touchpoint is read from its driving source)

The generator and test do not hardcode the touchpoint answers — each is **derived from the source that drove
it**, so the gate fails if any source drifts:

- `sft_train` — parsed from the run's dumped `tensorboard/config.yaml` (`actor.model.openpi.assets_dir` /
  `asset_id`) and cross-checked against `run_embodiment.log`: all 8 FSDP workers must log the same resolved
  directory (`runlog_matches_resolved: true`, `runlog_worker_loads: 8`).
- `rlinf_eval` — parsed from `behavior_ppo_openpi_pi05_pytorch_eval.yaml` (`actor.model.openpi.assets_dir` /
  `asset_id`).
- `reference` — resolved by executing the reference repo's **own** `get_config(
  "pi05_b1k-task0000_sft_pytorch_mixed").assets_dirs` + `data.repo_id` in the reference venv, so the canonical
  source is the reference's code, not a copied literal.
- `converter_out` — the copied file, additionally asserted equal to its input (`equals_reference_input:
  true`).

## Artifacts

- Evidence JSON: `docs/evidence/phase4_normstats_alignment.json` (all hashes, resolved paths, run proof,
  OLD-vs-NEW quantile diff, per-touchpoint provenance).
- Generator (reproducible, source-deriving): `tests/unit_tests/_normstats_alignment_dump.py`.
- Exact-gate test: `tests/unit_tests/test_openpi_pytorch_normstats_alignment.py` (4 tests; derive each
  touchpoint from its source, resolve through the production `resolve_norm_stats_dir`, and assert byte-identity
  + the run-log worker cross-check; skip-gated per source when the external assets / reference venv are
  absent).

## Consequence for the eval-gap investigation

AC-1 is closed with a **null result on the suspect axis**: norm-stats are already aligned end-to-end, so the
`0.2578`-vs-`0.3203` gap is not explained by a norm-stats misresolution. Suspicion moves to the converter
value-parity (AC-2) and the eval-gap statistical significance (AC-3, the gap is ~1.5 SE on an unpaired
128-episode eval). The OLD/NEW/reference norm-stats matrix collected here feeds AC-4's same-batch step-1
root-cause.
