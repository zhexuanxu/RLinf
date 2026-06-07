# Phase-4 — Independent audit of converter + norm-stats + eval-knob matching

**Verdict: an independent analyze-route audit (Codex `gpt-5.5:high`, via `/humanize:ask-codex`) of the RLinf
openpi_pytorch BEHAVIOR SFT→convert→eval pipeline against the reference (`openpi-comet-pytorch-mixed`) found
NO correctness gap across all eight audited surfaces — every surface PASS with file references. One low-risk
hardening item was raised and applied (a duplicate-bare-key guard in the converter's prefix-strip). This is
the task10 closure: the cross-check confirms the already-verified surfaces and surfaces no missed bug.**

This audit is the independent cross-check the plan requires after the converter/eval setup (an `analyze`-route
task targeting the AC-1/AC-2/AC-3 surfaces). It reasons over the ACTUAL production paths, not lookalikes.

## Scope and method

The audit was run with a tightly-scoped prompt (per the "keep Codex scoped" lesson): the prompt carried the
plan summary, the AC-1..AC-6 verdicts, and the exact files + the eight questions; Codex read the named files
in both repos and returned a per-surface PASS/GAP verdict with `file:line` references. Audited surfaces:
converter boundaries; key/dtype handling; EMA vs non-EMA checkpoint choice; norm-stats resolution; norm-stats
application; eval-knob matching; deterministic noise/rng; action denormalization.

## Per-surface verdict (independent, with file references)

| # | Surface | Verdict | Evidence (file:line) |
|---|---------|---------|----------------------|
| 1 | Converter boundaries | **PASS** | Only prefix-strip + float→bf16 + save/copy: `utils/export_sft_checkpoint.py:51-74`, `utils/sft_to_new_pytorch.py:123-133`; hardcoded `config.json` matches `pi05_base_pytorch_new/config.json:1-10`. |
| 2 | Key / dtype handling | **PASS** | Loader enforces exact key/shape/dtype `openpi_pytorch/__init__.py:63-110`; Pi0 has no non-float buffers wrongly cast — positional values are parameters/generated, not fp32 buffers (`siglip.py:154-158`, `pi0.py:57-66`). |
| 3 | EMA vs non-EMA | **PASS** | Reference config sets `ema_decay=None` (`openpi/training/config.py:852-872`); save uses `ema_params` only when non-None (`checkpoints.py:149-155`) → the reference eval ckpt is raw final weights, matching RLinf (no EMA confound). |
| 4 | Norm-stats resolution | **PASS** | Resolver maps a set `asset_id` to exactly `{assets_dir}/{asset_id}` and rejects blank/missing (`pi0_model/normalize.py:89-119`); eval and SFT both call it (`__init__.py:236-245`, `behavior_sft_data_loader.py:391-404`) → same canonical file. |
| 5 | Norm-stats application | **PASS** | RLinf quantile (un)normalize math (`normalize.py:122-144`) matches reference `openpi/transforms.py:141-145` and `:175-181`; **neither path clips** normalized values. |
| 6 | Eval-knob matching | **PASS** | Eval reads shape/knobs from YAML not the ckpt config (`__init__.py:204-216`, `:251-290`); template sets `num_steps:5`, 32 chunk, 23 env dim, 200 max tokens (`model/pi0_5_pytorch.yaml:15-41`); reference has no hidden temperature/CFG path, only `sample_kwargs`→`sample_actions` (`openpi/policies/policy.py:81-94`). |
| 7 | Deterministic noise/rng | **PASS** | Worker injects only a seeded generator (`huggingface_worker.py:371-432`); sampler draws randomness ONLY for the initial noise when `noise is None` (`pi0.py:398-401`), then the Euler loop is deterministic (`:414-452`) → a fixed-noise paired comparison truly isolates weights. |
| 8 | Action denormalization | **PASS** | RLinf unnormalizes the full model action THEN slices via `BehaviorOutputs` (`processing.py:154-162`, `behavior_policy.py:135-142`); reference order is outputs→`Unnormalize`→`B1kOutputs` (`openpi/policies/policy_config.py:84-88`, `b1k_policy.py:168-174`) — same order + same 32→23 mapping. |

## Other finding (hardening) — applied

> **OTHER (low-risk hardening):** `_strip_wrapper_prefix` would silently OVERWRITE if two checkpoint keys
> normalized to the same bare key (`export_sft_checkpoint.py:61-74`); add a duplicate-key assertion — though
> this is not observed for the Pi0 checkpoints audited.

**Applied.** `_strip_wrapper_prefix` now raises `ValueError` if two distinct source keys collapse to the same
bare key (refusing to silently drop a tensor), instead of overwriting. This never fires for the real BEHAVIOR
checkpoints (the converter value-parity test, which runs the strip on the actual checkpoint, still passes), so
it is a pure safety guard — it does not change the converted output for valid inputs. Re-verified:
`test_openpi_pytorch_converter_parity.py` passes.

## Closure

task10 is closed: **no correctness gap** was found across converter boundaries, key/dtype, EMA/non-EMA,
norm-stats resolution + application, eval-knob matching, deterministic noise/rng, or denormalization — the
independent cross-check confirms AC-1..AC-6. The single hardening item raised was applied and re-verified.

## Artifacts

- Raw audit transcript: `.humanize/skill/<ts>/output.md` (Codex `gpt-5.5:high`, exit 0); the eight
  `SURFACE n: PASS` verdicts + `OTHER` are reproduced above.
- Hardening fix: `rlinf/models/embodiment/openpi_pytorch/utils/export_sft_checkpoint.py` (duplicate-bare-key
  guard in `_strip_wrapper_prefix`).
- Re-verification: `tests/unit_tests/test_openpi_pytorch_converter_parity.py` (4 passed).
