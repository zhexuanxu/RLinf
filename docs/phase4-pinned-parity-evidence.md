# Phase-4 AC-5 — pinned first-50 training-step parity (re-confirmed)

**Verdict: the pinned-input training-step parity HOLDS under the current canonical config — the real 8-GPU
FSDP `train_vla_sft.py`, replaying the reference rank-0-fanout first-50 batches + shared noise/time, reproduces
the reference model's per-step loss `50/50 within |Δ|≤0.03`, max `|Δ| = 0.002383`. This re-confirms the prior
R37 result under the current code/config.**

So the FSDP / optimizer / forward / LR-schedule stack is aligned for IDENTICAL inputs. The significant ~9% eval
gap (AC-3) is therefore **not** a training-stack regression on pinned inputs — it must be in the training
INPUTS (the raw→normalized norm-stats application, or the production data pipeline), which **AC-4 (task8)** will
root-cause. Per the plan's AC-5 negative test, a PASS means no broad FSDP/optimizer rework is warranted.

## The protocol

1. **Reference arm** (`_ref_pinned_run.py`, reference `openpi-comet` py-3.11 venv, GPU): builds the reference
   trainer's rank-0 FANOUT (pull `world_size`=8 successive micro-batches of 32 → global batch 256/step) for 50
   steps on the BEHAVIOR data, runs the real `models_pytorch_new.Pi0` with the shared fp32-master +
   autocast-bf16 AdamW+clip+warmup-LR loop on those batches + a SHARED fixed noise/time, and dumps the batches
   (`ref_pinned_batches.npz`, 50×256), the noise/time (`ref_pinned_noise_time.npz`), and the per-step
   reference loss + per-step batch/noise/time sha256 (`ref_pinned_dump.json`).
2. **RLinf arm** (the real 8×A800 FSDP `train_vla_sft.py`, `behavior_pi05_vla`, FULL_SHARD bf16-mixed-precision,
   canonical config): the config-gated pinned loader (`+data.pinned_inputs_npz` / `+data.pinned_noise_time_npz`)
   replays the IDENTICAL batches + noise/time through the production training step; per-step `train/loss` is
   logged to tensorboard.
3. **Compare** per-step `|rlinf − ref| ≤ 0.03`.

## Result

| | value |
|---|---|
| within `|Δ|≤0.03` | **50/50** |
| max `|Δ|` | **0.002383** (step 0) |
| first-5 `|Δ|` | 0.00238, 0.00184, 0.00035, 0.00096, 0.00059 |
| last-5 `|Δ|` | 0.00026, 0.00010, 0.00030, 0.00034, 0.00004 |

The per-step losses track tightly (e.g. step 0: RLinf 0.30725 vs ref 0.30487; both descend 0.30→0.08 over the
50 steps). Identical to the R37 envelope (R37 also reported 50/50, max `|Δ|=0.0024`).

## Why this matters for the eval gap

This is the AC-5 half of the training investigation the AC-3 finding triggered. It rules out a forward /
gradient / optimizer / FSDP regression: on identical pinned inputs RLinf == the reference per-step. Combined
with AC-1 (norm-stats files byte-identical) and AC-2 (converter value-lossless), the remaining surface for the
~9% behavior gap is the production training INPUTS — specifically the norm-stats *application* (raw actions →
quantile-normalized) and the data pipeline, which the pinned proofs (pre-normalized dumped batches) deliberately
bypass. **AC-4 (task8)** drives each repo's full raw→normalized→loss path on the same raw batch to localize it.

## Artifacts

- Evidence: `docs/evidence/phase4_pinned_first50_parity.json` (+ `.csv`, per-step).
- Generator (reproducible): `tests/unit_tests/_pinned_parity_dump.py` (reads the RLinf tensorboard + the
  reference dump).
- Reference dumper: `tests/unit_tests/_ref_pinned_run.py` (reference venv).
- Gate test: `tests/unit_tests/test_openpi_pytorch_pinned_parity.py` (every step within the gate; max |Δ|
  recomputes from the per-step losses; skip-gated when the evidence is absent).
- Prior Phase-3 evidence (consistent): `docs/evidence/r37_pinned_first50.{csv,json}`,
  `docs/sft-first50-step-evidence.md`.
