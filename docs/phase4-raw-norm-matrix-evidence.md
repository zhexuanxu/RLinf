# Phase-4 AC-4 (CORE) — same-batch raw→normalized→loss + norm-stats matrix

**Verdict: on the IDENTICAL raw sampled batch (the first 8 sequential frames), RLinf's FULL
raw→quantile-normalized→loss path (canonical NEW stats) reproduces the reference — raw hashes match,
tokens/masks byte-identical, normalized state/actions within `2.9e-8`, same-batch loss `0.4038` vs reference
`0.4029` (`|Δ| = 0.0009 ≤ 0.01`). The per-batch normalization APPLICATION is correct on these frames. So this
test RULES OUT a same-raw-batch transform / norm-stats-application bug as the cause of the significant ~9%
eval gap (AC-3). It does NOT, by itself, prove that run-to-run training-trajectory variance is the COMPLETE
explanation of the gap — a single sampled batch cannot exercise the full 30k-step production trajectory, and
this artifact does not record task/sample-id/global-batch provenance. It removes the sampled same-batch
normalization surface from suspicion; the behavior-level magnitude of the trained-weights difference is
quantified separately (AC-6).**

This is the surface the pinned proofs (AC-5) deliberately bypass: AC-5 replays PRE-normalized batches; AC-4
drives the RAW actions through each repo's OWN `normalize_quantile` → loss.

## Protocol (not pre-normalized)

1. **Reference arm** (`_ref_raw_norm_dump.py`, reference py-3.11 venv, GPU): for the first 8 frames, dump the
   RAW LeRobot fields BEFORE any transform (`transformed._dataset[i]`: raw head/left/right images, raw state,
   raw action chunk, task) + a sha256 over each raw field; dump the reference per-sample transform
   (`._transform(frame)` under the canonical NEW stats: normalized+padded state/actions, tokenized prompt+mask,
   resized images); and compute the reference `Pi0.compute_loss(train=True, rng=None, noise, time)` with numpy
   noise/time.
2. **RLinf arm** (`_raw_norm_matrix_dump.py`, GPU): feed the IDENTICAL raw frames through the production
   `BehaviorSftTransform` (`_repack`→`BehaviorInputs`→resize→`normalize_quantile`→tokenize→pad — NOT the pinned
   loader); re-hash the raw fields (must equal the reference); compare normalized state/actions/tokens/images;
   build the RLinf `Pi0` and compute the loss on its OWN transform output with the SAME noise/time.
3. **Norm-stats matrix:** transform the same raw batch under OLD vs NEW(=canonical=reference-resolved) stats and
   attribute the normalized/loss delta to the stat file.

## Same-batch comparison (canonical NEW stats)

| quantity | value |
|---|---|
| raw batch hashes match reference | **true** (same raw input, not pre-normalized) |
| tokens / masks byte-identical | **true** |
| normalized state max |Δ| vs reference | 2.94e-8 |
| normalized actions max |Δ| vs reference | 2.96e-8 |
| base image max |Δ| vs reference | 0.0 |
| same-batch loss (RLinf) vs reference | **0.4038 vs 0.4029** (|Δ| 0.0009 ≤ 0.01) |

## Norm-stats matrix (the application IS load-bearing)

| stats | sha256 | RLinf same-batch loss | |Δ| vs reference |
|---|---|---|---|
| NEW (canonical = reference-resolved) | `ff7e1ff0…` | **0.4038** | **0.0009** |
| OLD (pre-switch) | `d66ed168…` | 0.3488 | 0.0541 |

The OLD stat file changes the normalized actions and the loss by **5.4%** on the SAME raw batch — so the
matrix is sensitive and the test is meaningful. The production SFT run used the NEW file (AC-1), which is the
one that matches the reference.

## Why the ~9% eval gap is not a same-input surface

Every same-input surface is now proven aligned:
- **AC-1**: the norm-stats files are byte-identical across train/convert/eval/reference.
- **AC-2**: the converter is value-lossless (bitwise).
- **AC-5**: on identical PRE-normalized inputs, the FSDP/optimizer/forward stack reproduces the reference
  per-step (50/50, max 0.0024).
- **AC-4 (this)**: on the identical RAW batch, the FULL raw→normalized→loss path reproduces the reference
  (|Δ| 0.0009), and the production effective batch was the correct 256 (`global_batch_size: 256`,
  `micro_batch_size: 32` — the Phase-3 R34 spawn-worker fix was active, not the buggy 32).

So no per-sample or per-batch code path on the SAMPLED batch explains the significant ~9% eval gap between
the single RLinf-trained and single reference-trained checkpoint. A confirmed same-input MATCH on these frames
REMOVES the sampled same-batch normalization/loss surface from suspicion; it is consistent with the gap being
the different training TRAJECTORY of two INDEPENDENT 30000-step SFT runs (data shuffle/coverage/RNG, the
plan's "benign independent shuffle"), but a single sampled batch does NOT by itself prove trajectory variance
is the COMPLETE cause. Establishing whether the trajectory difference is a SYSTEMATIC RLinf disadvantage or
run-to-run training variance would require multiple independent RLinf-vs-reference training runs (as Phase-3's
R24 did for the loss curve), which is beyond this same-batch root-cause scope. The behavior-level magnitude of
the trained-weights difference is quantified on fixed real eval observations separately (AC-6). No localized
same-batch bug exists to fix.

## Artifacts

- Evidence: `docs/evidence/phase4_raw_norm_forward_matrix.{json,csv}`.
- Reference dumper: `tests/unit_tests/_ref_raw_norm_dump.py` (reference venv).
- RLinf comparison + matrix generator: `tests/unit_tests/_raw_norm_matrix_dump.py`.
- Gate test: `tests/unit_tests/test_openpi_pytorch_raw_norm_matrix.py` (raw-batch identity; canonical
  same-batch match; matrix present + sensitive; skip-gated when the evidence is absent).
