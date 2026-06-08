# Phase-6 — Root-Cause the First-Step Loss / Grad-Norm Gap and Prove Strict Dataloader + Optimizer Alignment

## Goal Description

The RLinf BEHAVIOR pi0.5 8-GPU FSDP SFT training run logs a first-step loss of ~0.35 and a larger grad-norm,
while the reference trainer (`openpi-comet-pytorch-mixed`, config `pi05_b1k-task0000_sft_pytorch_mixed`) logs
step-0 `loss=0.24609375, grad_norm=2.1718, lr=2.4975e-08`. Over the curve RLinf trends lower. The user requires
a fresh, deeper investigation that does NOT assume the prior conclusion ("benign RNG/shuffle"), built on three
pillars the user stated explicitly:

1. **Make the two dataloaders provably consistent** — under 8 GPUs, the same batch size, and the same dataset,
   the data both repos see must be the same at every step (at the **global-batch** level). The user's decision:
   the correct bar is **identical global-batch multiset per step** (each global step's full 256-frame set is
   identical across repos; which rank receives which frame may differ harmlessly).
2. **Given identical step-0 data, require compute parity** — once the global step-0 batch is proven identical,
   the first-step loss/grad-norm must match within tolerance. The user's stated reasoning: *a gap this large on
   identical data would be a compute-side problem* — so if a material residual remains on identical data, this
   plan must localize and fix the compute bug.
3. **Prove the lr + optimizer update logic is strictly identical** — per-step learning-rate schedule and AdamW
   parameter updates of the two repos must be strictly identical (verifiable with a fake model + fake data).

Because the two referenced runs (`logs/20260605-12:39:44-behavior_pi05_vla`,
`logs/20260608-04:14:29-behavior_pi05_vla`) recorded no git commit, it is unverified whether they ran the
already-committed rank/world-size streaming-loader fix. The user's decision: run a fresh **short, instrumented**
8-GPU SFT job on the current fixed-loader code, proving at runtime that the fix path executed (effective batch
256, rank-disjoint), to obtain trustworthy step-0 numbers before drawing any conclusion.

This effort BUILDS ON (and must not redo) the committed prior results: forward-loss parity on identical
batch+weights+noise/time is already proven (same-batch max |Δ| 0.0024); the mixed-precision recipe and fp32
grad-norm are already proven effectively identical on identical weights; and the effective-batch root cause
(RLinf rank-replicated batch 32 vs reference 256) was already found and fixed by threading explicit
`dist_rank`/`dist_world_size` into the streaming dataset. The NEW questions are: (a) is the loader **strictly**
multiset-consistent at every one of the 30k steps, (b) on a confirmed-fixed run is the first-step gap real or an
artifact of the un-provenanced old runs, (c) if a residual remains on identical data, where is the compute bug,
and (d) is the lr/optimizer update strictly identical.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.
"Strictly identical" means exact for discrete quantities (frame-id multisets, lr sequence, integer step
indices) and within an explicit numeric tolerance for floating-point quantities (stated per criterion).
"Step-0 loss" is defined precisely as: the per-micro-batch loss reduced to a single global scalar the SAME way
each trainer logs it (global mean over the effective batch), recorded BEFORE `optimizer.step()` (so it is a
pure forward on the initial loaded weights); "step-0 grad_norm" is the pre-clip, fp32-accumulated, all-reduced
global gradient norm. Both repos' logging semantics for these two scalars must be matched exactly before any
numeric comparison.

- **AC-1: Trustworthy step-0 provenance from a confirmed-fixed, instrumented run.** A short 8-GPU SFT run on the
  current code emits a committed provenance + step-0 artifact proving the rank/world-size loader fix was
  EXERCISED AT RUNTIME (not merely present in source), and capturing the real step-0 loss/grad-norm/lr under the
  production stack.
  - Positive Tests (expected to PASS):
    - A committed artifact records, for the fresh run: git commit + dirty-diff status, the exact launch command,
      the fully-resolved Hydra config, torch/ray/openpi package versions, base-checkpoint sha, norm-stats sha,
      dataset path, seed, world_size=8, num_workers, micro/global batch, and grad-accumulation.
    - A runtime instrumentation record proves the streaming partition ran rank-disjoint at effective batch 256
      (each rank's per-step frame-id set is disjoint; the union per global step has exactly 256 distinct frames),
      i.e. the fix path executed — NOT the rank-replicated 32 path.
    - The artifact records the run's step-0 `loss`, `grad_norm` (pre-clip fp32 global), and `lr` with the logging
      semantics defined above, and the step-0 `lr` equals the reference warmup init 2.4975e-08 within 1e-12.
  - Negative Tests (expected to FAIL/be rejected):
    - An artifact whose runtime instrumentation shows rank-replicated batches (per-step union < 256 distinct
      frames, e.g. 32) is flagged as the fix NOT exercised — the run is not usable as the trusted baseline.
    - A provenance artifact missing any required field (commit/dirty status, resolved config, checkpoint sha,
      norm-stats sha, package versions) is rejected.
    - A forged/synthetic step-0 artifact not produced by the instrumented run path is rejected.

- **AC-2: The two dataloaders produce an identical global-batch multiset at every step across the full 30k-step
  schedule.** A rolling per-global-step frame-identity audit over all 30k steps proves multiset equivalence
  under the production 8-GPU topology (reference rank-0 fanout vs RLinf per-rank streaming), with the same
  dataset, seed, and global batch size.
  - Positive Tests (expected to PASS):
    - For each global step `s` in `[0, 30000)`, the multiset (or set, since frames are distinct within a step)
      of frame identifiers fed across all 8 ranks is identical between the two repos; this is verified by a
      rolling cryptographic hash of the per-step sorted frame-id list, compared step-by-step, and the two rolling
      hash streams match for all 30k steps.
    - Both repos report exactly 256 distinct frame identifiers per global step (rank-disjoint, no rank-0
      replication), recorded alongside the hash stream.
    - The audit records BOTH a per-rank frame-id hash AND the global per-step set hash, so a failure
      distinguishes a rank-assignment difference (per-rank hashes differ, global set matches — acceptable under
      the chosen bar) from a true sample-set divergence (global set hashes differ — a real loader misalignment).
    - A re-run of the capture produces byte-identical hash streams (instrumentation does not perturb worker
      RNG / sampler / ordering).
  - Negative Tests (expected to FAIL/be rejected):
    - If at ANY global step the global frame-id set differs between the repos, the audit FAILS and the first
      diverging step + the differing frame ids are reported (this is a real dataloader misalignment to fix, per
      the user's requirement that the loaders be strictly consistent).
    - A loader configured to read `torch.distributed` inside spawned workers (the pre-fix path) is detected as
      rank-replicated (global set < 256 distinct) and fails.
    - An audit that silently samples a subset of the 30k steps without recording exactly which steps were
      covered is rejected (the chosen scope is FULL 30k rolling-hash; any reduction must be explicitly logged
      and is a failure of this AC).

- **AC-3: On an identical global step-0 batch, the first-step loss and grad-norm match within tolerance; any
  material residual is localized to a concrete compute cause.** A 2×2 cross-feed plus an identical-data
  production-stack comparison classifies the first-step gap as data-side or compute-side, and — given identical
  data — drives compute parity or a concrete bug.
  - Positive Tests (expected to PASS):
    - 2×2 cross-feed: each repo's ACTUAL step-0 batch (fully materialized — images post crop/resize, normalized
      state/actions, masks, padding, tokenized prompt, frame ids) is fed through BOTH repos' models with
      identical loaded weights, identical norm-stats, identical sampled flow `noise`/`time`, identical dtype /
      autocast / train-mode; for each of the two batches, the forward loss matches across the two models within
      the committed band (bf16: per-sample |Δ| ≤ rel 5% or abs 0.01; fp32 path: tighter |Δ| ≤ 1e-3), classifying
      the gap as DATA-side or COMPUTE-side.
    - Identical-data step-0 parity: with AC-2 establishing an identical global step-0 batch, the two production
      stacks' step-0 `loss` and pre-clip fp32 global `grad_norm` match within tolerance (loss abs ≤ 0.01 bf16;
      grad_norm rel ≤ 2%), confirming no compute-side defect on identical data.
    - If a residual beyond tolerance remains on identical data, it is LOCALIZED to a concrete cause with
      evidence (per-module grad-norm breakdown, loss denominator/mask, reduction op mean-vs-sum, dtype/precision
      at the divergent surface) and a fix is applied that brings the identical-data comparison within tolerance.
  - Negative Tests (expected to FAIL/be rejected):
    - A cross-feed that does NOT hold weights / norm-stats / noise / time / dtype / train-mode identical across
      the two models is rejected (it cannot isolate data vs compute).
    - A "compute-side" claim asserted without the matched 2×2 cells (or a "data-side" claim asserted without
      showing the same batch yields the same loss on both models) is rejected.
    - A residual-gap "fix" that changes the loss on the reference-matched identical batch but is not justified by
      a localized divergence surface is rejected.

- **AC-4: The lr schedule and AdamW optimizer update are strictly identical between the two repos.** A
  standalone deterministic probe (fake model with representative parameter groups, fake fixed gradients) shows
  identical per-step lr and identical post-update parameter deltas over the warmup+cosine schedule. (Purpose:
  this governs strict trajectory/lr identity and post-update state; it is NOT required to explain the pre-update
  step-0 loss, which is logged before the optimizer step.)
  - Positive Tests (expected to PASS):
    - The per-step lr sequence is identical for a representative span covering warmup and into cosine decay
      (e.g. steps 0, 1, 999, 1000, 1001, and sampled decay steps): step-0 lr = 2.4975e-08, step-1000 lr = peak
      2.5e-5, identical to within 1e-12 (the schedule is deterministic), with matching step-indexing semantics
      (lr logged before vs after the scheduler step is matched).
    - With identical fake gradients fed to both update logics from identical fake initial weights, the AdamW
      parameter deltas (and `exp_avg`/`exp_avg_sq` states) match within fp32 tolerance (rel ≤ 1e-5) across the
      tested steps, using matching betas (0.9, 0.95), eps (1e-8), weight_decay (1e-10), and parameter-group
      construction.
  - Negative Tests (expected to FAIL/be rejected):
    - A mismatch in any of: warmup formula (`peak_lr/(warmup+1)` init), step indexing, betas/eps/weight_decay,
      grad-accumulation semantics, or lr-logged-before-vs-after-step is detected and reported.
    - A probe that compares only the lr sequence without the parameter-delta update (or vice versa) is
      incomplete and rejected.

- **AC-5: A committed, evidence-backed conclusion that resolves the first-step gap.** A conclusion document
  states, with committed artifacts, exactly which of the following holds and why: (a) the loaders were NOT
  multiset-consistent → the divergence is a real loader misalignment, now fixed (AC-2); (b) on a confirmed-fixed
  run with identical step-0 data the loss/grad match → the original ~0.35-vs-0.246 came from the un-provenanced
  old runs (e.g. pre-fix effective batch) and is resolved by AC-1's rerun; or (c) a concrete compute bug was
  found on identical data and fixed (AC-3). Plus AC-4's optimizer/lr verdict.
  - Positive Tests (expected to PASS):
    - The conclusion cites the AC-1 provenance/rerun artifact, the AC-2 30k multiset audit result, the AC-3
      cross-feed + identical-data parity result, and the AC-4 optimizer/lr result, with concrete numbers.
    - Every headline claim ("loaders strictly consistent", "no compute defect on identical data", "optimizer
      strictly identical") cites the specific committed gate/artifact that enforces it.
  - Negative Tests (expected to FAIL/be rejected):
    - A conclusion that asserts alignment without the committed audits, or that re-uses the prior "benign
      RNG/shuffle" claim without the 30k multiset proof, is rejected.
    - A conclusion claiming a compute bug without a localized divergence surface, or claiming no bug without the
      identical-data comparison, is rejected.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A complete diagnostic + verification suite: a confirmed-fixed instrumented 8-GPU rerun with a committed
provenance/step-0 artifact (AC-1); a full 30k-step rolling per-step global-multiset audit of both loaders with a
committed hash stream and a CI-style gate (AC-2); a 2×2 cross-feed plus identical-data production-stack step-0
loss/grad comparison that classifies and, if needed, fixes a localized compute cause (AC-3); a standalone
deterministic lr+AdamW parity probe with a gate (AC-4); and a committed conclusion citing all artifacts (AC-5).
Existing probes/tests are reused and extended rather than rewritten.

### Lower Bound (Minimum Acceptable Scope)
A confirmed-fixed instrumented rerun yielding trustworthy step-0 numbers and a runtime proof of rank-disjoint
effective batch 256 (AC-1); a full-30k global-multiset loader audit with committed hash streams and an explicit
pass/fail (AC-2); an identical-global-step-0 loss/grad comparison through both production stacks with the 2×2
cross-feed classification (AC-3); a deterministic lr+optimizer-delta parity probe (AC-4); and a committed
conclusion (AC-5). Any compute residual on identical data must at minimum be localized to a concrete surface
even if the full fix is deferred with justification.

### Allowed Choices
- Can use: the existing probes/dumpers (`tools/sft_grad_parity_probe.py`, `tools/sft_loader_sequence_probe.py`,
  `tools/sft_replay_parity_probe.py`, `tests/unit_tests/_ref_fanout_dump.py`, `_ref_pinned_run.py`, the committed
  pinned-artifact infrastructure, `test_openpi_pytorch_sft_sharding.py`); the reference venv via subprocess for
  reference-side dumps; numpy-seeded fixed noise/time; frame-id hashing for the loader audit; a fake model +
  fake grads for the optimizer probe; runtime instrumentation hooks that observe (not alter) loader/optimizer
  state.
- Cannot use: changing either trainer's training semantics to force a number to match; relaxing the
  identical-input requirement of the cross-feed; sampling the 30k audit without explicitly logging coverage;
  asserting any conclusion from config text instead of a runtime/committed artifact; writing scratch/large
  outputs under `/tmp` (must use `/mnt/public/xzxuan/tmp`).

> **Note on Determinism**: The dataloader-consistency bar is fixed by user decision to **identical global-batch
> multiset per step** (not byte-identical rank assignment). The audit scope is fixed to **full 30k rolling-hash**.
> A confirmed-fixed instrumented rerun **is required**. These are not open choices.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not
> prescriptive requirements.

### Conceptual Approach
- **AC-1 (provenance + rerun):** add a step-0 instrumentation hook in the SFT worker/loader that, on the first
  global step, records each rank's frame-id set and asserts the global union has 256 distinct ids
  (rank-disjoint); dump a provenance JSON (git rev + `git diff --stat` dirty flag, resolved Hydra config,
  package versions, checkpoint/norm-stats sha256, seed, world/micro/global batch). Launch a short job (a handful
  of steps) on 8 GPUs and capture the real step-0 loss/grad_norm/lr; write outputs under `/mnt/public/xzxuan/tmp`
  then commit the compact artifact under `docs/evidence/`.
- **AC-2 (30k multiset audit):** emit, from each loader, a per-global-step sorted frame-id list; fold it into a
  rolling hash (e.g. running sha256 of `step || sorted(frame_ids)`); compare the two rolling-hash streams
  step-by-step. The reference side reuses the rank-0 fanout model (`_ref_fanout_dump.py` mirrors how rank 0
  pulls `world_size*grad_accum` successive micro-batches); the RLinf side dumps its per-rank streaming partition
  with the explicit `dist_rank`/`dist_world_size` fix. Record per-rank AND global-set hashes so failures
  localize. Running the full 30k requires a data-only pass (no model), which is tractable; log exact coverage.
- **AC-3 (cross-feed + identical-data parity):** capture each repo's fully-materialized step-0 batch without
  perturbing loader RNG/order (clone tensors out of the first batch). Reuse the cross-feed machinery of
  `tools/sft_grad_parity_probe.py` (reference dump in its venv, RLinf forward/backward on identical inputs).
  Hold weights/norm-stats/noise/time/dtype/train-mode identical. Then, on the identical global step-0 batch
  established by AC-2, compare the production-stack step-0 loss + pre-clip fp32 global grad_norm; if a residual
  remains, walk the ladder sample-ids → raw tensors → transformed tensors → model inputs → forward loss →
  backward grad-norm, with a per-module grad breakdown, to localize.
- **AC-4 (lr/optimizer probe):** build a tiny fake model whose parameter groups mirror the real betas/eps/wd
  grouping; feed identical fixed fake gradients; step both repos' lr-schedule + AdamW logic in lockstep; compare
  the lr sequence and the per-step parameter deltas + optimizer state. Reuse the `openpi_cosine` scheduler
  (`rlinf/hybrid_engines/fsdp/utils.py`) vs the reference warmup+cosine (`scripts/train_pytorch_new.py`).

### Relevant References
- `rlinf/data/datasets/behavior/behavior_sft_dataset.py` — streaming partition with explicit
  `dist_rank`/`dist_world_size` (the committed fix) and `partition_chunk_indices`.
- `rlinf/data/datasets/behavior/behavior_sft_data_loader.py` — DataLoader build (spawn context, worker_init),
  passes `dist_rank`/`dist_world_size`.
- `rlinf/workers/sft/fsdp_sft_worker.py` / `fsdp_vla_sft_worker.py` — train step, loss (`train/loss`), pre-clip
  fp32 all-reduced grad-norm (`train/grad_norm`), global/micro batch + grad-accum wiring.
- `rlinf/hybrid_engines/fsdp/utils.py` — `openpi_cosine` scheduler, AdamW build, fp32 grad-norm.
- Reference: `scripts/train_pytorch_new.py` — rank-0 fanout loop, lr schedule + AdamW; `_ref_fanout_dump.py`,
  `_ref_pinned_run.py`, `_ref_model_grad_dump.py`.
- Existing evidence: `docs/sft-loss-parity-evidence.md`, `docs/sft-first50-step-evidence.md`,
  `docs/evidence/r34_production_batch_topology.json`.
- Reference run log: `…/wandb/run-20260602_093939-w2kz4275/files/output.log` (step-0 loss 0.246, grad_norm 2.17).

## Dependencies and Sequence

### Milestones
1. **M1 — Trusted baseline (AC-1):** instrument + short confirmed-fixed 8-GPU rerun; commit provenance + step-0
   artifact + runtime rank-disjoint/256 proof.
   - Phase A: provenance + runtime instrumentation hook.
   - Phase B: short rerun, capture step-0 loss/grad/lr, commit artifact.
2. **M2 — Loader strict multiset audit (AC-2):** full 30k rolling per-step global-multiset hash audit of both
   loaders; commit hash streams + gate. Depends on M1 (audit the confirmed-fixed loader).
3. **M3 — Identical-data first-step parity (AC-3):** 2×2 cross-feed classification + identical-global-step-0
   production-stack loss/grad comparison; localize+fix any compute residual. Depends on M2 (identical batch
   established).
4. **M4 — Optimizer/lr strict identity (AC-4):** standalone deterministic probe; independent of M1–M3.
5. **M5 — Independent audit + synthesis (AC-5):** independent analyze-route audit of the methodology/evidence,
   then a committed conclusion. Depends on M2, M3, M4.

Dependencies: M2→M1; M3→M2; M4 independent; M5→(M2,M3,M4). M1 and a head-start on M4 may proceed in parallel.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Add step-0 provenance + runtime rank-disjoint/256 instrumentation; run a short confirmed-fixed 8-GPU SFT job; commit the provenance + step-0 loss/grad/lr artifact and the runtime proof that the loader fix executed | AC-1 | coding | - |
| task2 | Build the full-30k rolling per-global-step frame-id multiset audit for both loaders (reference rank-0 fanout vs RLinf per-rank streaming), recording per-rank AND global-set hashes; commit the hash streams + a pass/fail gate; include an instrumentation-non-perturbation re-run check | AC-2 | coding | task1 |
| task3 | Capture each repo's fully-materialized step-0 batch; run the 2×2 cross-feed (identical weights/norm-stats/noise/time/dtype/mode) to classify data-vs-compute; on the identical global step-0 batch compare production-stack step-0 loss + pre-clip fp32 grad-norm; localize+fix any residual to a concrete surface | AC-3 | coding | task2 |
| task4 | Build a standalone deterministic lr+AdamW parity probe (fake model with representative param groups + fixed fake grads); assert identical per-step lr sequence and per-step parameter deltas/optimizer state across the two update logics over warmup+cosine | AC-4 | coding | - |
| task5 | Independent audit of the methodology + committed evidence: are the loss/grad-norm logging semantics matched, the cross-feed truly identical-input, the 30k audit truly exhaustive and correctly hashed, the optimizer probe representative; no claim asserted from config text | AC-1, AC-2, AC-3, AC-4 | analyze | task3, task4 |
| task6 | Synthesize the committed conclusion document tying AC-1..AC-4 artifacts to one of the three resolution outcomes (loader misalignment fixed / old-run artifact resolved by rerun / compute bug found+fixed) plus the optimizer verdict | AC-5 | coding | task5 |

## Claude-Codex Deliberation

### Agreements
- The forward-loss/grad parity on identical input and the effective-batch (32→256) fix are already PROVEN and
  must NOT be redone; this effort targets the new questions (strict 30k multiset consistency, a confirmed-fixed
  step-0 baseline, identical-data compute parity, strict lr/optimizer identity).
- A 2×2 cross-feed (each repo's actual step-0 batch through both models, all else identical) is the correct way
  to classify the first-step gap as data-side vs compute-side.
- The first move must be verifying whether the 0605/0608 runs used the fixed loader; since provenance can't
  confirm it, a short confirmed-fixed instrumented rerun is required.
- "Step-0 loss" must be defined precisely (pre-`optimizer.step`, global-mean reduction, matched logging
  semantics); the optimizer/lr probe governs trajectory/lr identity, not the pre-update step-0 loss.
- Tolerances must be explicit per surface; the loader audit must record both per-rank and global-set hashes so a
  failure distinguishes rank-assignment from sample-set divergence; instrumentation must be proven
  non-perturbing.

### Resolved Disagreements
- **Dataloader invariant** (Codex flagged byte-identical-per-rank as stricter than training-equivalence): RESOLVED
  by user → **identical global-batch multiset per step** (rank assignment may differ harmlessly). Rationale: this
  is the true training-equivalence bar, fully explains curve differences, and is achievable without forcing
  RLinf to reproduce the reference's exact rank-0 fanout assignment.
- **First-step success definition** (diagnose-only vs hard parity): RESOLVED by user → establish identical global
  step-0 data, then REQUIRE compute parity on it; a material residual on identical data is treated as a
  compute-side bug to localize and fix. Rationale: the user's stated logic that identical data + large gap ⇒
  compute problem.
- **Audit scope** (bounded sample vs full): RESOLVED by user → **full 30k rolling-hash** (with non-perturbation
  re-run check). Rationale: the user requires strict per-step confirmation across the whole schedule.
- **Clean rerun vs offline-only**: RESOLVED by user → **yes, short instrumented rerun** proving the fix executed
  at runtime. Rationale: the 0605/0608 runs lack provenance to confirm the fixed loader was active.

### Convergence Status
- Final Status: `converged`

## Pending User Decisions

_No decisions remain PENDING. All four decisions surfaced during planning were resolved by the user and are
recorded below for traceability._

- DEC-1: Dataloader alignment bar.
  - Resolution: **Identical global-batch multiset per step** (not byte-identical rank assignment).
  - Decision Status: RESOLVED (user).
- DEC-2: First-step success definition.
  - Resolution: **Establish identical global step-0 data, then require compute parity on it; localize+fix any
    material residual as a compute bug.**
  - Decision Status: RESOLVED (user).
- DEC-3: 30k-step loader audit scope.
  - Resolution: **Full 30k rolling-hash** (coverage explicitly logged; non-perturbation re-run check).
  - Decision Status: RESOLVED (user).
- DEC-4: Fresh confirmed-fixed instrumented rerun.
  - Resolution: **Yes — run a short instrumented 8-GPU SFT job proving the loader fix executed at runtime.**
  - Decision Status: RESOLVED (user).

## Implementation Notes

### Numeric and semantic conventions (defaults, may be tightened during implementation)
- Loss parity (bf16): per-sample/per-batch |Δ| ≤ rel 5% OR abs 0.01 (the existing committed loss-parity band);
  fp32 path tighter at |Δ| ≤ 1e-3.
- Grad-norm parity (identical data): rel ≤ 2% on the pre-clip fp32 all-reduced global norm (consistent with the
  committed precision grad-parity gate); note grad-norm is effective-batch dependent, so comparisons must be at
  equal effective batch.
- LR: exact deterministic schedule; |Δ| ≤ 1e-12. Step-0 lr = 2.4975e-08, step-1000 lr = 2.5e-5.
- Optimizer parameter deltas: fp32 rel ≤ 1e-5 given identical fake gradients and identical initial weights.
- Frame-id multiset: EXACT per-step hash equality across repos.
- All scratch/experiment outputs, dumps, and logs MUST be written under `/mnt/public/xzxuan/tmp`, never `/tmp`;
  committed evidence artifacts live under `docs/evidence/`.

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone",
  "Step", "Phase", or similar workflow markers.
- These terms are for plan documentation only, not for the resulting codebase.
- Use descriptive, domain-appropriate naming in code instead.

--- Original Design Draft Start ---

# Investigate Why the First-Step Loss and Grad Norm Differ So Significantly

## Background

There is an obvious misalignment: the first-step loss and grad norm of the two repos' training runs are clearly not aligned. I also used `/mnt/public/xzxuan/logs_to_tensorboard.py` to plot the TensorBoard curves of the reference repo and two RLinf experiment runs. Focus on **RLinf_0605** and **openpi**. (RLinf_0608 is a newly launched SFT experiment.) The file `/mnt/public/xzxuan/tb_logs/image.png` contains the curves for the two experiments RLinf_0605 and openpi. You can clearly see that RLinf's loss is noticeably lower than openpi's.

Overall, I still believe the loss is **not strictly aligned**, and I need you to investigate this carefully again. **Do not skip this just because you did a round of analysis before** — you now need to dig in deeper. Write code: with **completely identical data, completely identical model, and identical norm stats**, feed both repos' SFT code and observe exactly what the first-step loss, grad, and grad norm are. Figure out why the gap is so large — this is clearly a problem!

## Verification Tasks

In addition, to verify why RLinf's loss curve is lower than the reference repo's, use code to actually validate the following two things:

1. **The dataloader logic is completely identical.** That is, under 8 GPUs, with the same batch size and dataset, confirm whether — across 30k steps — at every step, the data each GPU receives is *completely identical* between the two repos. Ensure that under this streaming mode and distributed training setup, the dataloader logic is **strictly aligned**! Write code to confirm this!

2. **The lr and optimizer logic is strictly identical.** That is, under the 30k-step distributed training setup, the per-step lr and optimizer updates of the two repos must be *strictly identical*. You may use a fake model and fake data to iterate quickly — the main point is that the two update logics must be strictly aligned!

## Supplementary Experiment Logs

1. **Reference repo log:** `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/wandb/run-20260602_093939-w2kz4275/files/output.log` — records the complete loss and grad norm.
2. **RLinf first run:** `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla` — this log only has 3 decimal places, so I recommend pulling the full data directly from TensorBoard.
3. **RLinf second run:** `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260608-04:14:29-behavior_pi05_vla` — use the same method as above.

As you can see, two RLinf SFT runs were performed, and the first-step loss in both is around **0.35**, which is clearly much larger than the reference repo's. The grad norm is also larger. This is definitely abnormal — I need you to inspect rigorously and tell me **where the code is wrong**!
--- Original Design Draft End ---
