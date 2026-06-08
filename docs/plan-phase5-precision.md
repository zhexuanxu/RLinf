# Phase 5 — Mixed-Precision Alignment Between RLinf SFT and the Reference

## Goal Description

Establish, by direct **runtime evidence** (not config text), whether RLinf's BEHAVIOR pi0.5 8-GPU FSDP SFT
mixed-precision recipe is identical to the reference (`openpi-comet-pytorch-mixed`,
`pi05_b1k-task0000_sft_pytorch_mixed`, launched `fsdp1` / `USE_AUTOCAST=0`), and make the two **strictly
consistent in effect** on every precision surface. The reference's effective recipe is: **fp32 master/load
params**, **FSDP `MixedPrecision(param_dtype=bf16, reduce_dtype=fp32)`**, **buffers left at the constructor
default**, **no `torch.autocast`**, **fp32 grad-norm**, **fp32 fused AdamW states**, **bf16 flow-matching
compute/loss**, **no EMA**.

A preliminary read of the RLinf code shows the effective runtime recipe **already appears to match** the
reference on every major surface: the SFT build sets `load_for_training: True`, so the master params load
**fp32 strictly** (never downcast); `fsdp_config.mixed_precision` resolves to `param_dtype=bf16` (via
`${actor.model.precision}`), `reduce_dtype=fp32`, `buffer_dtype=fp32`; `amp_autocast` is off; grad-norm is
accumulated in fp32; AdamW states are fp32; flow-matching loss is bf16. The draft's premise ("definitely not
aligned") therefore appears to be a **misreading of an overloaded config field**, and the draft's literal
edit (`precision: fp32`) would be **harmful**: because `param_dtype: ${actor.model.precision}`, setting
`precision` to fp32 would resolve the FSDP compute `param_dtype` to fp32 and **disable bf16 compute** — the
opposite of the intended recipe.

This plan therefore does two things, per the resolved user decisions: (1) **verify** the effective recipe on
both repos with a non-invasive dual-repo dtype-ledger harness and a same-batch controlled-step comparison;
and (2) **decouple the config footgun regardless** — set the FSDP compute `param_dtype` explicitly,
independent of the `precision` field, so a future change to the load-dtype selector can never silently break
bf16 compute. "Fully identical" is judged as **effective-runtime equivalence** (the same dtype on every
runtime surface), with benign mechanism differences (e.g. `buffer_dtype` set explicitly vs left default;
`fused` vs non-fused AdamW) explicitly observed and documented rather than forced to be textually identical.
If the runtime ledger confirms the recipe already matches, the plan **stops with a documented null result**:
precision is verified-aligned and is therefore not the cause of the residual eval gap measured in the prior
phase. The reference repository's training behavior is never modified.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.
"Identical" means **effective-runtime equivalence** (same dtype/state per surface), not bitwise config
identity; any benign mechanism difference must be observed and documented, not assumed. Every ledger record
must capture the surface name, the observed dtype/flag, the repository, the rank, and the resolved config
source, so a claim is always backed by a runtime observation rather than YAML text.

- **AC-1: A non-invasive dual-repo runtime dtype ledger exists and is reproducible.** A harness collects, for
  BOTH repos on the same pinned inputs, the effective dtype/flag of every precision surface: pre-FSDP
  loaded/master param dtype; post-FSDP exposed param dtype sampled BOTH outside forward AND during forward
  (FSDP can expose an fp32 handle while computing in bf16); the FSDP `MixedPrecision` object's `param_dtype`,
  `reduce_dtype`, `buffer_dtype`, and the flags `cast_forward_inputs` / `cast_root_forward_inputs` /
  `keep_low_precision_grads`; gradient dtype after backward; the gradient-reduction dtype; buffer dtypes
  (with the PyTorch/FSDP constructor default for an omitted `buffer_dtype` OBSERVED in the installed
  version, not inferred); optimizer state dtypes (`exp_avg`, `exp_avg_sq`) after one optimizer step, plus the
  AdamW construction flags (`fused` / `foreach` / `capturable`, betas, eps, weight_decay); the loss / model
  output / action / noise / time dtypes; `torch.autocast` state and grad-scaler no-op state; and the
  saved-checkpoint dtype plus the dtype after a load-after-save round trip. The harness is **non-invasive**:
  it inspects state dicts, the FSDP wrapper's mixed-precision object, and a small pinned-step run reusing the
  existing pinned-input infrastructure — it does NOT edit either trainer's source.
  - Positive Tests (PASS):
    - Running the harness emits a committed ledger artifact for each repo, with a value recorded for every
      surface above, tagged by rank, and the harness regenerates the committed ledger from its documented
      invocation.
    - The buffer-dtype default of the reference's omitted `buffer_dtype` is recorded as an OBSERVED value
      from the installed PyTorch/FSDP version.
  - Negative Tests (FAIL):
    - A "ledger" derived only from reading YAML/config text (no runtime introspection) is rejected.
    - A ledger missing any enumerated surface, or recording the reference `buffer_dtype` default by assumption
      rather than observation, fails.

- **AC-2: Per-surface effective-equivalence verdict between the two repos.** Each precision surface from AC-1
  is compared and marked PASS (effectively identical) or recorded as a justified, documented benign
  difference, on rank 0 AND at least one sharded (non-zero) rank. The expected verdict is: master/load params
  fp32 == fp32; FSDP compute `param_dtype` bf16 == bf16; `reduce_dtype` fp32 == fp32; buffers fp32 (RLinf
  explicit) == effective fp32 (reference default) [documented benign mechanism difference]; autocast off ==
  off; grad-norm accumulation fp32 == fp32; AdamW state fp32 == fp32 (with any `fused`/`foreach` construction
  difference documented); loss/action/noise/time bf16 == bf16; EMA none == none.
  - Positive Tests (PASS):
    - Every surface is either byte-equal in effective dtype or carries an explicit one-line justification of
      a benign mechanism difference, with no surface left unexamined.
    - The verdict holds on a sharded rank, not only rank 0.
  - Negative Tests (FAIL):
    - Declaring "identical" while any enumerated surface is unobserved, or while the reference buffer default
      was assumed, fails.
    - A surface whose effective dtype genuinely differs (e.g. master bf16 on one side) and is left
      unreconciled fails.

- **AC-3: Same-batch controlled-step parity, including a numeric grad-norm check.** On ONE pinned batch with
  identical injected flow-matching noise/time, identical RNG seed/device, identical model train/eval mode, and
  the same initial weights, a dtype ledger is captured before forward, after forward, after backward, before
  the optimizer step, and after the optimizer step, for both repos; and the gradient-norm VALUE is compared
  numerically (because the two grad-norm implementation paths differ — a custom mixed-precision fp32 reduction
  vs the reference FSDP `clip_grad_norm_`), not merely asserted equal because "both are fp32".
  - Positive Tests (PASS):
    - The per-stage dtype ledgers match across the two repos on the controlled step.
    - The grad-norm values match within a tight recorded tolerance, with the accumulation/implementation path
      documented.
  - Negative Tests (FAIL):
    - Asserting grad-norm parity from "both compute in fp32" without a numeric comparison on the controlled
      step fails.
    - A controlled step that does not pin noise/time/seed/initial-weights (so a difference could be RNG, not
      precision) fails the protocol.

- **AC-4: The config-coupling footgun is removed; the FSDP compute `param_dtype` is set explicitly.** The FSDP
  compute `param_dtype` is set to bf16 **independently of the load-dtype selector**, so that changing the
  model load/precision field can no longer silently change the FSDP compute dtype. The change PRESERVES the
  effective recipe (fp32 master + bf16 compute + fp32 reduce), verified by re-running the AC-1 ledger after the
  edit. The draft's literal `precision: fp32` edit is NOT applied.
  - Positive Tests (PASS):
    - After the change, the FSDP `MixedPrecision.param_dtype` observed at runtime is bf16 and does NOT change
      when the model load/precision selector is varied in a dry-run resolution; the AC-1 ledger is unchanged
      on every surface versus before the edit.
    - The load path still loads fp32 master params (the AC-2 master-params verdict is unchanged).
  - Negative Tests (FAIL):
    - A change that lets the load/precision field propagate into the FSDP compute `param_dtype` (so a fp32
      load selector yields fp32 compute, disabling bf16) fails.
    - Applying the draft's literal `precision: fp32` edit, or any edit that alters the effective compute dtype
      away from bf16, fails.

- **AC-5: A documented conclusion — verified-aligned null result, or an aligned-and-re-verified fix.** The
  plan ends with one of two explicitly-evidenced outcomes. If the ledgers (before and after the AC-4 decouple)
  show the recipe is effectively identical on every surface, the conclusion records a **null result**:
  precision is verified-aligned and is therefore NOT the source of the residual eval gap, leaving the
  trajectory-level causes (not in scope here) as the remaining suspects. If any surface showed a REAL
  effective-dtype mismatch, it is aligned to the reference and the affected ledger/controlled-step evidence is
  re-run to confirm the fix.
  - Positive Tests (PASS):
    - The committed conclusion states, per surface, "identical (verified)" or "was mismatched → aligned →
      re-verified", each backed by a ledger/controlled-step observation.
    - When all surfaces are identical, no functional precision patch beyond the AC-4 decouple is made (no
      cosmetic change is forced).
  - Negative Tests (FAIL):
    - A conclusion of "aligned" that is not backed by the runtime ledger (only by config edits) fails.
    - Forcing a precision-changing patch when the ledger already shows effective equivalence fails.

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
A non-invasive dual-repo runtime dtype ledger covers every enumerated precision surface (load/master, FSDP
`param_dtype`/`reduce_dtype`/`buffer_dtype` and the MixedPrecision flags, autocast, grad-scaler, grad-norm
dtype + numeric value, AdamW state dtype + construction flags, loss/action/noise/time dtype, saved-checkpoint
and load-after-save dtype), on rank 0 and a sharded rank, with the reference's omitted `buffer_dtype` default
observed in the installed version; a same-batch controlled-step comparison confirms the per-stage dtype
ledger and grad-norm numeric parity; the `param_dtype`/`precision` coupling is decoupled with the effective
recipe preserved and re-verified; and a per-surface conclusion records "identical (verified)" or "mismatched
→ aligned → re-verified". Any genuinely-mismatched surface is aligned to the reference and re-verified.

### Lower Bound (Minimum Acceptable Scope)
The dual-repo runtime ledger covers the master/load, FSDP `param_dtype`/`reduce_dtype`/`buffer_dtype`,
autocast, grad-norm dtype, AdamW state dtype, and loss-path dtype surfaces with a per-surface
effective-equivalence verdict (benign mechanism differences documented); the `param_dtype`/`precision`
coupling is decoupled with the effective recipe preserved and re-verified by re-running the ledger; and a
clear conclusion states whether the recipe is effectively identical (null result) or names the specific
mismatched surface that was aligned. The same-batch grad-norm numeric check is included; a full
all-stage controlled-step ledger may be reduced to the forward and post-backward stages if needed, provided
the grad-norm numeric parity is still checked.

### Allowed Choices
- Can use: non-invasive runtime introspection (inspecting loaded state-dict dtypes; reading the FSDP wrapper's
  `MixedPrecision` object and its flags; a small pinned-step harness reusing the existing pinned-input /
  injected-noise infrastructure); the reference repo and its committed scripts/config as read-only ground
  truth; the reference Python venv for cross-checks; committed ledger/CSV/JSON evidence artifacts under the
  repository's docs evidence area; a config change limited to **decoupling the FSDP compute `param_dtype` from
  the model precision field** (setting `param_dtype` explicitly to bf16) while preserving fp32 master load;
  documenting benign mechanism differences (explicit vs default `buffer_dtype`; `fused` vs non-fused AdamW)
  rather than forcing textual identity.
- Cannot use: editing the reference repository's training source or changing its training behavior; applying
  the draft's literal `precision: fp32` edit or any change that moves the effective FSDP compute dtype away
  from bf16 or the master params away from fp32; asserting precision identity from config/YAML text without a
  runtime observation; forcing a cosmetic precision patch when the ledger already shows effective equivalence;
  treating a single-rank (rank-0-only) observation as proof for sharded params/grads/optimizer state.

> **Note on Deterministic Designs**: The precision recipe target is fixed by the reference (fp32 master, bf16
> FSDP compute, fp32 reduce, no autocast); the ledger surfaces and the decouple are prescriptive. The genuinely
> open choices (verify-vs-patch posture, the "identical" materiality standard, the introspection method, and
> the stop-vs-pivot behavior on a null result) were resolved with the user and are recorded in `## Pending
> User Decisions`.

## Feasibility Hints and Suggestions

> **Note**: For reference and understanding only — conceptual, not prescriptive.

### Conceptual Approach
1. **Build the ledger harness first (non-invasive).** For RLinf, drive a tiny pinned SFT step through the real
   FSDP build, then read the dtypes off the wrapped model, the FSDP `MixedPrecision` object, the gradients
   after backward, and the optimizer state after one step; for the reference, do the equivalent from its
   training entry on a pinned step in its own venv. Emit a structured per-surface ledger for each.
2. **Diff the ledgers** surface by surface under the effective-equivalence standard, recording each as
   identical or a documented benign difference, on rank 0 and a sharded rank.
3. **Same-batch controlled step** with pinned noise/time/seed/initial-weights, capturing the per-stage dtype
   ledger and the numeric grad-norm on both sides.
4. **Decouple the config:** give the FSDP compute `param_dtype` its own explicit value (bf16) instead of
   interpolating it from the model precision field; keep `load_for_training: True` and the fp32 master load.
   Re-run the ledger to prove the effective recipe is unchanged.
5. **Conclude:** if every surface is identical, write the verified-aligned null result; otherwise align the
   specific mismatched surface to the reference and re-verify.

### Relevant References
- The reference training entry and launch script (`scripts/train_pytorch_new.py`, `run.sh`) and the reference
  training config — the precision flow (`mp_bfloat16` → fp32 load + `MixedPrecision(param_dtype=bf16,
  reduce_dtype=fp32)`, no autocast, fp32 grad-norm, fused AdamW).
- RLinf SFT configs: `examples/sft/config/model/pi0_5_pytorch.yaml` (`precision`, `load_for_training`),
  `examples/sft/config/behavior_pi05_vla.yaml` (`fsdp_config.mixed_precision`),
  `examples/sft/config/training_backend/fsdp.yaml` (`amp_autocast`).
- RLinf FSDP strategy / model manager (the FSDP wrap that builds the `MixedPrecision` policy; the
  `amp_autocast` context; the grad-norm utility; the AdamW construction) and the eval/SFT model factory
  (`rlinf/models/embodiment/openpi_pytorch/__init__.py`, the `load_for_training` fp32-load path).
- The existing pinned-input / injected-noise harness from the prior phase, reusable for the controlled step.

## Dependencies and Sequence

### Milestones
1. **M1 — Ledger harness (AC-1).**
   - Phase A: implement the non-invasive RLinf-side ledger over a pinned FSDP step.
   - Phase B: implement the reference-side ledger over a pinned step in the reference venv; observe the
     omitted-`buffer_dtype` default in the installed version.
2. **M2 — Surface diff + verdict (AC-2).** Depends on M1. Per-surface effective-equivalence verdict on rank 0
   and a sharded rank.
3. **M3 — Controlled-step parity (AC-3).** Depends on M1. Pinned-batch per-stage dtype ledger + numeric
   grad-norm parity.
4. **M4 — Decouple the footgun (AC-4).** Depends on M2 (so the decouple is proven to preserve the verified
   recipe). Set the FSDP compute `param_dtype` explicitly; re-run the ledger.
5. **M5 — Conclusion (AC-5).** Depends on M3, M4. Write the verified-aligned null result, or align a real
   mismatch and re-verify.

Dependency summary: M1 is the foundation; M2 and M3 both depend on M1 and are independent of each other; M4
depends on M2; M5 depends on M3 and M4. A genuine mismatch found in M2/M3 is aligned to the reference and the
affected evidence re-run before M5.

## Task Breakdown

Each task has exactly one routing tag: `coding` (Claude) or `analyze` (Codex via `/humanize:ask-codex`).

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Build the non-invasive dual-repo runtime dtype-ledger harness (RLinf pinned FSDP step + reference pinned step in its venv); record every enumerated surface incl. FSDP `MixedPrecision` object + flags, gradient/reduce/buffer dtypes (reference omitted-`buffer_dtype` default OBSERVED), AdamW state dtype + construction flags after step 1, autocast/grad-scaler state, loss/action/noise/time dtype, saved + load-after-save dtype; emit a committed ledger per repo, tagged by rank | AC-1 | coding | - |
| task2 | Diff the two ledgers surface by surface under the effective-equivalence standard; record each as identical or a documented benign mechanism difference, on rank 0 AND a sharded rank; commit the per-surface verdict | AC-2 | coding | task1 |
| task3 | Same-batch controlled-step comparison: pinned batch + identical noise/time/seed/initial-weights; capture per-stage dtype ledger (pre/after-forward/after-backward/pre-step/post-step) and compare the grad-norm VALUE numerically within a tight recorded tolerance | AC-3 | coding | task1 |
| task4 | Decouple the config footgun: set the FSDP compute `param_dtype` explicitly to bf16 (independent of the model precision field), keep fp32 master load; prove via a dry-run resolution that the load selector no longer feeds the compute dtype, and re-run the AC-1 ledger to confirm the effective recipe is unchanged | AC-4 | coding | task2 |
| task5 | Synthesize the conclusion: if every surface is identical, write the verified-aligned null result (precision is not the residual-gap cause); if a real mismatch was found, align it to the reference and re-run the affected ledger/controlled-step evidence; commit the per-surface conclusion | AC-5 | coding | task3, task4 |
| task6 | Independent audit (analyze route): cross-check that the precision-surface ENUMERATION is complete and that no surface was missed or asserted from config text rather than runtime observation, and that the reference omitted-`buffer_dtype` default was observed not assumed | AC-1, AC-2 | analyze | task2 |

## Claude-Codex Deliberation

### Agreements
- Actual runtime dtypes — not config text — are the ground truth; FSDP changes parameter views during
  forward/backward, so a config diff is insufficient.
- The draft's claimed discrepancy is most likely NOT real given `load_for_training: True` (fp32 master) +
  `precision: "bf16"` (compute), and the draft's literal `precision: fp32` edit would be harmful because the
  config interpolates `param_dtype` from `precision`, disabling bf16 compute.
- "Fully identical" should be judged as effective-runtime equivalence (per-surface dtype/state), with benign
  mechanism differences (explicit vs default `buffer_dtype`; `fused` vs non-fused AdamW) documented rather
  than forced to textual identity.
- The precision surfaces that must be checked for parity: master/load dtype, FSDP `param_dtype` /
  `reduce_dtype` / `buffer_dtype` (+ MixedPrecision flags), autocast, grad-scaler, grad-norm dtype + numeric
  value, AdamW state dtype + construction, loss/action/noise/time dtype, saved + load-after-save dtype, EMA.
- The grad-norm must be compared numerically, not assumed equal because both accumulate in fp32, since the
  two implementation paths differ.

### Resolved Disagreements
- **Draft premise vs evidence (resolved by user decision):** the draft asserts the configs are "definitely not
  aligned" and proposes literal edits, but the gathered runtime evidence indicates the effective recipe
  already matches and the literal edits would break bf16 compute. Reconciliation: the plan verifies the recipe
  with a runtime ledger (treating the draft's edits as a hypothesis), DECOUPLES the `param_dtype`/`precision`
  config coupling regardless (the user chose the decouple-regardless option), judges identity as
  effective-runtime equivalence, and stops with a documented null result if the recipe already matches — never
  applying the harmful literal edit. The draft's intent (fp32 master + bf16 compute + fp32 reduce) is honored;
  its specific config mechanism is corrected.
- **Verify-first vs patch-regardless (resolved):** verify-first via the runtime ledger, with the single
  config decouple applied regardless (it preserves the effective recipe and removes a real footgun).

### Convergence Status
- Final Status: `converged` (one Codex first-pass + one convergence round; the second Codex pass returned
  AGREE with only optional improvements — all folded into the ACs — and no required changes or high-impact
  disagreements; all user decisions resolved, no pending items remain).

## Pending User Decisions

- DEC-1: **How to proceed given the evidence that the recipe already matches.**
  - Claude Position: Verify-first via the runtime ledger; patch only a real mismatch and the config footgun.
  - Codex Position: Same — verify-first; patch only the config-coupling footgun or a proven mismatch.
  - Tradeoff Summary: Verify-first avoids a harmful/cosmetic edit; decoupling-regardless adds robustness.
  - Decision Status: **RESOLVED — Decouple the config regardless (explicit `param_dtype`, independent of
    `precision`), in addition to the runtime verification; do NOT apply the literal `precision: fp32` edit.**

- DEC-2: **Introspection method.**
  - Claude Position: Non-invasive external harness (inspect state dicts + FSDP MixedPrecision objects + a
    pinned-step), no trainer edits.
  - Codex Position: A runtime ledger is required either way; non-invasive is acceptable.
  - Tradeoff Summary: Non-invasive avoids editing the reference trainer; in-trainer hooks are more direct but
    intrusive.
  - Decision Status: **RESOLVED — Non-invasive external harness; do not edit either trainer's source.**

- DEC-3: **Behavior if the ledger confirms a match.**
  - Claude Position: Stop with a documented null result (precision is not the residual-gap cause).
  - Codex Position: Verify-first; if it matches, do not force a precision patch.
  - Tradeoff Summary: Stopping avoids scope creep; pivoting would expand into trajectory-level causes.
  - Decision Status: **RESOLVED — Stop with a documented null result; do not pivot or force a cosmetic patch.**

- DEC-4: **Materiality of "fully identical".**
  - Claude Position: Effective-runtime equivalence with documented benign mechanism differences.
  - Codex Position: Effective-runtime equivalence, not bitwise config identity.
  - Tradeoff Summary: Effective equivalence reflects what actually affects training; bitwise identity would
    force cosmetic, non-functional changes.
  - Decision Status: **RESOLVED — Effective-runtime equivalence (same dtype/state per surface), benign
    mechanism differences documented.**

## Implementation Notes

### Code Style Requirements
- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone",
  "Phase", "Step", "task<N>", or similar workflow markers. These belong in this plan document only.
- Use descriptive, domain-appropriate naming in code (e.g. a precision/dtype ledger, a mixed-precision
  introspection helper) instead.
- New behavior must be covered by tests/evidence; experiment/ledger/run outputs go under the project scratch
  location, never `/tmp`; preserve self-containment (no real `import openpi` in the package or relocated data
  modules); do not modify the reference repository's training behavior, and do not change the in-repo old
  `openpi/` package's behavior.
- The single permitted config change is decoupling the FSDP compute `param_dtype` from the model precision
  field; it must preserve the effective recipe (fp32 master load, bf16 compute, fp32 reduce) and be re-verified
  by the ledger.


--- Original Design Draft Start ---

# Task: Align Mixed-Precision Settings Between Repositories

The current code is definitely **not aligned in precision**.

## Reference

Carefully study the reference repository:

- Training script: `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/scripts/train_pytorch_new.py`
- Launch script: `/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/run.sh`

## How Our Code Handles Precision

In our code, the precision-related logic flows into the following block:

```python
model_load_dtype = config.pytorch_training_precision
mp_policy = None
if model_load_dtype == "mp_bfloat16":
    model_load_dtype = "float32"

if dist_method in ["no_dist", "ddp"]:
    assert use_autocast == True
elif dist_method == "fsdp1":
    if not use_autocast:
        from torch.distributed.fsdp import MixedPrecision
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
```

In other words, the reference repo effectively uses:

- `model_load_dtype = fp32`
- `param_dtype = torch.bfloat16`
- `reduce_dtype = torch.float32`

## The Discrepancy

This clearly **differs** from the current SFT code in the RLinf repository:

- `examples/sft/config/model/pi0_5_pytorch.yaml` sets `precision: "bf16"`, but it should be **fp32**.
- `examples/sft/config/behavior_pi05_vla.yaml` sets `param_dtype: ${actor.model.precision}`, but `param_dtype` should be **bf16**.

## Requirements

1. Thoroughly study and understand the mixed-precision setup in **both** the reference repo and the RLinf repo.
2. Ensure the two repositories are **strictly consistent** in their precision configuration.
3. **Test carefully** to verify that the settings are **fully identical**.
--- Original Design Draft End ---
