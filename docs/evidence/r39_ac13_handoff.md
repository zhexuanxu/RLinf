# AC-13 Final Handoff (task18, R39)

This is the final cross-cutting AC-13 closeout for the Phase-3 openpi_pytorch refactor + SFT alignment.
It records the four AC-13 invariants — full CPU suite green, byte-exact tokenizer parity skip-gating
(DEC-3), the in-repo old `openpi/` package no-touch, and changed-file Ruff/format hygiene — proven from a
clean tree, with exact commands and outputs.

- **Phase-3 branch point:** `e2092e85ccac3d02782fa4c6a5e136744aedd994` ("Merge remote-tracking branch
  'xzxuan/dualsys'") — the merge-base of `feat/openpi-pytorch-migration` with `main`.
- **Source revision at handoff:** see the committing revision of this file.

## (1) Full CPU suite green

```
TMPDIR=/mnt/public/xzxuan/tmp PYTHONPATH=$REPO MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m pytest tests/unit_tests/test_openpi_pytorch_*.py -q -rs
# -> 143 passed, 1 skipped, 4 warnings  (EXIT 0)
```

- **Result: 143 passed, 1 skipped** (EXIT 0).
- The **1 skipped** test is `test_openpi_pytorch_converters.py:621` (the JAX→PyTorch-new converter
  value check) — skipped because it is heavy (loads a ~14 GB JAX pytree) and env-gated:
  `set RLINF_RUN_JAX_CONVERT=1` with the JAX reference present to run it (DEC-5 heavy-evidence gate). This
  is a deliberate DEC-5 evidence skip, NOT a failure.
- **No tokenizer test is in the skip** — see (2): both tokenizer parity tests RAN and PASSED in this
  environment.

## (2) Byte-exact tokenizer parity skip-gating (DEC-3)

The bundled tokenizer `.model` was moved out of the repo (DEC-3); the byte-exact tokenizer parity tests
RUN-and-PASS when their external dependency is present and SKIP-gate (never fail) when it is absent. In
THIS environment both dependencies are present, so both ran and passed (`2 passed` in a focused run):

- **`test_openpi_pytorch_behavior_parity.py::test_tokenizer_parity_against_openpi`** — gated on the
  externally-installed `openpi` package (`pytest.importorskip("openpi.models.tokenizer")`). `openpi` IS
  importable in this venv → **RAN and PASSED**, asserting the vendored tokenizer matches `openpi`'s
  byte-for-byte. (If `openpi` were absent it would skip-gate, not fail.)
- **`test_openpi_pytorch_sft_ref_parity.py::test_tokenizer_exact_parity_vs_reference`** — gated on the
  reference py-3.11 venv (`_REF_PY`). Present in this environment → **RAN and PASSED**, asserting byte-exact
  PaliGemma tokenization vs the reference. (If `_REF_PY` were absent it would skip-gate, not fail.)
- **External tokenizer asset:** `/mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model`
  (4,264,023 bytes), sha256[:16] **`8986bb4f423f07f8`** — the vendored `pi0_model/tokenizer.py` resolves
  this default/override path; the byte-exact parity that ran uses it.

## (3) Old `openpi/` (in-repo) package untouched

```
git diff --name-only e2092e85..HEAD -- rlinf/models/embodiment/openpi/
# -> (empty: 0 files)
```

- **0 files** under `rlinf/models/embodiment/openpi/` changed across the entire Phase-3 branch
  (`e2092e85..HEAD`) — the old package is byte-for-byte unchanged, so its behavior is preserved by
  construction.
- The old package's 34 `*.py` files all parse OK. There are no dedicated old-`openpi` unit test modules in
  the repo (`tests/` has no `test_*openpi*` outside `test_openpi_pytorch_*`); the no-touch invariant is the
  AC-13 gate and it holds.

## (4) Changed-file Ruff `check` + `format --check` clean (whole Phase-3 diff)

```
git diff --name-only e2092e85..HEAD -- '*.py'   # -> 78 files (0 under old openpi/)
python -m ruff check   <all 78>                 # -> All checks passed!
python -m ruff format --check <all 78>          # -> 78 files already formatted
```

- **78 Python files** changed since the branch point (none under old `openpi/`).
- `ruff check` and `ruff format --check` over **all 78** are clean. Achieving this required a one-time
  hygiene normalization of earlier-round files that predated the per-round Ruff gate: **5 `I001`
  import-sort fixes** (`pi0_model/processing.py`, `test_openpi_pytorch_behavior_parity.py`,
  `test_openpi_pytorch_export.py`, `test_openpi_pytorch_parity_gpu.py`) and a `ruff format` pass over **28**
  format-drifted files (behavior-neutral whitespace/line-wrapping; the full suite below re-confirms
  behavior is unchanged). The R37/R38 files were already clean.

## Audit source-of-truth note (R38 advisory evidence)

The R38 advisory ~1 h trend artifact's raw tensorboard event file lived under
`/mnt/public/xzxuan/tmp/r38_advisory_sft_results/...` and was cleaned as round scratch. The **committed
`docs/evidence/r38_advisory_1h_trend.csv`** (800 per-step rows) + `...json` (trend stats + provenance) are
the audit source of truth for the DEC-1(c) advisory trend claim (both descend, Pearson r=0.975, 788/800
within 0.03 for context). The same applies to the GPU-evidenced first-50 artifacts (R37
`r37_pinned_first50.{csv,json}`, R36 `r36_pinned_residual.json`): the committed CSV/JSON + recorded hashes
are the audit artifacts per DEC-5.

## Artifact hashes (DEC-5)

- base model `pi05_base_pytorch_new/model.safetensors`: sha256[:16] `f6391204c480d6c5`
- task-0000 `norm_stats.json`: sha256[:16] `d66ed16830a98f90`
- tokenizer `paligemma_tokenizer.model`: sha256[:16] `8986bb4f423f07f8`

## Verdict

All four AC-13 invariants hold from a clean tree: the full CPU suite is green (143 passed, 1 skipped — the
skip is the heavy env-gated jax→new converter value check, a DEC-5 evidence gate, not a failure), both
byte-exact tokenizer parity tests RAN and PASSED here (their `openpi`-package / reference-venv deps are
present; they skip-gate rather than fail when absent), the old `openpi/` package is untouched (0 files),
and the whole Phase-3 Python diff is Ruff `check` + `format` clean. task18 (AC-13) is met.
