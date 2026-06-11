# Implementation Plan — pi0.5 VLM-VLA Mode (VLM Token Output) for RLinf `openpi_pytorch`

## Goal Description

Teach RLinf's pi0.5 stack (`rlinf/models/embodiment/openpi_pytorch`) to optionally let the VLM (PaliGemma backbone) output tokens, controlled by a new config `mode` with values `vla` and `vlm_vla`:

- **`vla`**: today's behavior, unchanged and numerically identical — action output only; the current tokenizer prompt format (`Task: ..., State: ...;\nAction: `) is preserved byte-for-byte.
- **`vlm_vla`**:
  - **SFT**: total loss = `language_loss_weight` × VLM cross-entropy loss (over the subtask response + EOS) + `action_loss_weight` × action-expert flow-matching loss. Both weights live in YAML (defaults 1.0 / 1.0); their quotient realizes the draft's "ratio" hyperparameter.
  - **Eval**: the VLM first autoregressively generates the subtask/reasoning text (batch size > 1 fully supported with per-row EOS handling), then the action expert denoises actions while attending to the prefix + generated-token KV cache with each row's EOS excluded.

The design is ported — ideas and semantics, not literal code — from the reference implementation `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05`: its tokenization template, the three per-token masks (`token_ar_mask`, `token_loss_mask`, `token_kv_cache_mask`), the `compute_ce_loss` branch of `modeling_pi05.py`, `generate_language`, `StaticKVCache`, `left_to_right_align`, and the knowledge-insulation stop-gradient attention split (whose mechanics live in `paligemma_with_multi_expert.py`, of which only the detach semantics are ported — single expert only).

Per-token semantics in `vlm_vla` (the draft's central table):

| mask | controls | prefix | response | EOS |
|---|---|---|---|---|
| `token_ar_mask` | attention shape (0 = bidirectional block, 1 = causal) | 0 | 1 | 1 |
| `token_loss_mask` | which positions count toward CE loss | False | True | True |
| `token_kv_cache_mask` | which tokens the action expert can see | True | True | False |

So: prefix is bidirectional, no loss, enters the KV cache; response is causal, has loss, enters the KV cache; EOS is causal, has loss, does **not** enter the KV cache (the action expert never attends to it, at train or eval time).

The landing scenario is the BEHAVIOR dataset `/mnt/public/xzxuan/data/2025-challenge-demos`, single task `turning_on_radio` with its four subtasks configured in YAML:

```yaml
task_subtasks:
  turning_on_radio:
    - "move to radio"
    - "pick up radio from coffee table"
    - "press radio"
    - "place radio on coffee table"
```

The dataset layer replaces `use_skill` with `fine_grained_level` (0 = one text item: the main task; 1 = two text items: main task as input plus the subtask string as the VLM response label), deletes `allow_left`/`allow_right` and all logic involving them, redefines `enable_gap` (True → gap frames are assigned to the next skill; False → gap frames are not used; YAML default True), and keeps streaming loading. Subtask labels come from each episode's `skill_annotation` (in `annotations/task-*/episode_*.json`) mapped through `task_subtasks` — `use skill` semantics exist only in `vlm_vla`, never in `vla`.

Out of scope per the draft: the `fast` tokenizer, value prediction, `modeling_critic.py`, the multi-expert architecture (we have only one expert), and `fine_grained_level` ≥ 2 (reserved for later levels 2, 3, …).

Execution environment: use `/mnt/public/xzxuan/.venv_pi/bin/python` as the interpreter; use `/mnt/public/xzxuan/tmp` for temporary files and test outputs.

## Acceptance Criteria

Following TDD philosophy, each criterion includes positive and negative tests for deterministic verification.

- AC-1: Both `fine_grained_level` values are implemented in the BEHAVIOR SFT dataset, and the text attached to streamed frames is verified frame-by-frame.
  - AC-1.1: `fine_grained_level = 0` yields exactly one text item — the pi0.5 input, which is the main task.
    - Positive Tests (expected to PASS):
      - Streaming iteration at level 0 yields items whose prompt is the main-task text and which carry no subtask/response field; collation produces the same `tokenized_prompt` / `tokenized_prompt_mask`-only batch contract as today.
      - Level 0 works without `task_subtasks` configured.
    - Negative Tests (expected to FAIL/be rejected):
      - Any level-0 item carrying a response/subtask string fails the verification script.
  - AC-1.2: `fine_grained_level = 1` yields two text items — the main task (input) and the subtask (VLM response label) — resolved deterministically from `skill_annotation`.
    - Positive Tests (expected to PASS):
      - Items carry prompt = main task AND response = `task_subtasks[task][skill_idx]`.
      - For `episode_00000030` (windows `[0,211]`, `[666,999]`, `[999,1408]`, `[1416,2038]`) with `enable_gap: true`: frame 100 → "move to radio"; gap frames 211–665 → "pick up radio from coffee table" (the next "pick up" segment, per the draft's example); frame 700 → "pick up radio from coffee table"; frame 1200 → "press radio"; gap frames 1408–1415 → "place radio on coffee table"; frame 1500 → "place radio on coffee table".
      - Resolution is relative to the annotation's `valid_duration = [valid_start, valid_end)`: for an episode with nonzero `valid_start` (e.g. `episode_00002150` with `valid_duration [262, 2170]`), the leading gap `[valid_start, first_skill_start)` maps to skill 0 when `enable_gap: true`.
      - A streamed-frames verification script prints frame → (prompt, response) on real data and the mapping holds for both `enable_gap` settings.
    - Negative Tests (expected to FAIL/be rejected):
      - With `enable_gap: false`, gap frames (e.g. frame 300 of `episode_00000030`) are never yielded for training.
      - Trailing-gap frames `[last_skill_end, valid_end)` are never yielded regardless of `enable_gap` (no next skill exists).
      - Frames outside `valid_duration` are never yielded.
      - An episode whose `skill_idx` exceeds the bounds of `task_subtasks[task]`, has duplicate `skill_idx`, or has overlapping windows aborts dataset construction with an error naming the episode.
  - AC-1.3: Config surface migration is complete and validated.
    - Positive Tests (expected to PASS):
      - `data.fine_grained_level` ∈ {0, 1} and `data.enable_gap` (default `true`) are honored; example YAMLs load; streaming chunked loading and rank/worker partitioning behave as before.
    - Negative Tests (expected to FAIL/be rejected):
      - `use_skill`, `allow_left`, `allow_right` are no longer accepted dataset parameters (passing them fails loudly rather than being silently ignored).
      - `fine_grained_level: 1` with `mode: vla` is rejected at startup ("use skill" is only allowed in `vlm_vla`).
      - `mode: vlm_vla` with `fine_grained_level: 0` is rejected at startup (nothing for the CE loss to supervise).
      - `skill_list` other than absent/`["all"]` is rejected (its weighting source is removed with the orchestrator machinery).

- AC-2: `vlm_vla` SFT training runs on the behavior task at `fine_grained_level = 1` and the loss goes down.
  - AC-2.1: Combined loss and metrics.
    - Positive Tests (expected to PASS):
      - A short SFT run on `turning_on_radio` logs `loss`, `language_loss`, `action_loss`, and `language_acc`; both component losses show a clear decreasing trend (per the draft, "see the loss go down" is a trend requirement, not a numeric threshold); a checkpoint is saved.
      - The total equals `language_loss_weight * ce + action_loss_weight * flow` for the configured weights.
    - Negative Tests (expected to FAIL/be rejected):
      - Perturbing token targets at `token_loss_mask = False` positions (prefix, padding) leaves the CE loss unchanged — any leakage fails the unit test.
      - `mode: vlm_vla` at level 1 without `task_subtasks` fails at startup.
  - AC-2.2: `stop_gradient_to_vlm` works as specified.
    - Positive Tests (expected to PASS):
      - With the flag `true`, backward of the flow loss alone leaves every VLM (PaliGemma expert) parameter with zero/absent gradient while action-expert parameters receive gradients; backward of the CE loss alone populates gradients on the VLM's Q/K/V/O/MLP parameters.
      - With the flag `false` (default), the flow loss reaches VLM parameters as well.
    - Negative Tests (expected to FAIL/be rejected):
      - With the flag `true`, any nonzero flow-gradient on a VLM parameter fails the autograd probe test.

- AC-3: A trained checkpoint is saved, evaluated, and its intermediate outputs inspected — the model reasonably outputs the subtask (including EOS).
  - Positive Tests (expected to PASS):
    - The inspection script in `toolkits/eval_scripts_openpi` loads the AC-2 checkpoint, runs batched eval (batch size > 1, mixed prompts), decodes and prints each sample's generated subtask text and EOS position, and prints the denoised actions with correct shapes.
    - The trained model's generations reasonably match the four subtask strings and terminate with EOS within `max_new_tokens` (greedy decoding, unconstrained).
    - The standard `predict_action_batch` env-eval path produces actions in `vlm_vla` mode, so normal rollout evaluation works.
  - Negative Tests (expected to FAIL/be rejected):
    - The un-finetuned base pi0.5 checkpoint fails the "reasonable subtask + EOS" inspection (demonstrating the check is not vacuous).
    - Rows that exhaust `max_new_tokens` without EOS are reported as non-terminated rather than silently passed.

- AC-4: Attention, masks, and KV-cache handling are deeply verified in both SFT and eval — concrete inputs are provided, outputs printed, and every stage confirmed (the draft's central emphasis).
  - AC-4.1: Token mask construction.
    - Positive Tests (expected to PASS):
      - Unit tests assert the mask table: prefix (BOS + `Task: {task}. State: {state}. Subtask: ` template) → ar 0 / loss False / kv True; response → ar 1 / loss True / kv True; EOS → ar 1 / loss True / kv False; right padding → input mask False, ar 0, loss False, kv False.
      - Eval-time tokenization emits the prefix-only variant of the same masks.
    - Negative Tests (expected to FAIL/be rejected):
      - Any deviation from the table (EOS with kv True, response with ar 0, loss on prefix or padding) fails the unit tests.
  - AC-4.2: SFT attention and CE mechanics.
    - Positive Tests (expected to PASS):
      - For a toy batch, the printed attention matrix shows the prefix as one bidirectional block, response tokens causal (cumsum block semantics of `make_attn_mask`), and suffix (action) tokens attending only to kv-True prefix positions plus themselves.
      - CE uses the next-token shift (targets are tokens shifted by one; the language offset is computed from tensor shapes, not a hardcoded image-token count) with per-sample loss-mask normalization.
    - Negative Tests (expected to FAIL/be rejected):
      - A suffix query attending an EOS column fails the assertion.
      - Padded-position logits influencing the CE value fails the unit test.
  - AC-4.3: Eval generation and KV cache.
    - Positive Tests (expected to PASS):
      - With batch size > 1, different prompt lengths, and different per-row EOS steps, printed per-stage outputs confirm: right-alignment (all rows share the same static-cache write column each step), per-row position ids, per-row validity masks excluding each row's EOS and post-EOS columns, and denoise attention consuming the validity mask.
    - Negative Tests (expected to FAIL/be rejected):
      - Any row whose EOS or post-EOS column is visible to denoise attention fails the per-row assertion.
      - Cache writes past the preallocated length raise.
  - AC-4.4: `vla` regression.
    - Positive Tests (expected to PASS):
      - With `mode: vla` (default), `embed_prefix` mask behavior, the `sample_actions` prefix-cache flow, the tokenizer output, and the SFT loss are unchanged; existing tests and a pinned-behavior regression test pass.
    - Negative Tests (expected to FAIL/be rejected):
      - The `vla` path reading any of the new mask fields or the static cache fails the guard test.

- AC-5: Code changes are minimal, clean, and contained.
  - Positive Tests (expected to PASS):
    - Core changes live only under `rlinf/models/embodiment/openpi_pytorch` and `rlinf/data/datasets/openpi_pytorch`; additional allowed touchpoints are example YAML configs, tests, the inspection script in `toolkits/eval_scripts_openpi`, the EN/ZH docs that currently document `use_skill`, and the single user-sanctioned generic scalar-metric block in `fsdp_vla_sft_worker`.
    - The two modes share code paths wherever reasonable: one tokenizer class, one `compute_loss` entry with a `vlm_vla` branch (mirroring the reference's `compute_ce_loss` branch), one `Attention` implementation with flag-guarded extensions, one `predict_action_batch` with a `vlm_vla` branch.
  - Negative Tests (expected to FAIL/be rejected):
    - Changes to other workers/runners/envs, or a duplicated parallel model class for the new mode, are rejected in review.

## Path Boundaries

Path boundaries define the acceptable range of implementation quality and choices.

### Upper Bound (Maximum Acceptable Scope)

The implementation delivers the full faithful port: the deterministic `valid_duration`-relative frame resolver with unit tests (including a nonzero-`valid_start` case); the dataset refactor with iterative (non-recursive) frame skipping and hard validation; the `vlm_vla` tokenizer path with all three masks plus `eos_token_id`/`decode` exposure; `Observation` extended with `token_kv_cache_mask` and per-batch mask plumbing through every touchpoint (`from_dict`, dtype/device transfer, preprocessing, collate, eval processor); the CE branch via the tied embedder decode with two YAML loss weights and a `language_acc` metric; the kv-cache-mask suffix blocking applied consistently at train and eval; the per-layer split-attention `stop_gradient_to_vlm`; an adapted `StaticKVCache` in RLinf's layout plus a `left_to_right_align` port powering batch-safe greedy generation with per-row EOS exclusion; the `predict_action_batch` `vlm_vla` branch returning decoded text alongside actions; the inspection script; config validation; the factory passing `max_token_len` explicitly; EN/ZH docs updates; autograd probe tests, mask-table tests, and `vla` regression guards.

### Lower Bound (Minimum Acceptable Scope)

All five acceptance criteria are satisfied with the user-confirmed decisions: deterministic resolver with the confirmed gap rules; level 0/1 dataset contract verified on real data; `vlm_vla` SFT with the two-weight combined loss and a functional `stop_gradient_to_vlm`; static-cache, right-aligned, batch-safe generation with per-row EOS exclusion; checkpoint trained, evaluated, and inspected; printed stage-by-stage mask/KV verification; `vla` numerics untouched. The draft plus user decisions make the design largely deterministic — the bounds differ mainly in test breadth and documentation polish, not in features.

### Allowed Choices

- Can use: internal naming and module placement within the allowed packages (e.g. the frame resolver as a module-level function or a small module in the behavior dataset package); static-cache class internals, provided the per-layer layout matches RLinf's `[batch, sequence, kv_heads, head_dim]` convention and the `vla` tuple path is untouched; the mechanism by which `Attention`/`Module` expose the optional static-cache path (extra parameter vs. small handle object); the `max_new_tokens` default; unit-test file organization; an optional sampling-temperature knob (default 0 = greedy, which is the acceptance path).
- Cannot use: any change to `vla`-mode numerics or its tokenizer string; core logic outside `rlinf/models/embodiment/openpi_pytorch` and `rlinf/data/datasets/openpi_pytorch`; worker modifications beyond the sanctioned metric block; HuggingFace `transformers` cache classes inside the vendored model (it stays self-contained); reintroduction of `allow_left`/`allow_right`/`use_skill`; orchestrator-based labels for level 1; silently skipping or clipping invalid annotations (must hard-error).

> **Note on Deterministic Designs**: The draft pins most of the design (mask table, template, mode semantics, dataset rules), and the user confirmed the remaining choices during plan review (two loss weights; static-cache port now; unconstrained greedy decoding; gap edge rules; hard-error config matrix; script location; the single worker touch). The boundaries above are correspondingly narrow.

## Feasibility Hints and Suggestions

> **Note**: This section is for reference and understanding only. These are conceptual suggestions, not prescriptive requirements.

### Conceptual Approach

1. **Dataset**: a pure function maps a frame index to a skill index or a skip decision, given the episode's `skill_annotation` windows (half-open `[start, end)`, sorted by start), `valid_duration`, and `enable_gap`. Rules: frames outside `[valid_start, valid_end)` are rejected before resolution; a frame inside window i belongs to skill i; the gap between windows i and i+1 belongs to skill i+1 when `enable_gap` else skipped; the leading gap `[valid_start, start_0)` belongs to skill 0 when `enable_gap` else skipped; the trailing gap `[end_last, valid_end)` is always skipped. `BehaviorSftDataset` calls this resolver; the orchestrator indirection (`load_orchestrators`, `load_orchestrators_data`, `_get_fine_grained_task`) and the window-extension/random-overlap logic (`_build_skill_boundaries`, `_get_skill_label`, `allow_left`, `allow_right`) are removed — with this data they only ever synthesize full-task fallbacks. Skipped frames advance via an iterative loop, not recursion (gaps can span hundreds of frames). At level 1, the item carries the main task as `prompt` and the resolved subtask as `response`.
2. **Tokenizer**: a `vlm_vla` method builds the prefix `BOS + "Task: {task}. State: {state}. Subtask: "` (cleaned text, trailing punctuation stripped before adding periods, single-space joins, trailing space), the response `"{subtask}."` encoded without special tokens, and the EOS token; emits `(tokens, input_mask, token_ar_mask, token_loss_mask, token_kv_cache_mask)` right-padded to `max_token_len`; exposes `eos_token_id` and `decode`. SFT uses prefix+response+EOS; eval uses prefix only.
3. **Observation plumbing**: add `token_kv_cache_mask`; treat all three token masks as per-batch `[B, L]` tensors threaded through construction, dtype/device transfer, preprocessing, collation, and the eval processor.
4. **SFT forward**: one unified prefix+suffix forward as today. The prefix embedding consumes the per-batch ar mask (images stay bidirectional); prefix and suffix ar masks are normalized to `[B, ·]` before concatenation. CE: take the language slice of the prefix hidden states (offset computed from shapes), project through the tied embedder decode, shift targets by one, mask with the loss mask, normalize per sample, average over the batch; also report token accuracy. The suffix→prefix attention is additionally blocked wherever `token_kv_cache_mask` is False, so the action expert never sees EOS — identical semantics at train and eval. Total loss combines the two terms with the YAML weights; the wrapper returns a dict (`loss`, `action_loss`, `language_loss`, `language_acc`), and the SFT worker generically logs scalar metrics from that dict.
5. **Stop-gradient**: when `stop_gradient_to_vlm` is true, each attention layer splits: prefix queries attend prefix K/V with gradients on (CE fully trains the VLM's Q/K/V/O/MLP); suffix queries attend the concatenation of detached prefix K/V and suffix K/V (the flow loss cannot reach VLM weights at any layer); the two outputs are concatenated back. Flag off → the existing unified attention runs unchanged.
6. **Generation**: preallocate per-layer K/V buffers sized prefill + `max_new_tokens` (RLinf layout); right-align each row's valid prefix tokens (`left_to_right_align` port) so every row writes the same cache column each step; prefill, then greedy-decode with per-row done flags. Finished rows keep performing masked dummy writes at the shared column, but a per-row column-validity mask marks columns at/after that row's EOS invalid; attention during generation and during the subsequent denoise consumes that validity mask. Positions continue per row from its valid length. After generation, run the standard denoise loop against the cache, then post-process actions; return decoded text and token ids alongside actions.
7. **Factory**: pass `openpi.max_token_len` explicitly into the model config instead of relying on its default.

### Relevant References

- `rlinf/models/embodiment/openpi_pytorch/openpi_action_model.py` — wrapper: `sft_forward` (returns a scalar today), `predict_action_batch` + eval-processor wiring.
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/pi0.py` — `embed_prefix` / `embed_suffix` / `compute_loss` / `sample_actions` / `make_attn_mask` (cumsum block semantics).
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/gemma.py` — `Attention` (per-expert Q/K/V/O, tuple-concat KV cache), `Module` (per-layer cache threading), `Embedder.decode` (tied logits projection for CE).
- `rlinf/models/embodiment/openpi_pytorch/pi0_model/model.py` — `Observation` dataclass (`token_ar_mask`/`token_loss_mask` exist but are unused today), `from_dict`, preprocessing.
- `rlinf/models/embodiment/openpi_pytorch/utils/tokenizer.py` — `PaligemmaTokenizer` (the `vla` prompt string lives here and must stay byte-identical).
- `rlinf/models/embodiment/openpi_pytorch/__init__.py` — model factory and `openpi.*` config ingestion (`max_token_len` is currently not passed through).
- `rlinf/data/datasets/openpi_pytorch/behavior/behavior_sft_dataset.py` — streaming dataset; `load_annotations` (the `skill_annotation` source that stays); the `use_skill` / `allow_left` / `allow_right` / orchestrator logic being replaced.
- `rlinf/data/datasets/openpi_pytorch/behavior/behavior_sft_data_loader.py` — transform + collate (today carries only `tokenized_prompt` and its mask).
- `rlinf/data/datasets/openpi_pytorch/behavior/processing.py` — `BehaviorEvalProcessor.build_observation` (eval-time tokenization).
- `rlinf/workers/sft/fsdp_vla_sft_worker.py` — `get_train_model_output` metric extraction (the single sanctioned worker touch).
- `examples/sft/config/behavior_pi05_vla.yaml` — current SFT config (`data.*` and `actor.model.openpi.*` sections, including `task_subtasks`).
- `docs/source-en/rst_source/examples/embodied/sft_openpi_pytorch.rst` and its ZH counterpart — currently document `use_skill`.
- Reference implementation `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05/`: `modeling_pi05.py` (the `compute_ce_loss` branch, `generate_language`, suffix blocking by kv-cache mask), `processing_pi05.py` (template and per-token mask construction), `static_kv_cache.py` (`StaticKVCache`, `left_to_right_align`), `configuration_pi05.py` (`language_loss_weight` / `action_loss_weight` / `stop_gradient_to_vlm` naming), `paligemma_with_multi_expert.py` (per-layer detach semantics only), `data_collator_pi05.py`, `README.md` (mask table). Note: its generation stops only when all rows emit EOS and tracks a scalar cache counter — the port must be per-row safe instead.
- Data: `/mnt/public/xzxuan/data/2025-challenge-demos/annotations/task-0000/episode_*.json` — `skill_annotation` entries (`skill_idx`, `frame_duration`) and `meta_data.valid_duration` (nonzero starts exist, e.g. `episode_00002150`).

## Dependencies and Sequence

### Milestones

1. Milestone M1 — Dataset and resolver (AC-1):
   - Phase A: pure frame→subtask resolver + unit tests (including `episode_00000030` windows, a nonzero-`valid_start` case, both `enable_gap` settings, validation errors).
   - Phase B: `BehaviorSftDataset` refactor (remove `use_skill`/`allow_left`/`allow_right`/orchestrators; new `enable_gap`; iterative skipping; level 0/1 item contract; `skill_list` restriction; config validation).
   - Phase C: streamed-frames verification script run on real data at both levels.
2. Milestone M2 — Tokenization and mask plumbing (AC-4.1): `vlm_vla` tokenizer path; `Observation` + full plumbing; transform/collate/eval-processor wiring; factory `max_token_len` pass-through.
3. Milestone M3 — SFT losses (AC-2, AC-4.2): CE branch + two weights + `language_acc`; kv-cache-mask suffix blocking; split-attention `stop_gradient_to_vlm`; dict return + worker generic metrics; autograd probes + `vla` regression guards.
4. Milestone M4 — Generation and eval (AC-3, AC-4.3): `StaticKVCache` + `left_to_right_align` ports; per-row-EOS `generate_language`; `predict_action_batch` `vlm_vla` branch; per-stage printed assertions.
5. Milestone M5 — Configs and docs (AC-1.3, AC-5): `vlm_vla` example YAML + cross-field validation; EN/ZH docs migration from `use_skill`.
6. Milestone M6 — End-to-end acceptance (AC-2, AC-3, AC-4): SFT training run, checkpoint, batched eval inspection, independent deep audits.

Dependencies: M2 depends on M1's item contract; M3 and M4 both depend on M2; M4's denoise reuses M3's kv-mask blocking for train/eval consistency; M5 consolidates the surface of M1–M4; M6 depends on all prior milestones.

## Task Breakdown

Each task must include exactly one routing tag:
- `coding`: implemented by Claude
- `analyze`: executed via Codex (`/humanize:ask-codex`)

| Task ID | Description | Target AC | Tag (`coding`/`analyze`) | Depends On |
|---------|-------------|-----------|----------------------------|------------|
| task1 | Pure frame→subtask resolver (valid_duration-relative, confirmed gap rules) + unit tests | AC-1.2 | coding | - |
| task2 | BehaviorSftDataset refactor: remove use_skill/allow_left/allow_right/orchestrators, new enable_gap semantics, iterative skipping, level 0/1 item contract, skill_list restriction, load-time validation | AC-1.1, AC-1.2, AC-1.3 | coding | task1 |
| task3 | Streamed-frames verification script + run on real data (both levels, both enable_gap settings) | AC-1 | coding | task2 |
| task4 | vlm_vla tokenizer path (template, three masks, eos_token_id/decode), Observation token_kv_cache_mask + per-batch mask plumbing (from_dict/dtype/device/preprocess), transform/collate/eval-processor wiring, factory max_token_len pass-through | AC-4.1 | coding | task2 |
| task5 | SFT vlm_vla loss branch: CE via tied embedder decode (shape-derived offset, shift, per-sample normalization), [B, ·] mask normalization at the concat site, kv-cache-mask suffix blocking, two YAML weights, dict return, worker generic scalar metrics | AC-2.1, AC-4.2 | coding | task4 |
| task6 | stop_gradient_to_vlm per-layer split attention in gemma Attention + autograd probe tests + vla regression guards | AC-2.2, AC-4.4 | coding | task5 |
| task7 | StaticKVCache (RLinf layout) + left_to_right_align ports; generate_language with per-row EOS and masked dummy writes; predict_action_batch vlm_vla branch returning text + actions | AC-3, AC-4.3 | coding | task4, task5 |
| task8 | Configs: behavior pi0.5 vlm_vla example YAML, cross-field validation (mode/level matrix), EN/ZH docs migration | AC-1.3, AC-5 | coding | task2, task5, task7 |
| task9 | Mask/attention/cache unit tests: mask table, make_attn_mask interplay, CE shift, padded-logit isolation, per-row EOS exclusion, cache-overflow raise | AC-4 | coding | task4, task5, task7 |
| task10 | SFT training run at level 1 on turning_on_radio: loss curves for both components, checkpoint saved | AC-2 | coding | task8, task9 |
| task11 | Inspection script in toolkits/eval_scripts_openpi: load checkpoint, batched eval (B > 1), print generated subtasks/EOS/masks/cache stages/actions; base-checkpoint negative check | AC-3, AC-4.3 | coding | task7, task10 |
| task12 | Independent deep audit of attention/masks/KV-cache handling in SFT and eval against the reference design and the mask table | AC-4 | analyze | task9, task11 |
| task13 | Final minimality and path-boundary review of the complete diff | AC-5 | analyze | task12 |

## Claude-Codex Deliberation

### Agreements

- Dataset-first sequencing isolates the most ambiguous data semantics before model work.
- `vla` mode must stay byte-compatible, including the existing tokenizer prompt string.
- `Observation` needs `token_kv_cache_mask` added; `token_ar_mask`/`token_loss_mask` already exist but are unused and unpopulated today.
- CE logits come from the tied embedder decode (the vendored model has no separate lm_head).
- The reference's generation is batch-unsafe (stops only when all rows emit EOS; scalar cache counter); the port must implement per-row EOS handling — a deliberate improvement, since the draft requires batch size > 1 at eval.
- `predict_action_batch` is the right eval venue, complemented by an in-repo inspection script.
- The full mask plumbing spans tokenizer → transform → collate → `Observation` → dtype/device transfer → preprocessing → eval processor.
- The static cache and split attention are implementable because RLinf's cache and attention logic are localized in the vendored `Attention`/`Module`.
- A deterministic resolver replaces the current window-extension/random-overlap skill logic; frame skipping becomes iterative.
- A generic scalar-metric block in the SFT worker is the right minimal worker touch (cleaner than aliasing hardcoded metric names).

### Resolved Disagreements

- `stop_gradient_to_vlm` mechanism: Claude proposed the faithful per-layer split-attention detach; first-pass Codex floated a two-pass approximation as lower-risk. Resolution: faithful split attention — second-pass Codex agreed it is conceptually sound, the touched file is inside the allowed surface, and the draft's stated semantics (CE fully trains the VLM's Q/K/V/O/MLP while flow gradients reach only the action expert) are exactly what the split achieves.
- Worker metrics: alias the CE metrics into existing hardcoded keys (zero worker change) vs. a tiny generic worker change. Resolution: tiny generic change — Codex judged it "reasonable and probably cleaner than aliasing"; the user sanctioned it as the single allowed worker touch.
- Static cache timing: defer to a tuple-concat prototype vs. port now. Resolution: port `StaticKVCache` + `left_to_right_align` now — the draft's requirement is binding, Codex confirmed implementability in RLinf's layout, and the user confirmed.
- Orchestrator removal rationale: Claude initially argued "the source files don't exist"; Codex corrected that `annotations/` does exist and is loaded today — only the orchestrator indirection synthesizes fallbacks. Resolution: keep `load_annotations` as the level-1 label source; deliberately remove the orchestrator machinery with that corrected rationale.
- `skill_list` interplay: Claude initially called it orthogonal; Codex showed its weighting reads the orchestrator data being removed. Resolution: hard-error on any `skill_list` other than absent/`["all"]`.
- Mask shapes: Claude initially relied on broadcasting; Codex showed the suffix ar mask is concatenated with the prefix mask and must match. Resolution: all token masks are `[B, L]`, with explicit normalization to `[B, ·]` at the concatenation site.
- Mixed-EOS cache semantics: Codex required a concrete design instead of "rows stop contributing". Resolution: masked dummy writes — every row writes the shared (right-aligned) column each step; a per-row column-validity mask invalidates columns at/after that row's EOS; acceptance asserts EOS exclusion per row.
- Resolver frame range: Claude's rules were relative to raw episode frames; Codex verified real episodes with nonzero `valid_start` (e.g. `episode_00002150`, `valid_duration [262, 2170]`). Resolution: all rules are `valid_duration`-relative, with a dedicated nonzero-`valid_start` unit test.

### Convergence Status

- Final Status: `converged` — three review rounds; the final round returned zero disagreements and zero required changes, plus one optional improvement (explicit `max_token_len` pass-through in the factory), which is folded into the plan.

## Pending User Decisions

All decisions were resolved with the user during plan review; none remain pending.

- DEC-1: Meaning of the truncated draft note #6 ("Make sure …")
  - Claude Position: likely an editing leftover or an emphasis lead-in; ask rather than guess.
  - Codex Position: N/A - open question
  - Tradeoff Summary: a lost requirement here could silently change scope.
  - Decision Status: It leads into the following emphasis line — "Focus heavily on the design of attention, masks, and the KV cache." No new scope; AC-4 is the corresponding criterion.
- DEC-2: Loss-combination knob shape
  - Claude Position: two weights (`language_loss_weight`, `action_loss_weight`), matching the reference config.
  - Codex Position: surfaced both options; noted the draft's wording says "a ratio".
  - Tradeoff Summary: two weights allow independent ablation and reference parity; a single ratio is one fewer knob and matches the draft literally.
  - Decision Status: Two weights — `language_loss_weight` + `action_loss_weight`, defaults 1.0/1.0; their quotient is the draft's ratio.
- DEC-3: Static cache mandate
  - Claude Position: port `StaticKVCache` + `left_to_right_align` in the first implementation (draft requirement binding).
  - Codex Position: initially preferred a tuple-concat prototype first; round 2 asked for an explicit mandate or an explicit user downgrade.
  - Tradeoff Summary: faithful port honors the draft and avoids per-step concat churn; tuple-concat is simpler but deviates from the stated requirement.
  - Decision Status: Port the static cache and right-alignment now, flag-guarded so the `vla` path is untouched.
- DEC-4: Eval decoding policy
  - Claude Position: unconstrained greedy until EOS or `max_new_tokens`.
  - Codex Position: asked whether output should be constrained to the four subtask strings.
  - Tradeoff Summary: unconstrained decoding reveals what the model actually learned (what AC-3 inspects); constrained decoding guarantees well-formed output but hides failure modes and adds machinery.
  - Decision Status: Unconstrained greedy (temperature 0).
- DEC-5: Gap and boundary semantics
  - Claude Position: half-open `[start, end)` windows; `valid_duration`-relative; leading gap → skill 0 when `enable_gap`; trailing gap always skipped; gap between skills → next skill (draft-specified).
  - Codex Position: same, after contributing the `valid_duration` correction.
  - Tradeoff Summary: a stricter alternative would skip the leading gap too.
  - Decision Status: All rules confirmed as stated.
- DEC-6: `mode: vlm_vla` with `fine_grained_level: 0`
  - Claude Position: hard error (nothing for CE to supervise).
  - Codex Position: N/A - open question
  - Tradeoff Summary: hard error keeps the mode/level matrix unambiguous; allowing it with CE disabled blurs the modes.
  - Decision Status: Hard error at startup.
- DEC-7: Inspection-script location
  - Claude Position: in-repo for reproducibility.
  - Codex Position: in-repo (`toolkits/` or `tests/manual/`), not `/tmp`.
  - Tradeoff Summary: in-repo is reproducible; `/tmp`-only minimizes repo surface but loses the acceptance workflow.
  - Decision Status: `toolkits/eval_scripts_openpi`, with outputs written to `/mnt/public/xzxuan/tmp`.
- DEC-8: Worker modification allowance
  - Claude Position: a few-line generic scalar-metric extraction in `fsdp_vla_sft_worker` is "absolutely necessary" for verifying AC-2.
  - Codex Position: agreed ("reasonable and probably cleaner than aliasing").
  - Tradeoff Summary: without it, CE metrics are either invisible or logged under misleading hardcoded names.
  - Decision Status: Allowed — the single sanctioned worker touch.

## Implementation Notes

### Code Style Requirements

- Implementation code and comments must NOT contain plan-specific terminology such as "AC-", "Milestone", "Step", "Phase", or similar workflow markers
- These terms are for plan documentation only, not for the resulting codebase
- Use descriptive, domain-appropriate naming in code instead

### Repository Conventions

- Google Python style; Ruff lint/format; docstrings and type hints on public APIs; new files need the repository's standard license/copyright header (CI enforces this via `ruff --preview` CPY001 — reproduce locally with `pre-commit run --all-files`).
- Logging via `rlinf.utils.logging.get_logger()` or workers' `self.log_*`; no `print` in library code (the inspection script's purpose is printing and is exempt).
- Conventional Commits with `Signed-off-by` (`git commit -s`); config YAMLs hold static values only.
- Use `/mnt/public/xzxuan/.venv_pi/bin/python` for all runs; write temporary files and test outputs to `/mnt/public/xzxuan/tmp`.

## Output File Convention

This template is used to produce the main output file (e.g., `plan.md`).

### Translated Language Variant

When `alternative_plan_language` resolves to a supported language name through merged config loading, a translated variant of the output file is also written after the main file. Humanize loads config from merged layers in this order: default config, optional user config, then optional project config; `alternative_plan_language` may be set at any of those layers. The variant filename is constructed by inserting `_<code>` (the ISO 639-1 code from the built-in mapping table) immediately before the file extension:

- `plan.md` becomes `plan_<code>.md` (e.g. `plan_zh.md` for Chinese, `plan_ko.md` for Korean)
- `docs/my-plan.md` becomes `docs/my-plan_<code>.md`
- `output` (no extension) becomes `output_<code>`

The translated variant file contains a full translation of the main plan file's current content in the configured language. All identifiers (`AC-*`, task IDs, file paths, API names, command flags) remain unchanged, as they are language-neutral.

When `alternative_plan_language` is empty, absent, set to `"English"`, or set to an unsupported language, no translated variant is written. Humanize does not auto-create `.humanize/config.json` when no project config file is present.

--- Original Design Draft Start ---

# Implementing VLM Token Output for pi0.5 (VLM-VLA Mode)

## Overview

I want you to implement, on top of the current pi0.5, the ability for the **VLM part to output tokens**. Concretely, this means making the `eval` and `SFT` paths of the current `rlinf/models/embodiment/openpi_pytorch` support **two modes** (add a `mode` config), choosing between `vla` and `vlm_vla`:

- **`vla`**: the current code's `eval` and `SFT` behavior — **action output only**.
- **`vlm_vla`**: the new logic you implement.
  - For **SFT**: account for both the **VLM CE loss** and the **action expert flow-matching loss**.
  - For **eval**: support having the **VLM first output reasoning**, and then the **action expert output the action**.

This work has **two parts**: (1) model-structure support, and (2) the concrete landing scenario.

---

## Part 1 — Model Structure Support

> **Important!** A reference implementation already exists at `/mnt/public/xzxuan/repos/vla_lib`. Focus especially on the code under `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05`. I need you to be **very strict and very careful** in fully understanding its entire design! Then **port its ideas and design into RLinf's `openpi_pytorch`**.

### Requirements in `vlm_vla` mode

1. **Tokenizer** changes to the format:
   ```
   Task: ..  State: ..  Subtask: ... 
   ```

2. **Carefully reason about** the prefix, the response (reasoning), and the EOS with respect to **attention mask**, **loss computation**, and **whether they enter the KV cache**. Following the reference code:
   - **prefix**: bidirectional, no loss, enters KV cache
   - **response**: causal, has loss, enters KV cache
   - **EOS**: causal, has loss, does **not** enter KV cache

3. **Each token has** `token_ar_mask`, `token_loss_mask`, and `token_kv_cache_mask`:

   | mask | controls what | prefix | response | EOS |
   |---|---|---|---|---|
   | `token_ar_mask` | attention shape (0 = bidirectional block, 1 = causal) | `0` | `1` | `1` |
   | `token_loss_mask` | which positions count toward CE loss | `False` | `True` | `True` |
   | `token_kv_cache_mask` | which tokens the action expert can see | `True` | `True` | `False` |

4. **Add an SFT config: `stop_gradient_to_vlm`.** If `True`, the CE loss can still fully train the VLM's **Q/K/V/O/MLP**, while the gradient of the flow loss flows **only into the action expert** and does **not pollute** the VLM's semantic representations.

5. **When generating tokens, pay attention to how to make good use of `StaticKVCache` and `left_to_right_align`.**

6. In `rlinf/models/embodiment/openpi_pytorch/openpi_action_model.py`, the code for the two modes should **reuse as much as possible**. For example:
   - For **SFT training**, add a branch: if `mode == vlm_vla`, mirror the logic at line **794** (`if compute_ce_loss:`) of `/mnt/public/xzxuan/repos/vla_lib/vla_lib/models/vlas/openpi05/modeling_pi05.py`.
   - For **eval**, also reuse as much as possible: if `vlm_vla`, add an `if`, then go through the `generate_language` function, update the KV cache, handle EOS, etc., and then run the **action expert denoise** to generate actions.

### Things to note!

1. During **eval, batch size must be allowed to be greater than 1!**
2. **`fast` tokenizer** — ignore it.
3. **Value prediction** — ignore it.
4. **`modeling_critic.py`** — ignore it.
5. **`paligemma_with_multi_expert`** — don't consider it; we have **only one expert**.
6. Make sure 

> Focus heavily on the design of **attention, masks, and the KV cache**.

---

## Part 2 — The Concrete Landing Scenario

Whether `eval` or `SFT`, we still uniformly use the **behavior** dataset. For `eval`, it's reasoning first. Below I focus on **SFT training**.

To let the VLM of the SFT'd model know what to output, we need a **subtask label**.

For the SFT dataset we still use `/mnt/public/xzxuan/data/2025-challenge-demos`, except that when training `vlm_vla`, we now need to **use skill**. The current `use skill` implementation has problems, so you need to **refactor the entire `use skill`**.

- The previous approach was presumably pure `vla` mode: the VLA input was **one of four skills**, and it output an action.
- Now:
  1. `use skill` is **only allowed in `vlm_vla`**, not in `vla`.
  2. In `vlm_vla` mode, the usage is: the **input is still the main task**, and the **VLM outputs the subtask**.

For now we still consider only the single task **"turn on radio"**. This task's subtask is a **four-way choice**:

```yaml
task_subtasks:
  turning_on_radio:
    - "move to radio"
    - "pick up radio from coffee table"
    - "press radio"
    - "place radio on coffee table"
```

The **SFT loss has two parts**: (1) the subtask **CE loss**, and (2) the action expert **flow-matching loss**. There is a **ratio** between the two — make this a **hyperparameter placed in the YAML!**

### Changes to `enable_gap`, `allow_left`, `allow_right`

I want to change how they are used:

1. **`allow_left`, `allow_right`: delete them**, along with all logic that involves them.
2. **`enable_gap`** logic changes to: if `True`, **all gap frames are assigned to the next skill**; if `False`, **gap frames are not used**.
   - Example with `/mnt/public/xzxuan/data/2025-challenge-demos/annotations/task-0000/episode_00000030.json`: frames `0–211` are labeled `skillidx=0`, so these frames are the **"move to radio"** subtask. Then frames `211–666` in this JSON are **gap frames**; `enable_gap` selects whether to use them. If used, they belong to the **next "pick up" segment**. If not, these frames are **not used for training**.
   - Put this config in the YAML, **default `True`**.

### Replace `use_skill` config with `fine_grained_level`

As stated above, `use skill` is not used in `vla` mode and is used in `vlm_vla`. But in the config, **remove `use_skill`** and use **`fine_grained_level`** instead:

- **`fine_grained_level = 0`**: corresponds to the current situation, i.e. `use skill = False`.
- **`fine_grained_level = 1`**: the `vlm_vla` behavior described above.

I will later think about deeper levels **2, 3, ...** (but that's for later).

### Dataset refactor

You'll likely need to modify the logic in `rlinf/data/datasets/openpi_pytorch/behavior/behavior_sft_dataset.py` involving `fine_grained_level`, `self.orchestrators = self.load_orchestrators`, `load_orchestrators_data`, `_get_fine_grained_task`, etc.

My requirements:

1. **`level = 0`**: only one text item — the pi05 input, which is the **main task**.
2. **`level = 1`**: two text items — one pi05 input (still the **main task**), plus an **output that is the subtask!**

How to refactor and optimize the rest of the design is **entirely up to you**. I want the dataset to still load via **streaming**, except that each load now additionally needs the **logic to read the subtask**.

---

## Acceptance Criteria

1. **Dataset part**: perfectly implement both `fine_grained_level`s. You need to verify that the text corresponding to the read frames satisfies the requirements above!
2. **SFT training part**: implement training on the behavior task at `fine_grained_level = 1`, and **see the loss go down**.
3. On top of (2), **save a trained checkpoint**, **eval it**, and inspect the model's intermediate outputs — does it **reasonably output the subtask** (including EOS)?
4. **Deeply verify** that in both SFT and eval, the handling of **attention, masks, and KV cache** is all correct. Provide some inputs, **print the outputs**, and confirm one by one that **every stage is correct!**
5. Keep your code changes minimal and clean, minimizing the impact on other parts of the codebase. All core changes should live under `rlinf/models/embodiment/openpi_pytorch` and `rlinf/data/datasets/openpi_pytorch`. Do not modify other workers (e.g., fsdp_vla_sft_worker) unless absolutely necessary.


**NOTE**: My design may have shortcomings, mistakes, or flaws. So if you have any questions, anything you feel is lacking in my design, or any point where you think you could do better, please raise it and ask me. Let's discuss every detail thoroughly before confirming the final plan. Use `/mnt/public/xzxuan/.venv_pi/bin/python` as the interpreter for this project. Use `/mnt/public/xzxuan/tmp` for temporary files and test outputs.
--- Original Design Draft End ---
