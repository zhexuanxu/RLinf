# VLM_VLA SFT failure — root cause, fix, and review

Companion to `docs/plan-phase8-optimize.md`. Records the confirmed root cause, the
applied fix, and the vla-regression review. The final `success_once` numbers are
appended when the corrected runs finish (see "Validation status").

## Symptom
Four-subtask `vlm_vla` SFT trained cleanly (loss down, `language_acc` high, correct
subtask emitted at eval, posture learned) yet evaluated at ~0 `success_once`
(broken run = 1/64 = 0.0156), while the byte-compatible pure-VLA baseline reaches
0.25. A constant-subtask control (perfect language) also failed (0.0156), isolating
the failure to the **action path**, not subtask-language difficulty.

## Root cause (confirmed): global gradient-clip coupling
`vlm_vla` SFT optimizes `total = language_loss_weight*CE + action_loss_weight*flow`
(both 1.0) under ONE optimizer and ONE global `clip_grad_norm_` at `clip_grad=1.0`
over all parameters. The subtask CE loss back-propagates only into the large shared
VLM backbone and **dominates the global gradient norm**, so the single clip
coefficient scales DOWN the action-expert update (which comes only from the flow
loss) every step. The baseline `vla` run has no CE term, hence no coupling.

Evidence — `toolkits/vlm_vla_diagnostics/grad_norm_audit.py` on the broken
checkpoint (one batch, bucketed by VLM vs action-expert, fixed noise/time):

| run | global norm | VLM | action-expert | clip coef @1.0 |
|---|---|---|---|---|
| CE-only | 21.18 | 21.18 | 0.00 | 0.047 |
| flow-only | 10.48 | 6.34 | 8.34 | 0.095 |
| combined | 24.23 | 22.75 | 8.34 | 0.041 |

The action-expert post-clip update is 0.344 (combined) vs 0.796 (flow-only) =
**0.43×** — chronic under-training of the action expert. `stop_gradient_to_vlm=True`
does NOT fix it (the action update worsens to ~0.37×, because CE still dominates the
global norm). Eliminated hypotheses: dataset frame-distribution (level-1 keeps 94% of
level-0 frames, balanced); train/eval position & KV parity (already fixed + pinned by
`tests/unit_tests/test_openpi_vlm_vla_model.py::TestSuffixPositionParity`; 21/21
vlm_vla unit tests pass).

## Fix: relieve the coupling by down-weighting CE (+ knowledge insulation)
`examples/sft/config/behavior_pi05_vlm_vla_fix.yaml`: `language_loss_weight=0.1`,
`stop_gradient_to_vlm=True` (hedge `…_fix_lw03.yaml` uses 0.3). Down-weighting CE
means it no longer dominates the global clip norm, so the action expert trains at
(near) full strength.

Verified before committing GPU-days:
- Grad-audit fix-preview (`--preview-lw 0.1`): action-expert attenuation **0.18 → 0.84**;
  combined norm **21 → 4.6**.
- 500-step real-data probe: `action_loss` 0.27→**0.025**, `language_acc`→**0.99**,
  `grad_norm`=**0.215** (< clip → no throttling; coupling relieved in practice).

If lw down-weight underperforms on the full run, the principled alternative is a
**per-expert gradient clip** (clip VLM and action-expert separately, each at 1.0;
keeps `language_loss_weight=1.0`), or the DEC-7 level-0 action-frame mixture.

## vla-regression review (AC-4 / DEC-3): PASS at code level
Changes to existing shared code are limited to:
- `examples/sft/run_vla_sft.sh`, `examples/embodiment/eval_embodiment.sh`: additive,
  backward-compatible Hydra-override passthrough (no extra args → unchanged behavior;
  the legacy `ROBOT_PLATFORM` positional is still honored).
- `rlinf/models/embodiment/openpi_pytorch/openpi_action_model.py`: the structured
  generation summary lives entirely inside the `if generation is not None` branch,
  which runs only in `vlm_vla` mode (vla has `generation=None`), and is
  exception-wrapped. No vla forward/loss/tokenizer/sampling code is touched.

The fix configs are new `vlm_vla`-only files. The pure-`vla` tokenizer string and SFT
loss are unchanged; `test_action_only_format_unchanged` (vla tokenizer guard) and the
attention/cache equivalence tests pass. Conclusion: no `vla`-mode numerics changed.

## Validation status — FIX FAILED; root cause REOPENED (2026-06-18)

The CE-down-weight fix did **not** work. `eval/success_once` for the lw=0.1 run = **0.0**
(64 traj) — worse than the broken baseline (0.0156). The grad-clip coupling was a
**real mechanism but not the cause of the eval failure**: relieving it did not recover
success, and down-weighting CE additionally **collapsed the eval-time subtask
prediction** (generation summary: `"move to radio."` ~75%, `"press radio."` 7/1020).

Corrected reading of the evidence:
- The constant-subtask **control** (trivial, correct-by-construction subtask) already
  scored 0.0156 → the **action expert produces bad actions independent of subtask
  prediction**. This points at the vlm_vla **action path**, NOT the gradient-clip
  coupling, and NOT (only) subtask prediction.
- The two diagnostics I wrongly substituted away are now the priority: (task4)
  **real-model** train-vs-eval action parity (the unit tests only cover tiny synthetic
  models — a real-model eval-denoise mismatch would explain "trains fine, evals ~0"),
  and (task5) **teacher-forced offline action-MSE** comparing baseline-VLA vs broken
  vlm_vla given the correct subtask.

Reopened hypotheses (action path): H3 prompt/template shift (the action expert now
conditions on `Subtask: <text>` + subtask tokens in the KV instead of the baseline
`Action:` prompt); H4-real (eval static-cache denoise diverges from the training
forward on the REAL model/inputs, undetected by the tiny-model unit tests). Decisive
next test: real-model teacher-forced parity (training forward vs eval denoise, same
noise) — parity FAIL ⇒ eval-path bug; parity PASS ⇒ the action expert is trained-bad
(prompt/training). Checkpoints available: baseline `…/RLinf_pi05/…-behavior_pi05_vla/pi05_sft_pytorch_new`,
broken `…/20260611-…-behavior_pi05_vlm_vla/…/pi05_sft_pytorch_new`, lw=0.1 fixed run.

Lesson: do NOT substitute a mechanism proxy (grad-norm) for the direct outcome gate
(teacher-forced action-MSE) before committing GPU-days. The action-MSE control would
have caught this cheaply.
