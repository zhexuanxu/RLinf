# VLM_VLA SFT failure — root cause, fix, and review

Companion to `docs/plan-phase8-optimize.md`. Records root-cause hypotheses,
diagnostic evidence, failed probes, and the vla-regression review. The final
`success_once` numbers are appended when a corrected run succeeds.

## Symptom
Four-subtask `vlm_vla` SFT trained cleanly (loss down, `language_acc` high, correct
subtask emitted at eval, posture learned) yet evaluated at ~0 `success_once`
(broken run = 1/64 = 0.0156), while the byte-compatible pure-VLA baseline reaches
0.25. A constant-subtask control (perfect language) also failed (0.0156), isolating
the failure to the **action path**, not subtask-language difficulty.

## Former lead hypothesis: global gradient-clip coupling
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

This was a real mechanism in the earlier audit, but the later lw=0.1
free-generation run failed and the fixed-real-batch audit below shows no active
clip coupling once the subtask CE is saturated. Treat this as an observed training
mechanism, not the root cause.

## Failed probe: down-weight CE (+ knowledge insulation)
`examples/sft/config/behavior_pi05_vlm_vla_fix.yaml`: `language_loss_weight=0.1`,
`stop_gradient_to_vlm=True` (hedge `…_fix_lw03.yaml` uses 0.3). This relieved the
clip-coupling mechanism in the proxy, but it did **not** recover eval success.

Verified before committing GPU-days:
- Grad-audit fix-preview (`--preview-lw 0.1`): action-expert attenuation **0.18 → 0.84**;
  combined norm **21 → 4.6**.
- 500-step real-data probe: `action_loss` 0.27→**0.025**, `language_acc`→**0.99**,
  `grad_norm`=**0.215** (< clip → no throttling; coupling relieved in practice).

The lw=0.1 run scored `eval/success_once=0.0` over 64 trajectories, worse than the
broken 0.0156 baseline. Do not treat this config as a fix candidate without new
evidence from the direct action-path diagnostics.

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

### Re-diagnosis results (2026-06-18)
- **Fixed-real-batch grad audit:** `/mnt/public/xzxuan/tmp/grad_norm_audit_real_batch_broken.txt`
  shows `language_loss=0.0000`, `language_acc=1.0000`, `action_loss=0.0128`, no global
  clipping, and action-expert update attenuation `1.0000` for both
  `stop_gradient_to_vlm=False` and `True`. On this batch, CE/clip coupling is not an
  active failure mode.
- **Teacher-forced action-MSE control:** `/mnt/public/xzxuan/tmp/action_mse_base_vs_broken.txt`
  gives `M_base=0.0115228426`, `M_broken=0.0131088737`, ratio `1.1376`, passing the
  `<=1.5x` proxy gate. The broken checkpoint is close to baseline under the offline
  one-step teacher-forced flow-MSE proxy.
- **Real-model train/eval velocity parity:** `/mnt/public/xzxuan/tmp/offline_velocity_parity.txt`
  runs the broken checkpoint with teacher-forced non-EOS subtask tokens in a
  `StaticKVCache`. The required bf16 check fails the strict `1e-3` tolerance
  (`max_abs_velocity_diff=0.0234375`), while the same harness in float32 passes
  (`/mnt/public/xzxuan/tmp/offline_velocity_parity_fp32.txt`,
  `max_abs_velocity_diff=1.19e-6`). This points to bf16 cached-eval numerical/order
  sensitivity rather than a gross mask/context mismatch.

### PROXIMATE CAUSE FOUND (2026-06-18): action expert fits ~3x worse in vlm_vla
- Full closed-loop evals on the CORRECT step-30000 checkpoints (correct env): broken@bf16=0.0156,
  broken@fp32=0.0, lw=0.1@30000=0.0, lw=0.1@fp32=0.0, lw=0.3@30000=0.0. So: CE weight ratio,
  and bf16-vs-fp32 eval precision, are BOTH ruled out as fixes.
- `toolkits/vlm_vla_diagnostics/full_denoise_mse.py` (FULL Euler denoise vs GT actions on the saved
  real batch, normalized): baseline-VLA MSE 0.00072, broken-VLM-VLA 0.0025 — **~3x worse**, at every
  num_steps (5/10/20/40 → ratio 3.5/3.2/3.0/2.9). The one-step flow-MSE proxy (1.14x) UNDER-stated
  this; the systematic fixed-point error of the velocity field is ~3x. This ~3x action-fit gap (≈2x
  per-step RMS) is the proximate cause of the closed-loop collapse (0.0156 vs baseline 0.25).
- It is NOT gradient magnitude (lw=0.1 gave full action gradient + relieved clip, still 0.0/3x), NOT
  frame count (level-1 keeps 94%), NOT num_steps, NOT bf16. The remaining difference from the working
  baseline is the **joint subtask-CE training**: training the VLM to generate subtasks reshapes the
  prefix representations the action expert reads (and/or the action expert attending the subtask
  tokens), degrading the action fit even with stop_gradient_to_vlm=True (CE still trains the VLM).
- OPEN sub-hypotheses for the fix (each needs a ~30h train): (a) VLM subtask-shaping degrades the
  action-relevant prefix features (KI not fully achieved); (b) the action expert attending the subtask
  response tokens degrades its fit (test: block suffix from response, action expert conditions on
  images+task+state only like baseline, while the VLM still generates the subtask for CE). Candidate
  recipes: action-expert-only warmup then enable CE; a stronger knowledge-insulation scheme; or (b).
  DEC-7 (level-0 action frames) is DEPROVED as a fix (94% already used).

**Current status:** proximate cause = ~3x action-fit gap from joint subtask-CE training; exact
mechanism (a vs b) and fix need a training experiment. Earlier note: direct diagnostics ruled out
active CE/clip coupling on the fixed batch and bf16 eval precision. Next step is task8/task9
synthesis, plus the still-missing teacher-forced env eval, before any further full
training run.
