# Optimize VLM_VLA SFT Training Performance on the Behavior Task

In this round, I need you to work carefully, dig deep into the code, perform an in-depth analysis, and optimize the performance of my SFT training.

## Current Situation

### Baseline (pure VLA SFT — works)

A pure VLA SFT run, trained **only** on the Behavior `task 0000` training data, corresponds to the config `examples/sft/config/behavior_pi05_vla.yaml`. The model produced by this run reaches an **eval accuracy of 0.25** in the actual eval.

- Full training log: `/mnt/public/xzxuan/repos/RLinf_pi05/logs/20260605-12:39:44-behavior_pi05_vla`
- Full eval log: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-16:08:02-behavior_ppo_openpi_pi05_pytorch_eval`

> Note: this training was run in a **different repository**. The code in that repo and the code in the current repo are **aligned under this VLA training config**.

### The key difference: VLM_VLA mode

Compared with VLM_VLA training, the difference is that the latter:
1. Uses **subtasks**.
2. Splits the SFT loss into **two parts**.
3. At eval time, lets the model **first autoregressively generate the subtask, then generate the action**.

### The problem (VLM_VLA — broken)

The model obtained from SFT training in **VLM_VLA mode** evals very poorly. The config is `examples/sft/config/behavior_pi05_vlm_vla.yaml`.

- My SFT training log: `/mnt/public/xzxuan/repos/RLinf/logs/20260611-13:45:13-behavior_pi05_vlm_vla`
- My eval log: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-15:54:57-behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`

Observed behavior:
- During training, the **SFT loss is decreasing**, and **language accuracy keeps rising**.
- In the eval log, the model **does output a subtask**, and the **output content is at least correct**. Watching the video, the **overall posture is learned** as well.
- However, **`success_once` is very poor (0%)** — i.e., the training result is bad.

### Control experiment

I also trained a **control experiment**: `/mnt/public/xzxuan/repos/RLinf/logs/20260613-05:42:48-behavior_pi05_vlm_vla_test`. The config is `examples/sft/config/behavior_pi05_vlm_vla_test.yaml`

- The **only** modification is using `examples/sft/config/behavior_pi05_vlm_vla_test.yaml` (the run above used `examples/sft/config/behavior_pi05_vla.yaml`). The only difference is that in the `test` yaml, **`task_subtasks` is entirely changed to output `"turning on radio"`** instead of the normal four subtasks.
- Eval log for this control: `/mnt/public/xzxuan/repos/RLinf/logs/20260616-16:15:06-behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`

Results of the control:
- At eval time, the model outputs **only** `'turning on radio.' [EOS]`.
- The **training accuracy stays at 1** throughout.
- The eval **`success_once` is 0.01**.

## Summary

The SFT training performance under **VLM_VLA mode is very poor**. I need you to deeply analyze **which stage is the root cause**. Here are a few directions:

1. **Code correctness (training and eval).** Is the attention handling correct? Is the KV cache handling correct? Etc.
2. **Hyperparameters.** Could a hyperparameter be causing such a stark difference — e.g., the **ratio between the CE loss and the action loss**?
3. **`stop_gradient_to_vlm`.** Should this be turned on? Verify that the logic for setting this config to `true` vs `false` is correct — i.e., if `true`, the VLM portion's weights should **not change** after training. (You can train a few steps, dump a checkpoint, and verify against the base model.)

## Required reference papers

You must carefully study the ideas in these papers (they are official OpenPI papers related to Pi05):
- **OpenPI05**: https://arxiv.org/abs/2504.16054
- **Knowledge Insulating**: https://arxiv.org/abs/2505.23705

## Notes / Constraints

1. My three eval runs used **exactly the same code** (i.e., the current eval in the current repo), to ensure the differences truly come from the **training**.
2. Environment: `/mnt/public/xzxuan/.venv_pi`
3. `/mnt/public/xzxuan/tmp` is for intermediate results and test outputs, etc.
4. For both training and eval, the **full config** can be found in `tensorboard/config.yaml` inside the log directory. Use this to inspect the **actual** training and eval configs.
5. Commands:
   - Training: `bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_vla`
   - Eval: `bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval`
6. **Hardware.** This machine has **8 GPUs**. You can also access another 8-GPU machine via:
   ```
   ssh -p 40431 root@183.233.148.6
   ```
   The two machines have **identical configurations** and **share storage**, so you effectively have **16 GPUs** and can run two sets of experiments in parallel. Use the same code, environment, and data on both machines — I've already verified it runs. First validate that you can: log in, load the environment, see the GPUs, and execute scripts — make sure you can fully control all 16 GPUs.
7. Refer to `docs/plan-phase8-vlm-vla-mode.md` and `.humanize/rlcr/2026-06-11_10-03-58` for last rlcr loop, where we achieve the full vlm_vla mode code(reference to /mnt/public/xzxuan/repos/vla_lib)

## Acceptance Criterion

There is **only one** acceptance criterion: by **whatever method**, the final **VLM_VLA mode**, based on the **four different subtask outputs**, must reach a **`success_once` of 25 or above** in eval.

A single training run may take ~30h. Add a hook so that when training finishes, convert the checkpoint into an eval-ready ckpt with:
```
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new
```
then run eval. The final result must reach **25 or above**.