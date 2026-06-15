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