# RLinf 具身训练 PPO 全流程深度解析

以 `bash examples/embodiment/run_embodiment.sh behavior_ppo_openpi_pi05` 为例，本文档完整剖析从 rollout 数据收集到 PPO 训练的每一步。

---

## 第一部分：整体架构概览

RLinf 的具身训练采用三个分布式 Worker 协作：

```
┌─────────────┐     obs      ┌──────────────────┐    trajectory    ┌───────────────┐
│  EnvWorker  │ ──────────►  │  RolloutWorker   │ ──────────────► │  ActorWorker  │
│ (环境交互)   │ ◄────────── │ (策略推理)        │                  │ (PPO训练)      │
│             │   actions    │ huggingface_     │                  │ fsdp_actor_   │
│ env_worker  │              │ worker.py        │                  │ worker.py     │
└─────────────┘              └──────────────────┘                  └───────────────┘
```

**关键参数**（以 pi05 配置为例）：
- `num_action_chunks = 32`：每次策略推理生成 32 个连续动作（action chunk）
- `action_dim = 23`：每个动作 23 维
- `max_steps_per_rollout_epoch = 3000`：每个 epoch 最多 3000 步
- `n_train_chunk_steps = 3000 // 32 = 93`：每个 epoch 策略推理 93 次
- `rollout_epoch = 8`：每次训练迭代收集 8 个 epoch 的数据
- `B = total_num_envs // world_size`：每个 worker 的并行环境数

---

## 第二部分：Rollout 数据收集 —— EnvWorker 与 RolloutWorker 的配合

### 2.1 总体流程

`EnvWorker._run_interact_once()` 和 `RolloutWorker.generate_one_epoch()` 是成对运行的。它们通过 channel 异步通信，形成 **"乒乓球"式的交互**：

```
EnvWorker                                RolloutWorker
   │                                          │
   │──── send obs ────────────────────────►    │
   │                                          │── predict(obs) → actions, logprobs, values
   │    ◄──── send RolloutResult ─────────    │
   │                                          │
   │── env.chunk_step(actions) → rewards,dones│
   │── send next obs ────────────────────►    │
   │                                          │── predict(next_obs) → ...
   │    ◄──── send RolloutResult ─────────    │
   │                                          │
   │   ... 重复 n_train_chunk_steps 次 ...     │
   │                                          │
   │── send final obs ───────────────────►    │
   │                                          │── predict → 只取 value (bootstrap)
   │    ◄──── send final RolloutResult ───    │
```

### 2.2 `bootstrap_step()` —— 获取初始观测

**代码位置**: [env_worker.py:817-863](rlinf/workers/env/env_worker.py#L817-L863)

**作用**: 在每个 epoch 开始前，获取环境的初始 observation。

**为什么需要它**: 交互循环的第一步需要 obs 来让策略推理，但此时还没有执行过任何 action，所以需要一个 "bootstrap" 步骤来获取起始 obs。

**`auto_reset` 的影响**:

- **`auto_reset=False`（episodic 模式）**: 每个 epoch 都调用 `env.reset()`，从头开始新 episode。dones 全部初始化为 False。
  ```python
  extracted_obs, infos = self.env_list[stage_id].reset()
  dones = torch.zeros((B,), dtype=bool).unsqueeze(1).repeat(1, num_action_chunks)
  # dones shape: [B, num_action_chunks] 全 False
  ```

- **`auto_reset=True`（连续模式）**: **不 reset 环境**，而是使用上一个 epoch 结束时缓存的 obs（`self.last_obs_list`）。这意味着环境从上次停下的地方继续，episode 可以跨 epoch 延续。
  ```python
  env_output = EnvOutput(
      obs=self.last_obs_list[stage_id],  # 上个 epoch 的最后一个 obs
      dones=zero_dones,                   # 全 False，因为这是"起始"
  )
  ```

**关键理解**: `auto_reset=True` 时，环境内部在某个 env 的 episode 结束后会自动 reset 该 env 并继续，不需要外部显式调用 reset。这样 rollout 可以连续收集数据，不会因为某个 env 提前结束而中断整个 batch。

---

### 2.3 主循环：逐步详解

#### EnvWorker 侧（`_run_interact_once`，Line 927-1001）

```python
for chunk_step_idx in range(self.n_train_chunk_steps):  # 93 次
    for stage_id in range(self.stage_num):               # pipeline stage
```

**每一步做的事情**：

**Step A: 处理上一步的干预动作**（Line 934-938，可忽略）

**Step B: 接收策略推理结果**（Line 952-954）
```python
rollout_result = self.recv_rollout_results(input_channel, mode="train")
```
从 RolloutWorker 接收：
- `rollout_result.actions`: `[B, num_action_chunks, action_dim]` = `[B, 32, 23]` —— 策略输出的动作序列
- `rollout_result.prev_logprobs`: `[B, num_action_chunks, action_dim]` = `[B, 32, 23]` —— 当前策略下动作的对数概率（PPO 需要）
- `rollout_result.prev_values`: `[B, 1]` —— 当前状态的 Value 估计 V(s_t)（GAE 需要）
- `rollout_result.bootstrap_values`: `[B, 1]` —— **终止状态**的 Value 估计（仅当有 env done 时有值）
- `rollout_result.forward_inputs`: dict —— 模型的输入数据（训练时要重新前向传播用）
- `rollout_result.versions`: `[B, action_dim]` —— 模型权重版本号

**Step C: 计算奖励**（Line 955-957）
```python
rewards = self.compute_bootstrap_rewards(
    env_output, rollout_result.bootstrap_values, reward_model_output
)
```
详见下面 2.4 节。

**Step D: 打包结果并累积**（Line 958-977）
```python
chunk_step_result = ChunkStepResult(
    actions=rollout_result.forward_inputs.get("action", None),  # [B, num_action_chunks * action_dim]
    prev_logprobs=rollout_result.prev_logprobs,                  # [B, num_action_chunks, action_dim]
    prev_values=rollout_result.prev_values,                      # [B, 1]
    forward_inputs=rollout_result.forward_inputs,                # dict
    versions=rollout_result.versions,                            # [B, action_dim]
    dones=env_output.dones,                                      # [B, num_action_chunks]
    truncations=env_output.truncations,                          # [B, num_action_chunks]
    terminations=env_output.terminations,                        # [B, num_action_chunks]
    rewards=rewards,                                             # [B, num_action_chunks]
)
self.rollout_results[stage_id].append_step_result(chunk_step_result)
```

注意这里的 **时间对齐关系**：
- `env_output` 是 **上一步** env 执行动作后返回的结果（对于第 0 步就是 bootstrap 的初始 obs）
- `rollout_result` 是策略对 **当前 obs** 的推理结果
- 所以 `chunk_step_result` 记录的是：**"在 env_output 这个状态下，策略做了这个 action，得到了这些 reward 和 done"**

**Step E: 在环境中执行动作**（Line 979-981）
```python
env_output, env_info = self.env_interact_step(rollout_result.actions, stage_id)
```
`env_interact_step()` 内部调用 `env.chunk_step(actions)`，一次性执行 `num_action_chunks=32` 个原子动作步。返回：
- `env_output.obs`: 32 步之后的最终 observation
- `env_output.rewards`: `[B, num_action_chunks]` —— 32 步中每步的奖励
- `env_output.dones`: `[B, num_action_chunks]` —— 32 步中每步是否结束
- `env_output.terminations`: `[B, num_action_chunks]` —— 自然终止（任务成功/失败）
- `env_output.truncations`: `[B, num_action_chunks]` —— 时间截断（达到最大步数）
- `env_output.final_obs`: 如果有 env done 了，保存 done 时的 observation

**Step F: 发送新 obs 给 RolloutWorker**（Line 982-989）
```python
self.send_env_batch(rollout_channel, {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]})
```

#### RolloutWorker 侧（`generate_one_epoch`，Line 412-459）

与 EnvWorker 配对：

```python
for _ in range(self.n_train_chunk_steps):  # 93 次
    for _ in range(self.num_pipeline_stages):
        env_output = await self.recv_env_output(input_channel)  # 接收 obs
        actions, result = self.predict(env_output["obs"])        # 策略推理

        rollout_result = RolloutResult(
            actions=actions,                   # [B, 736]
            prev_logprobs=result["prev_logprobs"],  # [B, 23]
            prev_values=result["prev_values"],      # [B, 1]
            bootstrap_values=self.get_bootstrap_values(
                env_output.get("final_obs", None)    # 仅 done 时有值
            ),
            save_flags=save_flags,
            forward_inputs=result["forward_inputs"],
            versions=...,
        )
        self.send_rollout_result(output_channel, rollout_result, mode="train")
```

**`get_bootstrap_values(final_obs)`** 的作用：
- 当某个 env 在这一步 done 了（`auto_reset=True` 下），`final_obs` 会包含 done 时的 observation
- 对这个 `final_obs` 做一次策略前向传播，**只取 Value head 的输出**
- 这个值用于 truncation（时间截断）时的 TD bootstrap：被截断的 episode 并没有真正结束，未来还有潜在回报，需要用 V(s_T) 来估计

---

### 2.4 `compute_bootstrap_rewards()` —— 奖励计算与 truncation 处理

**代码位置**: [env_worker.py:640-680](rlinf/workers/env/env_worker.py#L640-L680)

```python
def compute_bootstrap_rewards(self, env_output, bootstrap_values, reward_model_output):
    rewards = env_output.rewards  # [B, num_action_chunks]

    # 1. 混合环境奖励和外部奖励模型的奖励
    if reward_model_output is not None:
        rewards = env_reward_weight * rewards + reward_weight * reward_model_output

    # 2. 处理 truncation 的 bootstrap
    if auto_reset and bootstrap_values is not None and env_output.dones is not None:
        if bootstrap_type == "standard":
            last_step_truncations = env_output.truncations[:, -1]  # 只看最后一步是否 truncated
        else:
            last_step_truncations = env_output.dones[:, -1]

        final_values = torch.zeros_like(rewards[:, -1])
        final_values[last_step_truncations] = bootstrap_values[last_step_truncations]

        # R_T' = R_T + gamma * V(s_{T+1})
        rewards[:, -1] += gamma * final_values

    return rewards
```

**为什么需要 bootstrap rewards？**

这是 PPO/Actor-Critic 中处理 **truncation vs termination** 的关键：
- **Termination**（自然结束）：任务成功或失败，V(s_terminal) = 0，不需要 bootstrap
- **Truncation**（时间截断）：达到最大步数被强制停止，但 agent 本来还能继续获取奖励。如果不补偿，算法会认为"截断后回报为 0"，导致低估这些状态的价值

所以对被 truncated 的 env，在最后一步的 reward 上加上 `gamma * V(s_{T+1})`，用 Value 网络估计的"未来回报"来补偿。

---

### 2.5 `compute_bootstrap_rewards` 与额外步骤的关系 —— 为什么两个都需要？

这两个机制解决的是 **两个不同场景** 的 bootstrap 问题，缺一不可。

**额外步骤**（2.6 节）解决的是：epoch 结束时 episode **还没 done**，GAE 最后一步需要 V(s_T)。

**`compute_bootstrap_rewards`** 解决的是：episode 在 epoch **中间** 因 truncation done 了（`auto_reset=True` 下）。

**为什么额外步骤的 V(s_T) 不够？** 关键在于 dones 会在 GAE 中把 V(s_{t+1}) 屏蔽掉。回忆 GAE 公式：

```python
delta = rewards[step] + gamma * values[step+1] * (~dones[step+1]) - values[step]
```

假设 chunk step 50（共 93 步）时 episode 被 truncated（`auto_reset=True`）：

- `dones[50]` = True
- auto_reset 后，`obs_51` 已经变成 **新 episode** 的第一帧
- `values[51]` = V(新 episode 的 s_0)，跟旧 episode 毫无关系

GAE 在 step 50 计算时，因为 `~dones[51]` = False，V(s_51) 被清零：
```
δ_50 = r_50 + γ * V(s_51) * 0 - V(s_50)
     = r_50 - V(s_50)
```
这相当于把 truncation 当成了 termination，**认为未来回报为 0**。但 truncation 只是时间截断，agent 本来还能继续拿奖励！

**`compute_bootstrap_rewards` 的巧妙之处**：它在 **reward 层面** 提前补偿：
```python
rewards[:, -1] += gamma * V(s_terminal)
```
这样 GAE 算到这一步时：
```
δ_50 = (r_50 + γ * V(s_terminal)) + γ * V(s_51) * 0 - V(s_50)
     = r_50 + γ * V(s_terminal) - V(s_50)   ← 正确！
```
**把 bootstrap value 藏进 reward 里，绕开了 dones 的屏蔽。**

两个机制处理的场景对比：

| 场景 | 谁来处理 | 怎么处理 |
|------|---------|---------|
| epoch 结束，episode **没 done** | 额外步骤的 V(s_T) | GAE 正常用 `values[T]`（`~dones[T]`=True，不屏蔽） |
| episode 在 epoch **中间 truncated** | `compute_bootstrap_rewards` | 把 `γ*V(s_terminal)` 加进 reward，绕开 dones 屏蔽 |
| episode 在 epoch **中间 terminated**（自然结束） | 不需要处理 | V=0 是正确的（任务真的结束了） |

---

### 2.6 主循环之后的额外步骤 —— 为什么需要多执行一次？

**代码位置**: [env_worker.py:1003-1037](rlinf/workers/env/env_worker.py#L1003-L1037)（EnvWorker）和 [huggingface_worker.py:448-459](rlinf/workers/rollout/hf/huggingface_worker.py#L448-L459)（RolloutWorker）

```python
# EnvWorker 侧 (Line 1003-1037)
for stage_id in range(self.stage_num):
    env_output = env_outputs[stage_id]
    rollout_result = self.recv_rollout_results(input_channel, mode="train")
    rewards = self.compute_bootstrap_rewards(env_output, rollout_result.bootstrap_values, ...)

    chunk_step_result = ChunkStepResult(
        prev_values=rollout_result.prev_values,  # ← 只有 value，没有 actions！
        dones=env_output.dones,
        truncations=env_output.truncations,
        terminations=env_output.terminations,
        rewards=rewards,
    )
    self.rollout_results[stage_id].append_step_result(chunk_step_result)
```

```python
# RolloutWorker 侧 (Line 448-459)
for _ in range(self.num_pipeline_stages):
    env_output = await self.recv_env_output(input_channel)
    actions, result = self.predict(env_output["obs"])

    rollout_result = RolloutResult(
        actions=actions,
        prev_values=result["prev_values"],  # ← 只保留 value
        bootstrap_values=self.get_bootstrap_values(env_output.get("final_obs", None)),
    )
    # 注意：没有 prev_logprobs, forward_inputs, save_flags, versions
```

**为什么需要这个额外步骤？**

这是 GAE (Generalized Advantage Estimation) 算法的要求。GAE 的公式：

```
δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
A_t = δ_t + (γλ) * (1 - done_{t+1}) * A_{t+1}
```

注意 `δ_t` 需要 `V(s_{t+1})`！所以如果我们收集了 T 步的数据（t=0,1,...,T-1），我们需要 T+1 个 value 估计：`V(s_0), V(s_1), ..., V(s_T)`。

- 主循环的 93 步收集了 `V(s_0)` 到 `V(s_{92})`（共 93 个 value）
- **额外的这一步收集 `V(s_{93})`**，即最后一个状态的 value
- 这个额外步骤只取 value，**不需要 action、logprob 等**，因为它纯粹是为了计算最后一步的 TD target

最终 trajectory 的 shape（`append_step_result` 累积后，`to_trajectory` stack 后）：
- `rewards`: `[n_chunk_steps+1, B, num_action_chunks]` = `[94, B, 32]` （**T+1 步**，额外步也记了 reward）
- `prev_values`: `[n_chunk_steps+1, B, 1]` = `[94, B, 1]` （T+1 步）
- `dones`: `[n_chunk_steps+1, B, num_action_chunks]` = `[94, B, 32]` （T+1 步）
- `prev_logprobs`: `[n_chunk_steps, B, num_action_chunks, action_dim]` = `[93, B, 32, 23]` （T 步，额外步没有 logprob）
- `actions`: `[n_chunk_steps, B, action_dim * num_action_chunks]` = `[93, B, 736]` （T 步，额外步没有 action）
- `forward_inputs`: `[n_chunk_steps, B, ...]` = `[93, B, ...]` （T 步）

**T vs T+1 的区别就是这个额外步骤带来的。** rewards、dones、prev_values、terminations、truncations 都是 T+1 步；actions、prev_logprobs、forward_inputs、versions 是 T 步。

在训练前的 `process_nested_dict_for_train()`（[fsdp_actor_worker.py:112-125](rlinf/workers/actor/fsdp_actor_worker.py#L112-L125)）中，`dones`、`terminations`、`truncations`、`prev_values` 会被 `value[:-1]` 截回 T 步（因为 advantage 已算完，训练时不需要第 T+1 步）。rewards 不在截断列表中，但在 shuffle 展平时与 actions 等 T 步数据一起 reshape，多出的一步 reward 会被 `shuffle_id` 的索引范围自然排除（因为 `rollout_size` 基于 `prev_logprobs` 的 T 步大小计算）。

---

### 2.6 `auto_reset=True` vs `auto_reset=False` 对代码的完整影响

| 方面 | `auto_reset=True` | `auto_reset=False` |
|------|-------------------|---------------------|
| `bootstrap_step()` | 使用上一个 epoch 缓存的 obs | 调用 `env.reset()` |
| episode 结束时 | env 自动 reset 该 env，继续 rollout | 整个 batch 停止，所有 env 一起 reset |
| `compute_bootstrap_rewards()` | 会添加 truncation 的 bootstrap value | 直接返回原始 reward |
| `compute_loss_mask()` | **不调用**（所有步都有效） | **调用**（done 之后的步需要 mask 掉） |
| 跨 epoch | episode 可以跨 epoch 延续 | 每个 epoch 是独立的 episode |

---

## 第三部分：Actor 训练 —— PPO 如何落地

### 3.1 数据接收与预处理 (`_process_received_rollout_batch`)

**代码位置**: [fsdp_actor_worker.py:1111-1186](rlinf/workers/actor/fsdp_actor_worker.py#L1111-L1186)

#### Step 1: 维度重组

```python
# 输入 shape: [rollout_epoch * n_chunk_steps, bsz, num_action_chunks, ...]
#           = [8 * 93, B, 32, ...] (rewards)
#           = [8 * 94, B, 32, ...] (dones, 因为 T+1)

rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

# 输出 shape: [n_chunk_steps, rollout_epoch * bsz, num_action_chunks, ...]
#           = [93, 8*B, 32, ...] (rewards)
#           = [94, 8*B, 32, ...] (dones)
```

**作用**: 把 8 个 epoch 的数据从"时间维度拼接"变成"batch 维度拼接"。这样每个 epoch 变成 batch 中的一个独立样本，可以并行训练。

#### Step 2: `compute_loss_mask()` —— 仅在 `auto_reset=False` 时调用

**代码位置**: [metric_utils.py:165-186](rlinf/utils/metric_utils.py#L165-L186)

```python
def compute_loss_mask(dones):
    _, actual_bsz, num_action_chunks = dones.shape
    n_chunk_step = dones.shape[0] - 1  # T+1 → T

    # Step 1: 展平 dones
    flattened_dones = dones.transpose(1, 2).reshape(-1, actual_bsz)
    # shape: [(T+1) * num_action_chunks, bsz] = [(94) * 32, 8*B]

    # Step 2: 从末尾切片，只取 T*num_action_chunks + 1 个时间步
    flattened_dones = flattened_dones[-(n_chunk_step * num_action_chunks + 1):]
    # shape: [93*32 + 1, 8*B] = [2977, 8*B]

    # Step 3: cumsum 找出 done 之后的所有步
    flattened_loss_mask = (flattened_dones.cumsum(dim=0) == 0)[:-1]
    # shape: [2976, 8*B] = [93*32, 8*B]

    # Step 4: 恢复原始 shape
    loss_mask = flattened_loss_mask.reshape(n_chunk_step, num_action_chunks, actual_bsz)
    loss_mask = loss_mask.transpose(1, 2)
    # shape: [93, 8*B, 32]
```

**为什么 `flattened_dones` 使用 `-(xxx):` 切片？**

`dones` 的 shape 是 `[T+1, bsz, num_action_chunks]`。展平后是 `[(T+1)*num_action_chunks, bsz]`。

但我们需要的逻辑是：将 `num_action_chunks` 个子步拉成连续的时间线。问题在于第一个 chunk step 的 dones（来自 bootstrap_step）是全 False 的"虚拟"值。我们需要从最后面切出真正有意义的 `T * num_action_chunks` 个步加上 1 个额外的起始标记（用于 cumsum 的锚点）。

`-(n_chunk_step * num_action_chunks + 1):` 精确地切出了这个范围，**跳过了开头多余的 padding**。

**为什么 `flattened_loss_mask` 使用 `[:-1]` 切片？**

cumsum 的结果有 `T * num_action_chunks + 1` 个元素，但 loss_mask 只需要 `T * num_action_chunks` 个（对应 T 步的动作）。最后一个元素是"第 T+1 步是否已经 done"，这个我们不需要（因为第 T+1 步没有对应的 action 要训练），所以切掉。

**cumsum 的巧妙之处**: `cumsum(dim=0) == 0` 的意思是"在我之前（包括我）有没有出现过 done"。一旦某个时间步 done=True，cumsum 变成 >=1，后面所有步的 mask 都变成 False。这完美实现了 **"done 之后的所有步都不参与 loss 计算"**。

#### Step 3: `filter_rewards` —— 按奖励过滤样本

**代码位置**: [fsdp_actor_worker.py:1138-1185](rlinf/workers/actor/fsdp_actor_worker.py#L1138-L1185)

```python
if self.cfg.algorithm.get("filter_rewards", False):
    # 1. 计算每个 prompt group 的总奖励
    rewards = rewards.reshape(n_prompts, group_size, n_step).sum(dim=-1)  # [n_prompts, group_size]
    mean_reward = rewards.mean(dim=1)  # [n_prompts]

    # 2. 筛选奖励在范围内的 group
    mask = (mean_reward >= lower_bound) & (mean_reward <= upper_bound)

    # 3. 更新 loss_mask
    rollout_batch["loss_mask"] = reward_filter_mask & rollout_batch["loss_mask"]
```

**filter_rewards 控制什么？决定什么？**

它允许你 **只用特定奖励范围内的样本来训练**。例如：
- `rewards_lower_bound=0.2, rewards_upper_bound=0.8`: 只用中等奖励的样本
- 在 GRPO 等算法中，可以过滤掉全失败（reward=0）或全成功（reward=1）的 group，因为这些 group 内没有对比信号，对 advantage 估计没有贡献

本质上，`filter_rewards` 通过修改 `loss_mask` 来控制哪些样本参与梯度计算。被过滤掉的样本虽然收集了，但不会贡献 loss。

---

### 3.2 计算 Advantages 和 Returns

**代码位置**: [fsdp_actor_worker.py:1188-1215](rlinf/workers/actor/fsdp_actor_worker.py#L1188-L1215)

```python
advantages_and_returns = calculate_adv_and_returns(
    task_type="embodied",
    adv_type="gae",  # or "grpo"
    rewards=rollout_batch["rewards"],     # [93, 8*B, 32]
    dones=rollout_batch["dones"],         # [94, 8*B, 32]  ← T+1 步！
    values=rollout_batch["prev_values"],  # [94, 8*B, 1]   ← T+1 步！
    gamma=0.99,
    gae_lambda=0.95,
    reward_type="chunk_level" or "action_level",
    loss_mask=...,
)
```

#### 预处理：`preprocess_embodied_advantages_inputs()`

[algorithms/utils.py:67-131](rlinf/algorithms/utils.py#L67-L131)

这个函数把 chunk 结构的数据展平成 GAE 需要的 `[n_steps, bsz]` 格式：

```python
# rewards: [93, 8*B, 32] → transpose → [93, 32, 8*B] → reshape → [2976, 8*B]
rewards = rewards.transpose(1, 2).reshape(n_steps, bsz)

# dones: [94, 8*B, 32] → transpose → [94, 32, 8*B] → reshape → [3008, 8*B]
# 然后从末尾切 2977 个: [-(2976+1):] = [2977, 8*B]
flattened_dones_full = dones.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
dones = flattened_dones_full[-(n_steps + 1):]

# values: [94, 8*B, 1] → 类似处理 → 但从开头取 n_steps+1 个
flattened_values_full = values.transpose(1, 2).reshape((num_chunk + 1) * chunk_size, bsz)
values = flattened_values_full[:n_steps + 1]
```

**注意 dones 和 values 的切片方向不同！**
- dones 从末尾切 `[-(n_steps+1):]`：因为第一个 chunk 的 bootstrap dones 是人造的全 False，真正有意义的在后面
- values 从开头切 `[:n_steps+1]`：value 是策略在每个状态上的估计，第一个 value 对应第一个状态，是有意义的

#### GAE 核心算法

[algorithms/advantages.py:24-86](rlinf/algorithms/advantages.py#L24-L86)

```python
# 输入：
#   rewards: [2976, 8*B]   (T 步)
#   values:  [2977, 8*B]   (T+1 步)
#   dones:   [2977, 8*B]   (T+1 步)

gae = 0
for step in reversed(range(T)):  # 从 T-1 到 0 反向遍历
    # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
    delta = rewards[step] + gamma * values[step+1] * (~dones[step+1]) - values[step]

    # GAE 递推: A_t = δ_t + γλ * (1 - done_{t+1}) * A_{t+1}
    gae = delta + gamma * gae_lambda * (~dones[step+1]) * gae

    # Returns: R_t = A_t + V(s_t)
    returns[step] = gae + values[step]

# Advantages: A_t = R_t - V(s_t)
advantages = returns - values[:-1]

# 标准化 advantages（用 loss_mask 只统计有效步的均值和标准差）
advantages = safe_normalize(advantages, loss_mask=loss_mask)
```

**`~dones[step+1]` 的作用**: 当 step+1 是 done 时，切断 GAE 的递推。这确保 advantage 不会跨 episode 传播。

---

### 3.3 `run_training()` —— PPO 训练循环

**代码位置**: [fsdp_actor_worker.py:1305-1489](rlinf/workers/actor/fsdp_actor_worker.py#L1305-L1489)

#### Step 1: 数据 shuffle

```python
rollout_size = prev_logprobs.shape[0] * prev_logprobs.shape[1]
# = n_chunk_steps * (rollout_epoch * bsz) * num_action_chunks
# 展平后 shuffle 所有样本的顺序

shuffle_id = torch.randperm(rollout_size)
rollout_batch = process_nested_dict_for_train(rollout_batch, shuffle_id)
```

PPO 论文要求：在多个 update epoch 中，每个 epoch 随机打乱样本顺序。

#### Step 2: 分 mini-batch 训练

```python
update_epoch = cfg.algorithm.update_epoch  # 通常 1-4
for _ in range(update_epoch):  # PPO 的多次更新 epoch
    for train_global_batch in split_dict_to_chunk(rollout_batch, num_chunks):
        for idx, batch in enumerate(train_micro_batch):  # 梯度累积
```

三层循环：
1. **update_epoch**: PPO 对同一批数据训练多次（通常 1-4 次）
2. **global_batch**: 把全部数据切成若干 global_batch
3. **micro_batch**: 每个 global_batch 再切成更小的 micro_batch 做梯度累积

#### Step 3: 前向传播

```python
output_dict = self.model(
    forward_inputs=forward_inputs,  # 包含 obs、action 等模型输入
    compute_logprobs=True,          # 计算 π_θ(a|s) 的 logprob
    compute_entropy=True,           # 计算策略熵（entropy bonus 用）
    compute_values=True,            # 计算 V(s)（critic loss 用）
)
```

注意：这里用的是 **当前最新权重** 重新前向传播，得到 **新的 logprob** 和 **新的 value**。而 `prev_logprobs` 和 `prev_values` 是 rollout 时（旧权重）记录的。

#### Step 4: PPO Loss 计算

```python
loss, metrics_data = policy_loss(
    loss_type="actor_critic",
    logprobs=output_dict["logprobs"],      # 新策略的 logprob
    values=output_dict["values"],          # 新策略的 value
    old_logprobs=prev_logprobs,            # 旧策略的 logprob（rollout 时记录）
    advantages=advantages,                  # GAE 计算的 advantage
    returns=returns,                        # GAE 计算的 return
    prev_values=prev_values,               # 旧策略的 value（rollout 时记录）
    clip_ratio_high=0.2,
    clip_ratio_low=0.2,
    loss_mask=loss_mask,
)
```

**Actor Loss (策略损失)**:
```
ratio = exp(logprob_new - logprob_old)  = π_new(a|s) / π_old(a|s)
clipped_ratio = clamp(ratio, 1-ε, 1+ε)
L_actor = -min(ratio * A, clipped_ratio * A)   ← PPO clip
```

**Critic Loss (价值损失)**:
```
V_clipped = V_old + clamp(V_new - V_old, -ε, +ε)
L_critic = max(MSE(V_new, R), MSE(V_clipped, R))   ← 也做 clip
```

**Entropy Bonus**:
```
L_total = L_actor + L_critic - β * H(π)
```
entropy bonus 鼓励探索，防止策略过早收敛。

#### Step 5: 反向传播与优化

```python
loss /= gradient_accumulation  # 梯度累积要除以累积次数
grad_scaler.scale(loss).backward()  # AMP 混合精度反向传播
# ... 累积完所有 micro_batch 后 ...
optimizer_step()  # 梯度裁剪 + 优化器更新
lr_scheduler.step()
```

---

## 第四部分：PPO 算法在 RLinf 中的完整数据流总结

### 4.1 为 PPO 收集了哪些数据？

| 数据 | Shape (单 epoch) | 来源 | 用途 |
|------|-------------------|------|------|
| `forward_inputs` (obs, action等) | `[T, B, ...]` | rollout 时策略的输入 | 训练时重新前向传播 |
| `prev_logprobs` | `[T, B, action_dim]` | rollout 时策略输出 | PPO ratio 的分母 π_old(a\|s) |
| `prev_values` | `[T+1, B, 1]` | rollout 时 value head 输出 | **GAE 计算** + Critic loss 的 clip 基准 |
| `rewards` | `[T, B, num_action_chunks]` | 环境返回 | **GAE 计算** |
| `dones` | `[T+1, B, num_action_chunks]` | 环境返回 | **GAE 计算**（截断递推）+ loss mask |
| `terminations` | `[T+1, B, num_action_chunks]` | 环境返回 | 区分自然终止和截断 |
| `truncations` | `[T+1, B, num_action_chunks]` | 环境返回 | bootstrap reward 计算 |
| `versions` | `[T, B, action_dim]` | 模型版本号 | 数据版本追踪 |

### 4.2 哪些数据用于训练，哪些用于计算 advantage？

**用于计算 GAE advantage 和 returns（不直接参与梯度计算）：**
- `rewards` → GAE 的 r_t
- `prev_values` (T+1 步) → GAE 的 V(s_t) 和 V(s_{t+1})
- `dones` (T+1 步) → GAE 的 episode 边界
- `bootstrap_values` → 在 `compute_bootstrap_rewards` 中已经被加进 rewards 了

**用于 PPO 策略梯度训练（直接参与梯度计算）：**
- `forward_inputs` → 重新前向传播，得到 新 logprob, 新 value, 新 entropy
- `prev_logprobs` → PPO ratio 的分母（旧策略）
- `advantages` → PPO 策略梯度的权重（由 GAE 计算得到）
- `returns` → Critic loss 的目标值（由 GAE 计算得到）
- `prev_values` → Critic loss 的 clip 基准
- `loss_mask` → 控制哪些步参与 loss

### 4.3 PPO 在 RLinf 中的完整落地流程

```
1. ROLLOUT 阶段（EnvWorker + RolloutWorker）
   ├── bootstrap_step() → 初始 obs
   ├── for t = 0 to T-1:
   │   ├── π_old(obs_t) → action_t, logprob_t, V(s_t)
   │   ├── env.step(action_t) → reward_t, done_t, obs_{t+1}
   │   ├── if done & truncated: 计算 bootstrap V(s_terminal)
   │   └── 记录 (forward_inputs_t, logprob_t, V(s_t), reward_t, done_t)
   └── 额外一步: π_old(obs_T) → V(s_T)  ← 第 T+1 个 value

2. 数据传输（EnvWorker → ActorWorker）
   ├── 8 个 epoch 的数据打包成 Trajectory
   └── 发送到 ActorWorker

3. 预处理（ActorWorker._process_received_rollout_batch）
   ├── 维度重组: [epoch*T, B, ...] → [T, epoch*B, ...]
   ├── compute_loss_mask(dones): done 后的步 mask 掉
   └── filter_rewards: 按奖励范围过滤样本

4. GAE 计算（compute_advantages_and_returns）
   ├── 展平 chunk 结构 → [n_steps, bsz]
   ├── 反向遍历: δ_t = r_t + γV(s_{t+1})(1-d_{t+1}) - V(s_t)
   ├── GAE: A_t = δ_t + γλ(1-d_{t+1})A_{t+1}
   ├── Returns: R_t = A_t + V(s_t)
   └── Normalize advantages

5. PPO 训练（run_training）
   ├── Shuffle 所有样本
   └── for update_epoch 次:
       └── for each mini-batch:
           ├── 前向传播: π_θ(a|s) → logprob_new, V_new, entropy
           ├── ratio = exp(logprob_new - logprob_old)
           ├── Actor Loss = -min(ratio*A, clip(ratio)*A)
           ├── Critic Loss = max(MSE(V_new, R), MSE(clip(V_new), R))
           ├── Total Loss = Actor + Critic - β*entropy
           └── Backward + Optimizer step
```

这就是 RLinf 中 PPO 具身训练的完整闭环。

---

## 第五部分：深入 Q&A

### Q1: `final_obs` 是什么，什么时候有，如何服务 PPO？`obs` 和 `final_obs` 分别什么时候用？

#### `final_obs` 的来源

`final_obs` 来自 Gymnasium 的 auto-reset 机制。当一个环境 done 后，Gymnasium 会自动 reset 该 env，此时：
- `obs` = **reset 后新 episode 的第一个观测**（因为 env 已经自动 reset 了）
- `infos["final_observation"]` = **done 那一刻的最后观测**（episode 结束前的真正最终状态）

所以在 `env_interact_step()` 中（[env_worker.py:396-401](rlinf/workers/env/env_worker.py#L396-L401)）：
```python
final_obs = (
    self._build_chunk_final_obs(obs_list, infos_list)  # 用于外部 reward model
    if self.use_external_reward_model
    else infos["final_observation"]                     # 标准情况
    if isinstance(infos, dict) and "final_observation" in infos
    else None
)
```

**`final_obs` 只在有 env done 时才存在**（不 done 就没有 `final_observation`）。

#### `obs` vs `final_obs` 的使用场景

EnvWorker 每次 `send_env_batch` 都同时发送 `obs` 和 `final_obs`：
```python
self.send_env_batch(rollout_channel, {"obs": env_batch["obs"], "final_obs": env_batch["final_obs"]})
```

在 RolloutWorker 侧（[huggingface_worker.py:417-447](rlinf/workers/rollout/hf/huggingface_worker.py#L417-L447)）：

- **`obs`** → 用于 `predict(env_output["obs"])`，策略基于当前观测生成 action、logprob、value
- **`final_obs`** → **只用于** `get_bootstrap_values(env_output.get("final_obs", None))`

也就是说：
- `obs` 是策略做决策的依据 —— 产出 action, logprob, value（PPO 训练的核心数据）
- `final_obs` 只用来算一个额外的 value estimate V(s_terminal)，这个值被 `compute_bootstrap_rewards` 用于 truncation 处理

#### `final_obs` 在 `_merge_obs_batches` 中的处理

当多个 env_worker 的数据需要合并时（[huggingface_worker.py:644-651](rlinf/workers/rollout/hf/huggingface_worker.py#L644-L651)）：
```python
if any(final_obs is not None for final_obs in final_obs_list):
    final_obs_or_obs = [
        final_obs if final_obs is not None else obs_dict
        for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
    ]
    merged_final_obs = _merge_obs_dicts(final_obs_or_obs)
```
注意：如果某个 worker 没有 done 的 env（`final_obs=None`），就**用 `obs` 来补位**。这是因为 `get_bootstrap_values` 需要完整的 batch 做前向传播（不能有 None），但对没有 done 的 env，`bootstrap_values` 的值不会被使用（因为 `compute_bootstrap_rewards` 只在 `last_step_truncations` 为 True 的 env 上使用 bootstrap_values）。

#### 只有 `auto_reset` 下才用吗？

**`final_obs` 不是只在 `auto_reset=True` 下才存在**。看 `env_interact_step()` 的代码，`final_obs` 的获取不依赖 `auto_reset` 配置——只要环境返回了 `infos["final_observation"]` 就会被捕获。

但是 **`final_obs` 只在 `auto_reset=True` 下才对 PPO 有实际作用**，因为 `compute_bootstrap_rewards()` 中：
```python
if bootstrap_values is None or not self.cfg.env.train.auto_reset or ...:
    return adjusted_rewards  # auto_reset=False 直接返回，不使用 bootstrap_values
```

在 `auto_reset=False` 模式下，即使有 `final_obs`，`bootstrap_values` 也不会被加到 reward 里。

---

### Q2: 为什么 `prev_values` 的 shape 是 `[B, 1]` 而不是 `[B, 32]`？

你确认了 `action` shape 是 `[B, num_action_chunks, action_dim]` = `[B, 32, 23]`，`logprob` 也是 `[B, 32, 23]`（修正了之前文档中的错误）。但 `prev_values` 是 `[B, 1]`。

**原因：Value function 评估的是"状态"而非"动作"。**

PPO 里 Value function V(s) 的含义是"从状态 s 开始，当前策略能获得的期望累积回报"。它只与状态有关，与具体采取什么动作无关。

一次 chunk step 中，策略看到 **一个** observation（状态 s_t），然后输出 32 个连续动作。这 32 个动作是一次前向推理的产物，它们对应的"决策时刻的状态"是同一个 s_t。所以：

- **一个 observation → 一个 V(s_t)** → shape `[B, 1]`
- **一个 observation → 32 个 action → 每个 action 一个 logprob** → shape `[B, 32, action_dim]`

Value Head 的实现（[modules/value_head.py:18-67](rlinf/models/embodiment/modules/value_head.py#L18-L67)）也证实了这一点：
```python
class ValueHead(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(512, 128), output_dim=1, ...):
        # output_dim=1，输出标量
```

**在 GAE 计算前，value 会如何处理？**

在 `preprocess_embodied_advantages_inputs()` 中：
```python
# values: [T+1, bsz, 1]  (chunk_size=1 for values)
flattened_values_full = values.transpose(1, 2).reshape((num_chunk + 1) * 1, bsz)
values = flattened_values_full[:n_steps + 1]
```

当 `reward_type="action_level"` 时，rewards 会被展开成 `[T*32, bsz]` 的细粒度时间线，而 values 只有 `[T+1, bsz]`。GAE 的计算实际上是在 chunk 级别而非 action 级别做的（或者说 values 在时间维度上的分辨率是 chunk 级别的）。

当 `reward_type="chunk_level"` 时，rewards 先在 action_chunks 维度 sum 再做 GAE，此时 rewards 和 values 的时间分辨率一致。

---

### Q3: `get_bootstrap_values` 为什么是 `final_values[:, :1]`？

```python
def get_bootstrap_values(self, final_obs):
    ...
    actions, result = self.predict(final_obs)
    final_values = result["prev_values"]   # [B, 1]
    return final_values[:, :1].cpu().contiguous()
```

`result["prev_values"]` 已经是 `[B, 1]`（Value head 输出标量），`[:, :1]` 只是一个保护性切片，确保即使 value head 意外返回多维结果也只取第一个值。本质上 `[:, :1]` 和直接用 `final_values` 是等价的。

**为什么 `get_bootstrap_values` 要对 `final_obs` 做完整的 `predict`？**

因为模型是端到端的——要得到 Value head 的输出，必须先让整个 backbone（视觉编码器 + transformer 等）处理输入 obs 生成中间表征，Value head 再从表征中回归出标量。不能跳过 backbone 直接算 value。所以虽然我们只要 value，但必须完整跑一次 predict（action 和 logprob 的输出被丢弃）。

---

### Q4: 为什么 `compute_bootstrap_rewards` 只在 `auto_reset=True` 时才考虑 bootstrap？

你的问题很好：`auto_reset=False` 时，如果 `max_steps_per_rollout_epoch` < `max_episode_steps`，一个 epoch 内 episode 也可能没执行完，此时不做 bootstrap 不会有问题吗？

**答案：`auto_reset=False` 模式下的设计语义不同。**

在当前的 `behavior_ppo_openpi_pi05.yaml` 配置中：
```yaml
max_episode_steps: 4096
max_steps_per_rollout_epoch: 4096
```
两者相等，所以 epoch 长度恰好是一个完整 episode。

但更根本的原因是两种模式的设计意图不同：

1. **`auto_reset=False`（episodic 模式）**：
   - 每个 epoch 开头做 `env.reset()`，所有 env 从头开始
   - 设计假设：`max_steps_per_rollout_epoch >= max_episode_steps`，即一个 epoch 足以跑完整个 episode
   - 如果 episode 在 epoch 中间 done 了，后续步骤通过 `compute_loss_mask` 被 mask 掉（不参与训练）
   - **不做 bootstrap 是正确的**：因为 episode 要么自然结束（termination，V=0），要么跑到了 max_episode_steps 被截断（truncation）。对于 truncation 的情况，GAE 计算的最后一步会用额外步骤收集的 `V(s_T)` 来处理（因为 `~dones[step+1]` 不会截断 GAE 递推，如果 episode 还没 done 的话）

2. **`auto_reset=True`（连续模式）**：
   - 环境在 done 后自动 reset 继续跑，episode 可以跨 epoch
   - Episode 可能在 chunk step 的 32 个子步中间结束，此时 `final_obs` 被捕获
   - **必须做 bootstrap**：因为被 truncated 的 env 立即接上了新 episode 的 obs，如果不在 reward 上补偿 `γ * V(s_terminal)`，GAE 会错误地将"新 episode 第一步的 value"当作"旧 episode 结尾的 value"

换言之，`auto_reset=False` 下不需要在 `compute_bootstrap_rewards` 中处理 truncation，是因为 GAE 的 `values[step+1]` 本身就已经提供了正确的 bootstrap（只要 episode 还没 done，`~dones[step+1]` 为 True，`V(s_{t+1})` 正常参与 TD 计算）。而 `auto_reset=True` 下，episode 结束后 obs 被替换成新 episode 的 obs，`values[step+1]` 不再对应旧 episode，所以必须在 reward 层面提前补偿。
