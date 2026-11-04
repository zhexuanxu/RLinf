# RPent planner PPO

原版 LIBERO workflow 使用文本 planner，并调用 VLA 和 SAM3 服务。环境、任务清单、提取后的语言模型和冻结 memory 的准备方法见 [LIBERO 配置指南](../../../docs/source-zh/rst_source/examples/agentic/rpent.rst)。

激活 RLinf 环境后，在仓库根目录运行：

```bash
source /path/to/rlinf-env/bin/activate
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo
```

启动脚本保持与其他 agent 示例一致的配置名称参数和 Hydra 覆盖方式；默认配置为同步 PPO。

## 异步 agent PPO

`AsyncAgentPPORunner` 持续产生 episode，写入有容量上限的完成队列。训练按完成顺序消费完整 episode。Checkpoint 保存已完成轨迹、未完成 prompt ID、dataloader 进度和策略版本。发布权重前暂停并撤回未完成的模型请求，每个输出 token 记录实际生成它的策略版本。

配置好模型、数据和服务路径后，可通过原版配方加以下参数选择 SAO PPO：

```bash
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo \
  runner.synchronous=false algorithm.loss_type=dis algorithm.adv_type=sao_gae \
  algorithm.recompute_logprobs=false algorithm.importance_sampling_fix=false \
  algorithm.n_minibatches=1 +agentloop.truncate_response=false
```

| 参数 | 含义 |
| --- | --- |
| `runner.synchronous` | `false` 选择异步 runner。 |
| `rollout.max_in_flight_rollouts` | 最多同时产生多少条 episode，默认 `data.rollout_batch_size` 的两倍。 |
| `rollout.rollout_buffer_capacity` | 完成队列容量，默认 rollout batch 的两倍。 |
| `rollout.max_weight_staleness` | 允许的策略发布版本差；`null` 不设上限，过期轨迹用同一个 prompt 重试。 |
| `algorithm.gae_lambda_alpha` | SAO 折扣修正，默认 `1.5`，不能小于 `1`。 |
| `critic.loss_type` / `critic.update_epochs` | `ppo` 或 `sao` value loss，每次 actor 更新对应的 critic 训练轮数。 |
| `runner.resume_dir` | 恢复包含异步队列状态的 checkpoint。 |

当前实现要求 Megatron actor/critic、SGLang rollout TP=PP=1、group size 1、单 minibatch、保留行为 logprobs，以及不截断 episode 输出。Actor 和 critic 共用设备，rollout 与它们使用相同设备或完全独立的设备。原版 LIBERO YAML、启动脚本和服务参数保持不变。
