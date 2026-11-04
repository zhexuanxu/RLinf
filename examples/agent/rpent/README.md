# RPent planner PPO

The original LIBERO workflow runs the text planner against the VLA and SAM3 services. Follow the [LIBERO setup guide](../../../docs/source-en/rst_source/examples/agentic/rpent.rst) to prepare its environment, task dataset, extracted language checkpoint and frozen memory.

Activate the RLinf environment and run from the repository root:

```bash
source /path/to/rlinf-env/bin/activate
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo
```

The launcher retains the same configuration-name argument and Hydra overrides as the other agent examples. The default recipe uses synchronous PPO.

## Asynchronous agent PPO

`AsyncAgentPPORunner` produces episodes continuously into a bounded completion queue. Updates consume completed episodes in completion order. Checkpoints preserve queued trajectories, pending prompt IDs, dataloader progress and policy versions. Publication pauses and retracts unfinished model requests; each sampled token retains the policy version that generated it.

To select SAO PPO using the original recipe, supply these overrides after configuring model, dataset and service paths:

```bash
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_ppo \
  runner.synchronous=false algorithm.loss_type=dis algorithm.adv_type=sao_gae \
  algorithm.recompute_logprobs=false algorithm.importance_sampling_fix=false \
  algorithm.n_minibatches=1 +agentloop.truncate_response=false
```

| Setting | Meaning |
| --- | --- |
| `runner.synchronous` | `false` selects the asynchronous runner. |
| `rollout.max_in_flight_rollouts` | Maximum episode producers; defaults to twice `data.rollout_batch_size`. |
| `rollout.rollout_buffer_capacity` | Completed-episode queue capacity; defaults to twice the rollout batch size. |
| `rollout.max_weight_staleness` | Maximum allowed published-policy lag; `null` disables this bound. Stale episodes retry the same prompt. |
| `algorithm.gae_lambda_alpha` | SAO discount correction; default `1.5`, must be at least `1`. |
| `critic.loss_type` / `critic.update_epochs` | `ppo` or `sao` value loss and critic update passes per actor update. |
| `runner.resume_dir` | Resume a checkpoint including its asynchronous queue state. |

The implementation requires Megatron actor/critic, SGLang with rollout TP=PP=1, group size 1, one minibatch, behavior logprobs, and untruncated episode responses. Actor and critic share devices; rollout uses either the same devices or a disjoint set. The original LIBERO YAMLs, launcher and service arguments remain unchanged.
