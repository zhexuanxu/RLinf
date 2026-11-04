<!-- sphinx-body -->

# RPent reinforcement learning

This example trains a multimodal Qwen3.5 policy with LIBERO feedback, using RPent to execute each episode and RLinf to optimize the policy.

## Choose the integration

The original `libero` workflow remains the launcher default. Its `train_qwen3.5_4b_ppo` and `eval_qwen3.5_4b` configs retain the original Pi0.5, memory and language-only model setup. See the {doc}`original LIBERO example </rst_source/examples/agentic/rpent>` for that workflow.

The separate `libero_rl` workflow uses a **complete multimodal** Qwen checkpoint, adaptive observation and six tools: `move_to`, `set_gripper`, `view_env`, `segment`, `back_project` and `finish`. It needs SAM3, but no Pi0.5 server or memory service. Select it explicitly with one of these configs:

| Config name, without `.yaml` | Purpose |
|---|---|
| `train_qwen3.5_4b_libero_rl_ppo` | Synchronous PPO template |
| `train_qwen3.5_4b_libero_rl_sao` | Asynchronous SAO template |
| `train_qwen3.5_4b_libero_rl_sao_local` | Launch profile containing this deployment's paths |
| `train_qwen3.5_4b_libero_rl_sao_3node` | Shared 16-GPU SAO, plus a separate environment node |
| `eval_qwen3.5_4b_libero_rl` | Multimodal evaluation template |

## Install code and environments

Use Linux NVIDIA hosts with a compatible driver, CUDA 12.8 toolkit, EGL/OpenGL libraries, Git and `uv`. Choose a shared root, or reproduce the same paths on every node. The feature branches below contain both sides of this integration; unrelated official-main revisions may not.

```bash
export STACK_ROOT="$HOME/rpent-stack"
mkdir -p "$STACK_ROOT/repos" "$STACK_ROOT/envs" "$STACK_ROOT/models" "$STACK_ROOT/data"
git clone --branch rpent-ppo https://github.com/zhexuanxu/RLinf.git "$STACK_ROOT/repos/RLinf"
git clone --branch rpent-ppo https://github.com/zhexuanxu/RPent.git "$STACK_ROOT/repos/RPent"
git clone https://github.com/NVIDIA/Megatron-LM.git "$STACK_ROOT/repos/Megatron-LM"
git -C "$STACK_ROOT/repos/Megatron-LM" checkout dc0ee41ada177d4c74ba46bb501bba5967264815
export MEGATRON_PATH="$STACK_ROOT/repos/Megatron-LM"
cd "$STACK_ROOT/repos/RLinf"
UV_TORCH_BACKEND=cu128 UNINSTALL_FA4=1 bash requirements/install.sh agentic \
  --venv "$STACK_ROOT/envs/rlinf" --python 3.12.12 --torch 2.11.0 \
  --engine sglang --sglang 0.5.12.post1 --transformers 5.6.1 \
  --use-mirror --no-root --install-rlinf
```

The installer honors an existing `MEGATRON_PATH`. It temporarily edits dependency metadata while resolving, so use an idle checkout and a new environment. `--no-root` requires native system dependencies to be installed already. `UNINSTALL_FA4=1` selects the validated FA2 route. See `bash requirements/install.sh --help` for other platforms.

Install the separate simulator/SAM3 environment from the matching RPent branch:

```bash
cd "$STACK_ROOT/repos/RPent"
uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
source "$STACK_ROOT/envs/rpent/bin/activate"
uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
```

The tested stack uses these versions. This table records compatibility, rather than promising that every future dependency resolution reproduces the same environment.

| Component | Tested versions |
|---|---|
| RLinf | Python 3.12.12; Torch 2.11.0+cu128; Transformers 5.6.1; SGLang 0.5.12.post1 |
| Training kernels | Megatron Core 0.18.3 at the commit above; Bridge 0.5.0; Transformer Engine 2.17.0; FlashAttention 2.8.3 built for Torch 2.11 |
| Qwen linear attention | flash-linear-attention/fla-core 0.6.0; TileLang 0.1.8; Triton 3.6.0 |
| Simulator environment | Torch 2.11.0+cu128; Transformers 4.57.6; NumPy 1.26.4; MuJoCo 3.3.0; robosuite 1.5.2; rpent-libero/rpent-liberopro 0.2.0; SAM3 0.1.0 |

Check the active environment with `python -m pip freeze`, `python -c 'import torch; print(torch.__version__, torch.version.cuda)'` and `git -C "$MEGATRON_PATH" rev-parse HEAD`. Preserve a package inventory after validating it. A matching CUDA compiler is needed for source builds; a networked CPU-only machine can download wheels/sources, but does not necessarily build the CUDA extensions. For offline nodes, prepare the complete dependency set on a compatible Linux/Python/CUDA build host, transfer it and the Git sources, then install with `uv pip install --no-index --find-links /path/to/wheelhouse ...`. A partial wheel cache is not a complete offline installation.

All Ray nodes need the same RLinf environment, source revision and model files. Keep RPent in its own environment; do not upgrade the RLinf Transformers installation to match RPent's. When supplying custom activation profiles, preserve the installer's Megatron, CUDA and NCCL paths. This recipe uses FA2 and disables Torch compilation; set `NVTE_FLASH_ATTN=1`, `NVTE_FUSED_ATTN=0` and `TORCH_COMPILE_DISABLE=1` before starting Ray. EGL must work on the environment node.

## Prepare models and standard LIBERO assets

Place the full Qwen3.5-4B model, or a compatible full SFT export, under `$STACK_ROOT/models/`. It must include the vision weights, tokenizer, processor and chat template. A language-only extraction cannot serve adaptive image turns. Download the base model on a networked machine, for example:

```bash
hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"
```

Obtain the SAM3 checkpoint following the [SAM3 repository](https://github.com/facebookresearch/sam3), and set `SAM3_CHECKPOINT_PATH` to its local `sam3.pt`. Assets are not bundled in the Python packages. In the RPent environment, download the standard assets and write a dedicated benchmark config:

```bash
source "$STACK_ROOT/envs/rpent/bin/activate"
export LIBERO_ASSETS="$STACK_ROOT/data/libero/assets"
export LIBERO_CONFIG_PATH="$STACK_ROOT/data/libero-standard-config"
env -u LIBERO_ASSET_PATH libero-download-assets \
  --assets-dir "$LIBERO_ASSETS" --skip-existing --no-cache
mkdir -p "$LIBERO_CONFIG_PATH"
python - <<'PY'
import os
from pathlib import Path
import yaml
import libero.libero as libero
assets = Path(os.environ['LIBERO_ASSETS']).resolve()
package_assets = Path(libero.__file__).resolve().parent / 'assets'
# LIBERO also constructs scene/mesh paths relative to its installed package.
if not package_assets.exists():
    if package_assets.is_symlink():
        raise RuntimeError(f'Broken LIBERO assets link: {package_assets}')
    package_assets.symlink_to(assets, target_is_directory=True)
for scene in ['libero_tabletop_base_style.xml', 'libero_floor_base_style.xml',
              'libero_kitchen_tabletop_base_style.xml',
              'libero_living_room_tabletop_base_style.xml']:
    assert (package_assets / 'scenes' / scene).is_file(), scene
config = libero.get_default_path_dict()
config['assets'] = str(assets)
Path(os.environ['LIBERO_CONFIG_PATH'], 'config.yaml').write_text(yaml.safe_dump(config))
PY
```

The installed package's assets link is necessary because LIBERO constructs some XML/mesh paths relative to its source, independently of config.yaml. Preserve an existing complete assets directory; repair a broken link instead of replacing scene geometry. Keep the package's benchmark/BDDL/initial-state paths from `get_default_path_dict()`. The `sao_3node` recipe passes this config explicitly to episode subprocesses, so a parent process configured for LIBERO-PRO cannot silently change the standard benchmark.

Create the standard 40-task training list with the RPent interpreter:

```bash
cd "$STACK_ROOT/repos/RLinf"
python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
  --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
  --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"
```

This writes 1,280 rows and a manifest containing task/state counts, seeds and hashes. Seed 0 is reserved for evaluation. The generator rejects seeds outside each task's initial-state range instead of silently reusing an initial state through modulo indexing. For separate dev/test manifests, use `--seeds 33-39 --split dev` or `--seeds 40-49 --split test` with a separate output directory. The old memory-based generator remains the default `--workflow libero` and still requires `--memory-dir`.

## Configure Shared 16-GPU SAO

Use the four clean-workflow profiles as an inheritance chain: `libero_rl_ppo` → `libero_rl_sao` → `libero_rl_sao_3node` → `libero_rl_sao_local`. Every name has the prefix `train_qwen3.5_4b_`. The first inherits the original `train_qwen3.5_4b_ppo`, which remains available for the original LIBERO integration. Common task, sampling and model settings live in the clean PPO profile; SAO overrides its update rules, the three-node profile overrides placement, and the local profile contains deployment paths.

Edit `config/train_qwen3.5_4b_libero_rl_sao_local.yaml` for your installation. Set the full multimodal model directory, training JSONL, RPent interpreter, LIBERO configuration, episode directory and training output directory. The parent profiles require these paths instead of supplying machine-specific defaults. Setting `rollout.model.model_path` also sets the actor/critic tokenizer and HF import paths.

| Resource or parameter | Configuration |
|---|---|
| Nodes 0–1, GPUs 0–15 | Rollout, critic and actor share all 16 GPUs in successive phases; weights, gradients and optimizer states offload between phases |
| Learners | TP2, CP1, DP8; sequence parallel; BF16; activation recomputation; response-only logits |
| Rollout | 16 TP1 SGLang replicas; `replica_affinity=false`; `max_running_requests=128` per replica; memory fraction 0.7 |
| Node 2, GPUs 0–7 | Eight agent workers, 32 episode slots each; one SAM3 service on GPU 0 |
| Completion scheduling | Batch 128; `max_in_flight_rollouts=null` and `rollout_buffer_capacity=null` each resolve to 256 |
| Training budget | 10 dataset epochs, no separate step cap; 1,280 input rows yield 100 learner updates; save every five updates |
| Optimization | One actor minibatch and two critic update epochs per learner update; microbatch 1; actor LR `1e-6`, critic LR `5e-6`; language parameters train, vision encoder/projector remain frozen |
| Episode | Simple prompt, adaptive observation, thinking, 100 turns, 10,000 env steps, four-hour budget |
| Generation | Context 262144, output 32768; temperature 1, top-p 0.95, top-k 20, presence penalty 1.5, repetition penalty 1.0 |
| Context management | Native message window 80→40, trigger fraction 0.7; no extra tool clearing or summarization |

SAO keeps up to 256 episodes in flight and consumes the next 128 completed trajectories, rather than waiting for a fixed synchronous batch to finish. An episode's slot includes model waits and tool execution; eight workers with 32 slots can start all 256 in-flight episodes. `max_env_initializations_per_worker=8` bounds cold starts to 64 processes across the eight workers. A permit is released after the first environment state is saved, not at episode completion, so all 256 environments can remain active after warmup. This limits CPU/import/CUDA-context pressure without reducing rollout capacity. The buffer holds complete trajectories. Additional pending prompts can be queued, but do not increase the in-flight limit. Each episode normally has one outstanding model request, so a per-replica limit of 128 is an admission ceiling, not 128 continuously active requests on every GPU.

Shared GPUs still require a phase switch: pause/retract model requests, offload rollout memory, train the critic and actor, synchronize weights, then resume the waiting requests. Environment work may continue during training, but model generation pauses. SAO also changes the learning rules: DIS actor loss, clip bounds 0.3/5.0, `sao_gae` with alpha 1.5, and the SAO critic objective. It uses behavior logprobs saved during rollout instead of recomputing old-policy logprobs. The task, model, prompt, sampling and compression settings remain inherited from clean PPO.

Tool results and images remain in subsequent prompts. Only policy-generated assistant tokens receive policy loss; user/tool/history tokens do not. Prefix-compatible turns merge, while context compression/history rewriting starts a new row. The critic and actor use the same response-position mapping. Response-only logits require CP1.

## Start three Ray nodes and SAM3

Before `ray start` on **each node**, source the RLinf environment and export `MEGATRON_PATH`, `NVTE_FLASH_ATTN=1`, `NVTE_FUSED_ATTN=0`, `TORCH_COMPILE_DISABLE=1`, and a suitable `RLINF_TIMEOUT` such as `10800`. Choose the actual network interface; `RLINF_COMM_NET_DEVICES=eth0` is an example, not a universal interface name. Ray workers inherit this environment at startup.

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
export HEAD_IP=10.0.0.10  # reachable head-node address
export RLINF_NODE_RANK=0
ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8
```

On model node 1 and environment node 2, respectively, use the same source/environment and set the corresponding rank and local address:

```bash
export RLINF_NODE_RANK=1  # use 2 on the environment node
export NODE_IP=10.0.0.11 # use the environment node's address there
ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8
```

Run `ray status --address="$HEAD_IP:6387"` and verify three nodes and 24 GPUs. Do not start a second conflicting Ray cluster on an occupied node. Verify multi-node NCCL communication before a long job; network/firewall/RDMA settings depend on the cluster.

In a separate terminal on node 2, start SAM3 using the RPent environment:

```bash
source "$STACK_ROOT/envs/rpent/bin/activate"
cd "$STACK_ROOT/repos/RPent"
export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101
```

The agent workers and episode processes all run on node 2, so `http://127.0.0.1:8101` reaches their local SAM3 service. The local policy HTTP endpoint forwards model requests through RLinf channels to the two model nodes. There is no requirement to expose SAM3 across nodes.

## Launch, inspect and resume

After filling the local profile, launch it from the head node:

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
cd "$STACK_ROOT/repos/RLinf"
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local
```

Hydra overrides follow the config name, for example `rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B-SFT"`. Append `--cfg job --resolve` to inspect the resolved configuration without starting training. Use `runner.resume_dir=/path/to/saved/checkpoint` to resume a complete SAO checkpoint, retaining both model/optimizer states and its buffered/pending rollout state. A synchronous PPO checkpoint cannot provide the SAO queue state. Actor/critic checkpoints and logs live under `runner.output_dir`/the experiment name; episode requests, tool images and results live under `tools.rpent.output_dir`.

Check finite losses/gradients, nonzero parameter updates, actual replica activity, task success and infrastructure errors. Use Hydra overrides for an isolated short smoke run, rather than maintaining another smoke YAML. A successful short smoke test establishes connectivity and optimizer execution; it does not establish that the longest 262144-token workload fits in memory. Keep failure evidence and diagnose memory limits before changing sharding or budgets.

Evaluation uses the separate explicit entrypoint:

```bash
bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl
```

Set its model, dataset, interpreter, benchmark config and placement for the deployment. The original no-argument launch behavior and original LIBERO YAML files remain available unchanged.
