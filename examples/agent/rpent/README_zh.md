<!-- sphinx-body -->

# RPent 强化学习

本示例使用 RPent 执行 LIBERO 任务，并通过 RLinf 根据环境反馈训练多模态 Qwen3.5 policy。

## 选择工作流

启动器默认仍为原版 `libero`。`train_qwen3.5_4b_ppo` 和 `eval_qwen3.5_4b` 保留原来的 Pi0.5、memory、纯语言模型配置；相关用法见{doc}`原版 LIBERO 示例 </rst_source/examples/agentic/rpent>`。

独立的 `libero_rl` 使用**完整多模态** Qwen checkpoint、adaptive 观测，以及六个工具：`move_to`、`set_gripper`、`view_env`、`segment`、`back_project`、`finish`。它需要 SAM3，无须 Pi0.5 或 memory 服务。必须显式选择以下配置：

| 配置名，不含 `.yaml` | 用途 |
|---|---|
| `train_qwen3.5_4b_libero_rl_ppo` | 同步 PPO 模板 |
| `train_qwen3.5_4b_libero_rl_sao` | 异步 SAO 模板 |
| `train_qwen3.5_4b_libero_rl_sao_local` | 含本地部署路径的最终启动配置 |
| `train_qwen3.5_4b_libero_rl_sao_3node` | 两台模型节点共享 16 卡运行 SAO，另加一台环境节点 |
| `eval_qwen3.5_4b_libero_rl` | 多模态评测模板 |

## 安装代码与环境

使用 Linux NVIDIA 主机，准备兼容驱动、CUDA 12.8 toolkit、EGL/OpenGL 库、Git 和 `uv`。选择共享根目录；无共享存储时，各节点需保持相同路径。下列功能分支包含此集成的两端实现，不能假设其他官方 main 版本也包含这些模块。

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

安装器复用已有的 `MEGATRON_PATH`。它在解析依赖时会临时修改依赖元数据，因此请使用空闲 checkout 和新环境。`--no-root` 要求原生系统依赖已准备好；`UNINSTALL_FA4=1` 选择经过验证的 FA2 路径。其他平台参数见 `bash requirements/install.sh --help`。

从匹配的 RPent 分支安装独立的 simulator/SAM3 环境：

```bash
cd "$STACK_ROOT/repos/RPent"
uv venv --python 3.12 "$STACK_ROOT/envs/rpent"
source "$STACK_ROOT/envs/rpent/bin/activate"
uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
```

以下记录已验证的兼容组合，不保证未来任意一次依赖解析都会得到完全相同的环境。

| 组件 | 已验证版本 |
|---|---|
| RLinf | Python 3.12.12；Torch 2.11.0+cu128；Transformers 5.6.1；SGLang 0.5.12.post1 |
| 训练内核 | Megatron Core 0.18.3，源码 commit 如上；Bridge 0.5.0；Transformer Engine 2.17.0；适配 Torch 2.11 的 FlashAttention 2.8.3 |
| Qwen 线性注意力 | flash-linear-attention/fla-core 0.6.0；TileLang 0.1.8；Triton 3.6.0 |
| 环境模拟器 | Torch 2.11.0+cu128；Transformers 4.57.6；NumPy 1.26.4；MuJoCo 3.3.0；robosuite 1.5.2；rpent-libero/rpent-liberopro 0.2.0；SAM3 0.1.0 |

使用 `python -m pip freeze`、`python -c 'import torch; print(torch.__version__, torch.version.cuda)'` 和 `git -C "$MEGATRON_PATH" rev-parse HEAD` 核对并在验证后保留版本清单。源码编译需要匹配的 CUDA 编译器；可联网的纯 CPU 节点适合下载 wheels/源码，但不一定能编译 CUDA 扩展。离线计算节点应先在兼容 Linux/Python/CUDA 构建机准备完整依赖，再传输 wheels 和 Git 源码，通过 `uv pip install --no-index --find-links /path/to/wheelhouse ...` 安装。部分 wheel 缓存不等于完整离线安装包。

所有 Ray 节点必须有相同的 RLinf 环境、源码和模型文件。RPent 保留独立环境，不要为了统一版本把 RLinf 的 Transformers 升级为 RPent 版本。若使用自定义激活脚本，保留安装器设置的 Megatron、CUDA、NCCL 路径。本方案使用 FA2 并关闭 Torch 编译，在启动 Ray 前设置 `NVTE_FLASH_ATTN=1`、`NVTE_FUSED_ATTN=0`、`TORCH_COMPILE_DISABLE=1`。环境节点还需要可用的 EGL。

## 准备模型和标准 LIBERO 资产

将完整 Qwen3.5-4B 或兼容的全参 SFT 导出放到 `$STACK_ROOT/models/`。其中必须包括视觉权重、tokenizer、processor 和 chat template；纯语言抽取模型不能处理 adaptive 带图回合。例如在联网机器下载基础模型：

```bash
hf download Qwen/Qwen3.5-4B --local-dir "$STACK_ROOT/models/Qwen3.5-4B"
```

按照 [SAM3 仓库](https://github.com/facebookresearch/sam3)说明获取权重，并将 `SAM3_CHECKPOINT_PATH` 指向本地 `sam3.pt`。Python 包不包含模拟器资产。在 RPent 环境下载标准资产并生成独立 benchmark 配置：

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

LIBERO 的部分 XML/mesh 路径直接相对安装包构造，独立于 config.yaml，所以还必须补齐包内 assets 链接。已有完整资产目录时保留；遇到断链应修复路径，不替换场景几何。保留 `get_default_path_dict()` 返回的包内 benchmark、BDDL 和初始状态路径。`sao_3node` 配置把此路径显式传给 episode 子进程，避免父环境中的 LIBERO-PRO 配置悄悄替换标准 benchmark。

继续使用 RPent 解释器生成标准 40-task 训练清单：

```bash
cd "$STACK_ROOT/repos/RLinf"
python examples/agent/rpent/prepare_tasks.py --workflow libero_rl \
  --libero-type standard --libero-config-path "$LIBERO_CONFIG_PATH" \
  --seeds 1-32 --output-dir "$STACK_ROOT/data/libero-rl-tasks"
```

生成 1,280 行及记录 task/初始状态数量、seeds、哈希的 manifest。seed 0 保留评测。生成器拒绝越过各任务初始状态范围的 seed，避免取模后重复同一初始状态。可使用 `--seeds 33-39 --split dev` 或 `--seeds 40-49 --split test`，为各 split 指定独立输出目录。原来的 memory 清单生成器仍为默认 `--workflow libero`，仍要求 `--memory-dir`。

## 配置共享 16 卡 SAO

Clean workflow 只保留四份依次继承的训练配置：`libero_rl_ppo` → `libero_rl_sao` → `libero_rl_sao_3node` → `libero_rl_sao_local`，完整文件名均带 `train_qwen3.5_4b_` 前缀。第一份继承原版 `train_qwen3.5_4b_ppo`，原版 LIBERO 集成仍可使用该基础配置。任务、采样和模型参数放在 clean PPO 中；SAO 只覆盖更新规则，三节点配置只覆盖 placement，local 配置只填写部署路径。

根据自己的安装位置编辑 `config/train_qwen3.5_4b_libero_rl_sao_local.yaml`，填写完整多模态模型、训练 JSONL、RPent 解释器、LIBERO 配置、episode 目录和训练输出目录。父配置将这些路径标记为必填，不提供特定机器的默认路径。`rollout.model.model_path` 同时设置 actor/critic 的 tokenizer 和 HF 导入路径。

| 资源或参数 | 配置 |
|---|---|
| 节点 0–1，GPU 0–15 | Rollout、critic、actor 分阶段共享全部 16 卡，阶段间 offload 权重、梯度和优化器 |
| Learner | TP2、CP1、DP8；开启 sequence parallel；BF16；activation recomputation；response-only logits |
| Rollout | 16 个 TP1 SGLang 副本；`replica_affinity=false`；每副本 `max_running_requests=128`；显存比例 0.7 |
| 节点 2，GPU 0–7 | 八个 agent worker，每个 32 个 episode 槽位；GPU 0 上一个 SAM3 服务 |
| 完成顺序调度 | Batch 128；`max_in_flight_rollouts=null` 和 `rollout_buffer_capacity=null` 分别解析为 256 |
| 训练预算 | 10 个数据 epoch，无额外 step 上限；1,280 条任务对应 100 次 learner 更新；每五次保存 |
| 优化 | 每次 learner 更新使用一个 actor minibatch、两个 critic update epoch；microbatch 1；actor LR `1e-6`、critic LR `5e-6`；训练语言参数，冻结视觉编码器/projector |
| Episode | Simple prompt、adaptive 观测、thinking、100 turns、10,000 env steps、四小时预算 |
| 生成 | Context 262144，output 32768；temperature 1、top-p 0.95、top-k 20、presence penalty 1.5、repetition penalty 1.0 |
| Context management | 原生消息窗口 80→40，触发比例 0.7；不额外清除工具结果或生成摘要 |

SAO 允许最多 256 条在途 episode，按完成顺序收集下一批 128 条轨迹，不再等待某个固定同步 batch 的全部轨迹。Episode 槽位包含等待模型与执行工具的时间；八个 worker 各 32 个槽位，足以同时展开全部 256 条在途轨迹。Buffer 保存已完成轨迹；额外排队的 prompt 不会提高在途上限。每个 episode 通常只有一个未完成模型请求，因此每副本 128 是准入上限，并不表示每张 GPU 始终都有 128 个活跃请求。

共享 GPU 仍需切换阶段：暂停并收回模型请求、卸载 rollout 显存、依次训练 critic 和 actor、同步权重，再恢复等待中的请求。训练时环境动作可以继续，但模型生成暂停。SAO 的学习规则也有变化：DIS actor loss、0.3/5.0 clip 边界、alpha 为 1.5 的 `sao_gae` 和 SAO critic 目标；它直接使用 rollout 保存的 behavior logprob，不重算旧 policy 的 logprob。任务、模型、prompt、采样和压缩参数均继承 clean PPO。

工具结果和图片保留在后续输入中。Policy loss 只作用于 policy 生成的 assistant token，user/tool/history token 不参与监督。逐 token 前缀兼容的连续 turn 合并，压缩或历史改写后分段。Actor 和 critic 使用相同的 response 位置映射；response-only logits 要求 CP1。

## 启动三节点 Ray 和 SAM3

**每个节点**启动 Ray 前都先 source RLinf 环境，导出 `MEGATRON_PATH`、`NVTE_FLASH_ATTN=1`、`NVTE_FUSED_ATTN=0`、`TORCH_COMPILE_DISABLE=1`，并设置足够的 `RLINF_TIMEOUT`，例如 `10800`。按实际网络选择网卡，`RLINF_COMM_NET_DEVICES=eth0` 仅为示例。Ray worker 在启动时继承这些环境变量。

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
export HEAD_IP=10.0.0.10  # 各节点可达的 head 地址
export RLINF_NODE_RANK=0
ray start --head --port 6387 --node-ip-address="$HEAD_IP" --num-gpus 8
```

模型节点 1 和环境节点 2 使用同样的源码/环境，各自填写 rank 和本地地址：

```bash
export RLINF_NODE_RANK=1  # 环境节点使用 2
export NODE_IP=10.0.0.11 # 环境节点填写自己的地址
ray start --address="$HEAD_IP:6387" --node-ip-address="$NODE_IP" --num-gpus 8
```

通过 `ray status --address="$HEAD_IP:6387"` 确认三节点、24 GPU。不要在已占用节点上启动冲突的第二个 Ray 集群。长训练前验证多节点 NCCL 通信；防火墙、网卡和 RDMA 设置取决于集群。

在节点 2 的独立终端使用 RPent 环境启动 SAM3：

```bash
source "$STACK_ROOT/envs/rpent/bin/activate"
cd "$STACK_ROOT/repos/RPent"
export SAM3_CHECKPOINT_PATH="$STACK_ROOT/models/sam3/sam3.pt"
python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8101
```

Agent worker 和 episode 子进程都位于节点 2，因此 `http://127.0.0.1:8101` 连接本地 SAM3。策略的本地 HTTP endpoint 通过 RLinf channel 将模型请求发往两台模型节点，无须跨节点暴露 SAM3。

## 启动、检查与恢复

填好 local 配置后，在 head 节点执行：

```bash
source "$STACK_ROOT/envs/rlinf/bin/activate"
cd "$STACK_ROOT/repos/RLinf"
bash examples/agent/rpent/run_train.sh train_qwen3.5_4b_libero_rl_sao_local
```

Hydra 覆盖参数放在配置名之后，例如 `rollout.model.model_path="$STACK_ROOT/models/Qwen3.5-4B-SFT"`。末尾加 `--cfg job --resolve` 可以只检查最终配置而不启动训练。使用 `runner.resume_dir=/path/to/saved/checkpoint` 从完整 SAO checkpoint 恢复两套模型/优化器与 buffered/pending rollout 状态。同步 PPO checkpoint 不包含 SAO 所需的队列状态。Actor/critic checkpoint 和日志位于 `runner.output_dir` 下的实验目录；episode 请求、工具图片和结果位于 `tools.rpent.output_dir`。

使用 Hydra 覆盖参数运行独立短 smoke，无须单独维护 smoke YAML。检查 loss/梯度是否有限、参数是否实际更新、模型副本活动、任务成功率及基础设施错误。短 smoke 通过证明连通和优化器执行，不代表最长 262144-token 工作负载一定能放入显存。遇到显存限制先保留证据并诊断，再决定切分或预算调整。

评测使用独立的显式入口：

```bash
bash examples/agent/rpent/run_eval.sh eval_qwen3.5_4b_libero_rl
```

按部署填写其模型、数据、解释器、benchmark 配置和 placement。原版不传配置名的启动行为及原版 LIBERO YAML 继续保持不变。
