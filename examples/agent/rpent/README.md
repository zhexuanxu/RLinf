# RPent Multi-Turn Embodied GRPO Training (Harness v2)

This example trains Vision-Language-Action reasoning models (`Qwen3.5-4B` through `Qwen3.8-27B`) with **synchronous oversampled GRPO** or **collocated asynchronous GRPO** across up to 5 manipulation simulators (`libero`, `robotwin`, `robodojo`, `robolab`, `metaworld`; plus `robocasa`) served by [RPent](https://github.com/RLinf/RPent).

## Architecture & Config Matrix

### Core Modules

| File | Responsibility |
| :--- | :--- |
| `rlinf/runners/rpent_runner.py` | `RPentGRPORunner` (sync GRPO + oversampling + early straggler cancellation + collocated `/pool/{sleep,wake}`) and `RPentAsyncGRPORunner` (collocated/disaggregated async GRPO + startup capacity validation + per-step audit JSONs). |
| `rlinf/workers/rollout_buffer/rollout_buffer_worker.py` | Asynchronous `RolloutBuffer` with per-env guaranteed floor (`min_groups_per_task`), per-env concurrency caps (`max_inflight_per_task`), FCFS surplus filling, staleness filtering, and multi-task dataloader checkpointing. |
| `rlinf/agents/rpent/agent_loop.py` | Async multi-turn tool-calling agent loop (`RPentAgentLoopWorker`), binary `/open` & `/execute` HTTP client, validation retries, and typed infra-retry accounting. |
| `rlinf/agents/rpent/history.py` | Harness v2 `EpisodeConversation` (`base` / `dynamic` compaction preserving opening images) + `AppendOnlyHistory` (token-ID prefix caching + prefix-chain merging for Megatron training) using zero-Base64 raw `bytes`. |
| `rlinf/agents/rpent/simulators.py` | Multi-simulator registry, task JSONL parsing, and balanced per-domain sync/async quotas (`mixed_simulators`). |
| `rlinf/hybrid_engines/sglang/multimodal/{qwen_vl,replay}.py` | External SGLang multimodal processor (`SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE`) with O(1) LRU image-tensor caching, content-addressed tmpfs image staging, and safetensors replay payload capture — **zero SGLang source patches**. |
| `rlinf/hybrid_engines/megatron/qwen_vl_inputs.py` | Exact Qwen-VL multimodal input assembly for Megatron forward/backward without modifying Megatron-LM sources. |

### Available Harness v2 Configs (`examples/agent/rpent/config/`)

| Config | Model / Scale | Mode | Simulators | Key Parallelism & Batching |
| :--- | :--- | :--- | :--- | :--- |
| `train_qwen3.8_27b_rpent_async_grpo_5env.yaml` | **27B** (`6` RL nodes = `48× H100` + `1` dedicated 8-GPU Env node) | **Collocated Async GRPO** | All 5 (`libero`, `robotwin`, `robodojo`, `robolab`, `metaworld`) | Rollout `TP=2` (24 replicas), Actor `TP=4, SP=true, CP=2, DP=6`, `agentloop=0-17`, `bs=32 × G=4 = 128` (`4`/env floor + `12` FCFS) |
| `train_qwen3.5_4b_rpent_grpo.yaml` | **4B** (`1` node: GPUs `0–3` RL, `4–7` Sim/SAM3) | Sync GRPO + oversampling | `libero`, `robotwin` | Rollout `TP=1` (4 replicas), Actor `TP=2, SP=true, DP=2`, `bs=2 × G=8` (oversampled `4 × 8`) |
| `train_qwen3.5_4b_rpent_async_grpo.yaml` | **4B** (`1` node: GPUs `0–3` RL, `4–7` Sim/SAM3) | Collocated Async GRPO | `libero`, `robotwin` | Overlay on `train_qwen3.5_4b_rpent_grpo`: `pause_mode=retract`, `max_inflight=6`, `bs=4 × G=8` |
| `train_qwen3.5_4b_rpent_grpo_5env.yaml` | **4B** (Sync 5-Env) | Sync GRPO + oversampling | All 5 | Overlay on `train_qwen3.5_4b_rpent_grpo`: `bs=5 × G=8` (`1` group/env, oversampled `10 × 8`) |
| `train_qwen3.5_4b_rpent_grpo_48gpu.yaml` | **4B** (`4` RL nodes = `32× H100` + dedicated Sim nodes) | Sync GRPO + oversampling | `libero`, `robotwin` | Rollout `TP=1` (32 replicas), Actor `TP=2, DP=16`, `32 × 8` oversampled `48 × 8 = 384`, early straggler cancel, collocated `/pool/{sleep,wake}` |
| `train_qwen3.5_4b_rpent_robolab_smoke.yaml` | **4B** (`4` RL GPUs smoke test) | Sync GRPO | `robolab`, `robotwin` | `max_steps=1`, `bs=2 × G=2`, `seq_length=16384` |

## Harness v2 Protocol Highlights (F1–F5)

1. **Zero engine patches & end-to-end zero-Base64 binary image pipeline**:
   - SGLang loads `rlinf.hybrid_engines.sglang.multimodal` via `SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE` (zero Megatron/SGLang source edits).
   - Binary `/sessions/open` and `/sessions/{sid}/execute` frames stream raw PNG `bytes` directly to `RPentAgentLoopWorker`; `stage_local_images()` writes each unique image once to local tmpfs by content hash.
2. **Compact feedback (F1)**: `coordinates` and `position_unit` are emitted only on `initial_observation` and on `frame_version` changes; `execute_actions` returns compact per-item results plus one top-level `final_state`.
3. **Per-env default display resolution (F2)**: Setting `tools.rpent.image_size: null` displays `512×512` (`libero`/`robocasa`/`metaworld`), `768×432` (`robolab`), and native (`robotwin` `320×240`, `robodojo` `640×480`) while retaining native RGB-D for SAM3 and back-projection.
4. **Physical gripper `jaw_mm` (F3)**: `tools.rpent.include_jaw_mm: true` exposes measured inner-pad clearance in millimetres across all 6 simulators.
5. **Dynamic batch compaction (F4)**: `agentloop.history_compaction.strategy: dynamic` strips older intermediate images (`[image omitted]`) and older assistant `reasoning_content` at the token high-water mark while preserving `messages[0]` (including step=0 opening images) and full tool/action text, cleanly advancing `AppendOnlyHistory.chain_id` with RadixCache reuse afterward.
6. **Step-0 opening images & `execute_actions(..., observe=...)` (F5)**: `POST /sessions/open` returns step=0 all-camera views immediately; `execute_actions` supports `observe="none" | "all" | <camera>` to save an extra `view_env` turn when post-action visual feedback is needed.

---

## Quick Start

### 1. Prepare Task JSONL Pools

```bash
# LIBERO (pro)
python examples/agent/rpent/prepare_tasks.py \
  --output-dir "${RPENT_ROOT}/data/libero"

# RoboTwin (from RPent task/seed table)
python examples/agent/rpent/prepare_tasks.py \
  --robotwin-seeds /path/to/robotwin_seed_table.json \
  --output-dir "${RPENT_ROOT}/data/robotwin"

# MetaWorld (run inside MetaWorld Python 3.11 venv with RPent + RLinf on PYTHONPATH)
python examples/agent/rpent/prepare_tasks.py \
  --metaworld --seeds 0-49 \
  --output-dir "${RPENT_ROOT}/data/metaworld"

# RoboLab (from RPent/robots/robolab/task_catalog.json)
python examples/agent/rpent/prepare_tasks.py \
  --robolab-catalog "${RPENT_ROOT}/repos/RPent/robots/robolab/task_catalog.json" \
  --seeds 0-9 \
  --output-dir "${RPENT_ROOT}/data/robolab"
```

> **RoboDojo task JSONLs**: Ensure every RoboDojo manifest uses `seed ∈ {0, 1, 2}` (selects the `Eval_Layout/RoboDojo/arx_x5/<seed>/` directory) and `layout ∈ [0..19]` (selects `<task>_<layout>.json`). Swapping `seed` and `layout` causes immediate session-open failure.

### 2. Start Simulator & SAM3 Services

See `RPent/robots/rl/README.md` and `RPent/tools/rl/sim_services_5env_h100.example.yaml` (for 5-env 27B) or `RPent/tools/rl/sim_services.example.yaml` (for 8-GPU 4B):

```bash
ulimit -n 1048576
PYTHONPATH="${RPENT_ROOT}/repos/RPent" \
  python "${RPENT_ROOT}/repos/RPent/tools/rl/serve.py" \
  --config /path/to/sim_services_5env_h100.yaml
```

### 3. Launch 6-Node 27B 5-Env Collocated Async GRPO

Start a 6-node (`48× H100`) Ray cluster pinning the control-plane interface (`--node-ip-address=<control_ip>`) and `ulimit -n 1048576`, then run on the head node:

```bash
ulimit -n 1048576
export RPENT_ROOT=/path/to/rpent_ppo
export RPENT_ENV_HOST=<DEDICATED_SIM_NODE_IP>
export RPENT_MODEL_PATH="${RPENT_ROOT}/models/Qwen3.8-27B"
export RLINF_TIMEOUT=28800
export RAY_ADDRESS="<HEAD_CONTROL_IP>:6379"
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
# Optional RDMA / RoCEv2 overrides (forwarded to workers when set, omitted when unset):
# export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7
# export NCCL_IB_GID_INDEX=5
# export NCCL_CROSS_NIC=0
export NO_PROXY="127.0.0.1,localhost,<CLUSTER_CIDR>"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

export PATH="/opt/venv/reason/bin:/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/compat:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64"
export PYTHONPATH="${RPENT_ROOT}/envs/rl-overlay:${RPENT_ROOT}/repos/RLinf:${RPENT_ROOT}/repos/RPent:/opt/venv/reason/Megatron-LM"

python examples/agent/rpent/train.py \
  --config-name train_qwen3.8_27b_rpent_async_grpo_5env
```

For single-node 8-GPU 4B training (sync or async):

```bash
RPENT_ROOT=/path/to/rpent_ppo RPENT_MODEL_PATH=/path/to/Qwen3.5-4B-SFT \
  bash examples/agent/rpent/run_train.sh \
  --config-name train_qwen3.5_4b_rpent_async_grpo \
  runner.max_steps=100
```

---

## General Engineering Guide: Scaling to 27B+ Multi-Node Multimodal Async GRPO

The issues below are **general, recurring failure modes** when scaling multimodal embodied RL to 27B+ dense models across multi-node 80 GB GPUs and heterogeneous 5-simulator pools. Every fix below is built into `rpent-harness-v2`.

### 1. Collocated Weight-Sync OOM at `Rollout TP=1`

- **Symptom**: `torch.OutOfMemoryError` inside `sync_model_to_rollout` / `validate_weight_first_sync` (e.g., on `posi_norm` or reshard buffers) with `< 20 MiB` free VRAM on 80 GB H100s.
- **Root Cause**: A 27B BF16 model takes `~54 GiB` of weights alone. During collocated weight sync, SGLang `TP=1` holds `~54.8 GiB` while Megatron Actor (`TP=4, CP=2`) simultaneously holds `~24.3 GiB` of sharded weights and resharding staging buffers (`79.1 / 79.2 GiB` peak).
- **Fix & Rule of Thumb**: Use **`rollout.tensor_parallel_size: 2`** (NVLink pairs; `24` replicas across 48 GPUs) and bound `rollout.max_running_requests` (e.g. `12`). Halving rollout weight footprint to `~29 GiB`/GPU leaves **`~26 GiB` free headroom** during weight sync, doubles decode memory bandwidth, and expands KV/Mamba cache headroom for 64k-token multimodal rollouts.

### 2. Long-Context 27B Actor Parallelism & Optimizer Sharding (`TP=4, SP=true, CP=2`)

- **Symptom**: Megatron forward/backward OOM on 32k–64k multimodal sequences even with full activation recomputation.
- **Root Cause**: At `CP=1`, both long-sequence activations and distributed optimizer states concentrate on fewer ranks.
- **Fix & Rule of Thumb**: Match one Actor model-parallel replica to one 8-GPU NVLink node via **`TP=4, SP=true, CP=2, PP=1`** (`DP = total_gpus / 8`, e.g. `DP=6` on 48 GPUs), combined with `use_distributed_optimizer: true` (shards optimizer states across `DP × CP = 12` → `~6.75 GiB`/GPU), `recompute_granularity: full` (`uniform`, `1` layer), `offload_{weight,grad,optimizer}: true`, `empty_cache_after_backward: true`, and `freeze_vision_model: true` / `freeze_vision_projection: true`.

### 3. Uneven DP Batching & `CP > 1` Packed-Sequence Dummy Row Padding

- **Symptom 1**: `AssertionError` in `megatron_worker.py` / `init_num_microbatches_calculator` whenever `rollout_batch_size` (e.g., `16` or `32` groups) is not divisible by `actor DP` (e.g., `DP=6` on 48 GPUs, which has factor 3).
- **Symptom 2**: `RuntimeError: The size of tensor a (4) must match the size of tensor b (0)` inside `megatron.bridge` `preprocess_packed_seqs` on CP rank 1.
- **Root Cause**: Standard Megatron assumes `gbs % (mbs * dp_size) == 0`, and `pad_actor_batch_rows` previously appended `1`-token dummy rows that split into an empty `0`-length shard on CP rank 1 (since Context Parallelism + Sequence Parallelism require each sequence length to be a multiple of `2 * CP * TP`).
- **Fix Implemented**:
  - Under `actor.enable_dp_load_balance: true`, `initialize.py` and `megatron_worker.py` round `gbs` up to a multiple of `mbs * dp_size`, `distributed.py` performs greedy LPT bin-packing across uneven DP ranks (guaranteeing `>= 1` group per rank), and `loss_scales.py` applies uniform global scaling (`uneven_dp_groups=True`).
  - `pad_actor_batch_rows(..., min_row_len=tp * cp * 2)` pads dummy rows to `min_row_len` (`16` tokens for `TP=4, CP=2`) with `attention_mask=True`, `response_mask=False`, and zero advantage/loss scale so gradients are mathematically unchanged while CP slicing never sees an empty shard.

### 4. Async GRPO Idle-Worker Gloo Timeout (`Timed out waiting 10800000ms for recv`)

- **Symptom**: Training runs normally for hours, then crashes at **exactly `3h 00m`** (`10,800,000 ms`) with `RuntimeError: Timed out waiting 10800000ms for recv` on a subset of `AgentLoopGroup` ranks.
- **Root Cause**: RLinf defaults `RLINF_TIMEOUT` to `180` minutes (`3h`) on all Gloo/NCCL channels. When `num_agentloop_workers` (e.g., placing `agentloop: 0-47` = 48 workers) exceeds `rollout_buffer.max_inflight_groups` (e.g., `18` or `36`), the excess workers remain permanently idle and blocked on `dataloader_channel.get()` until the 3-hour Gloo receive timeout aborts the entire job.
- **Two-Layer Fix (Always Apply Both)**:
  1. **Size `agentloop` strictly `<= max_inflight_groups`**: Ideally `~max_inflight_groups / 2` (e.g., `cluster.component_placement.agentloop: 0-17` = `18` workers for `max_inflight_groups: 36` → ~2 groups/worker, `0%` idle workers). Because `RPentAgentLoopWorker` is async I/O-bound, 18 workers easily saturate 36+ concurrent groups.
  2. **Raise timeouts safely below `INT32_MAX` ms**: Export `RLINF_TIMEOUT=28800` (`20` days = `1.728 × 10⁹ ms < INT32_MAX = 2,147,483,647 ms`) in driver and all worker `env_vars`, and set `actor.megatron.distributed_timeout_minutes: 28800`. Avoid values `> 35791` minutes (such as `201600`), which overflow 32-bit `poll(int timeout_ms)` inside Linux Gloo.

### 4b. DAPO Token-Mean GRPO & BF16 Image Replay Storage / Prefetch / Disk Cleanup

- **Compaction-chain invariant GRPO loss**: All RPent configs use `algorithm.loss_scales: []` + `algorithm.use_valid_token_scale: true` (DAPO token-mean). Because every generated token across compaction chains receives equal global weight `1 / global_valid_tokens`, loss scaling is 100% invariant to whether an episode splits across 1 or multiple compaction chains (and `validate_rpent_async_cfg` rejects combining `turn_level` with `use_valid_token_scale: true`).
- **BF16 on-disk image replay + async CPU prefetch**: `rlinf/utils/image_replay.py` persists `pixel_values` in `bfloat16` (halving GPFS/SSD footprint vs FP32), while `actor.multimodal: {prefetch_batches: 2, prefetch_bytes: 4294967296, cleanup_after_step: true}` overlaps CPU safetensors decoding with GPU forward/backward.
- **Automatic replay directory cleanup**: `<output_dir>/<episode_id>/training_images` is automatically reclaimed across every lifecycle path — after `MAMegatronActorWorker.run_training()` (post-`torch.distributed.barrier()` on `is_data_io_rank`), on cancelled/failed episodes in `RPentAgentLoopWorker._run_episode()`, on unselected sync oversampling candidates in `admit_rollout_candidates()`, and on `equal_reward` / `stale` / proactive-eviction / `stop()` drops in `RolloutBufferWorker` — while strictly preserving any `record_artifacts: true` files in `<episode_id>/`. Set `actor.multimodal.cleanup_after_step: false` only when debugging saved replay tensors.

### 5. Multi-Env Async `RolloutBuffer` Balancing & Scaling Rules

- **Problem**: Across heterogeneous simulators, fast MuJoCo environments (`metaworld`, `libero`) complete 5–10× faster than IsaacSim environments (`robodojo`, `robolab`). Pure FCFS starves slow environments out of training batches; rigid equal quotas stall the entire GPU cluster on the single slowest episode.
- **Fix Implemented (`RolloutBufferWorker`)**:
  - **`min_groups_per_task`**: Guarantees a per-env floor in every batch (e.g., `4` groups/env × `5` envs = `20` groups out of `rollout_batch_size=32`).
  - **FCFS surplus slots**: Remaining batch slots (`12` groups) fill first-come-first-served from whichever environments complete earliest, keeping all 48 GPUs saturated while preserving 5-env coverage every step.
  - **`max_inflight_per_task`**: Caps concurrent inflight groups per env to match physical simulator slots (`libero:10`, `robotwin:8`, `robodojo:4`, `robolab:4`, `metaworld:10` → matches `40/32/16/16/40` worker slots at `group_size=4`).
  - **Batch-size scaling rule**: When doubling `rollout_batch_size` (e.g. `16 → 32`), scale `min_groups_per_task`, `max_inflight_groups` (`>= rollout_batch_size + buffer`, e.g. `36`), `max_buffered_groups` (`~2× rollout_batch_size`, e.g. `64`), and `max_dropped_groups_per_step` (`~10× rollout_batch_size`, e.g. `320`) proportionally so stale-group drops never stall batch admission.
  - **Collocated async requirement**: Always set `rollout.pause_mode: retract` (enforced by `validate_rpent_async_cfg`) so SGLang cleanly retracts inflight requests before offloading VRAM for Megatron backward.

### 6. Sparse Per-Env Metric Aggregation (`KeyError` in `aggregate_agent_metrics`)

- **Symptom**: `KeyError: '__mean__/rpent/<env>/success_rate'` at the end of an async training step.
- **Root Cause**: In multi-env async runs, an individual `AgentLoopWorker` may only complete episodes for a subset of the 5 environments within a step; assuming every worker dict contains every key crashes aggregation.
- **Fix Implemented**: `rlinf/utils/metric_utils.py` filters `[d[k] for d in metrics_list if k in d]` with a zero-denominator guard, and `async_agent_runner.py` logs per-env `batch_task_count/<env>`, `size/<env>`, and `inflight/<env>` buffer telemetry.

### 7. SGLang Multimodal, Hopper (H100), & CUDA Driver Stability

| Issue / Failure | Root Cause | Fix & Operational Guidance |
| :--- | :--- | :--- |
| **`AttributeError` in `rlinf/utils/patcher.py` on H100 (Hopper)** | Hopper's `deep_gemm` inserts non-module sentinel objects lacking `__dict__` into `sys.modules`. | Guarded `_Patcher` in `rlinf/utils/patcher.py` with `if not hasattr(value, "__dict__"): continue`. |
| **CUDA IPC weight sync fails on `535.x` drivers (`storage._share_cuda_()`)** | PyTorch expandable allocator segments are incompatible with CUDA IPC handle export on older drivers. | Set `PYTORCH_ALLOC_CONF=expandable_segments:False` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False` in rollout & actor `env_vars`. |
| **CUDA 12.8 / `cutlass.cute` / `libcuda` symbol mismatch** | Placing `/usr/lib/x86_64-linux-gnu` before `/usr/local/cuda/compat` loads older host `libcuda.so` on `535.x` nodes. | Put `/usr/local/cuda/compat` **first** in `LD_LIBRARY_PATH` for SGLang/Megatron RL workers, while keeping simulator pool venvs on native host `LD_LIBRARY_PATH` (never mix `compat` into EGL/Vulkan simulators). |
| **SGLang CUDA Graph hang / illegal memory access across offload/reload cycles** | Multimodal prefill/decode mixed with weight sync and memory saver can corrupt captured CUDA graphs on `535.x` + `torch 2.11+cu128`. | Set `rollout.enforce_eager: true` and `rollout.sglang.enable_memory_saver: true`. |
| **SGLang multimodal shared-buffer corruption** | SGLang's multimodal embedding ring buffer can collide under concurrent image batches and sleep/wake cycles. | Export `SGLANG_MM_BUFFER_SIZE_MB="0"` in `rollout.env_vars`. |
| **`json.decoder.JSONDecodeError` in Triton `l2norm_fwd_kernel`** | Abrupt pod termination leaves 0-byte `*.json` cache files under `/root/.triton/cache`. | Before launch, audit and purge 0-byte files (`find ~/.triton/cache -type f -size 0 -delete`). |
| **Transient SGLang `TCPStore EADDRINUSE` during 24+ replica launch** | Concurrent multi-replica startup can occasionally collide on ephemeral TCP ports. | Cleanly retry launch once `ray status` reports `0.0/<N> GPU` used and simulator pools report `active_sessions: 0`. |

### 8. Multi-Node Networking, File Descriptors, & Storage

| Issue / Failure | Root Cause | Fix & Operational Guidance |
| :--- | :--- | :--- |
| **Ray GCS binds RDMA NIC instead of control-plane NIC** | Multi-NIC H100 nodes default Ray to data-plane interfaces (`net1`), breaking cross-node GCS/control plane. | Start Ray on every node with `--node-ip-address=<control_eth_ip>`, set `NCCL_SOCKET_IFNAME=eth0` / `GLOO_SOCKET_IFNAME=eth0`, and configure RDMA via `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_CROSS_NIC=0`. |
| **`GenerateInput` `SIGABRT: Too many open files`** | Multi-node Ray/Gloo channels + concurrent HTTP connections open `>1,400` FDs, exceeding container default `ulimit -n 1024`. | Set `ulimit -n 1048576` in Ray node startup, training launch scripts, and simulator supervisors. |
| **`httpx.RemoteProtocolError: Server disconnected`** | Default `httpx` `keepalive_expiry=5.0` raced with `uvicorn`'s `timeout_keep_alive=5`. | `RPentAgentLoopWorker._client()` sets `keepalive_expiry=30.0` (strictly below RPent server's `300s`), sets `trust_env=False`, and retries transient transport failures (`max_infra_retries: 3`). |
| **Shared BeeGFS/NFS short-reads & container `/` overlay eviction** | Concurrent module/asset reads across dozens of workers trigger short-reads; staging into `/` container overlay can cause pod eviction. | When on affected shared filesystems, stage repos/venvs/assets into `/dev/shm/rpent_ppo` (RAM tmpfs) and point Ray `--temp-dir` there. |
| **Simulator-side issues (Vulkan `ERROR_DEVICE_LOST`, air-gapped `omniverse.toml`, IsaacSim VRAM budget, RoboLab missing contact prims, `PYTHONPATH` isolation)** | See dedicated simulator troubleshooting guide. | Follow the pre-flight checklist and pitfalls table in [`RPent/robots/rl/README.md`](https://github.com/zhexuanxu/RPent/blob/rpent-harness-v2/robots/rl/README.md). |
| **Single broken task or transient simulator HTTP 500 kills an entire multi-node async RL run** | `RPentAgentLoopWorker.run_one_query` previously re-raised `RPentEpisodeError` after exhausting retries, crashing the whole Ray training job on one bad scene/task. | In async mode (`runner.synchronous=false`), `rlinf/agents/rpent/agent_loop.py` converts post-retry `RPentEpisodeError` into `GroupDiscarded(reason=f"rpent_{exc.kind}")`, letting `RolloutBufferWorker` cleanly release the slot and schedule the next task while synchronous mode still fails fast. |

---

## Unit Tests

```bash
PYTHONPATH=/path/to/RLinf:/path/to/RPent:/path/to/Megatron-LM \
  pytest tests/unit_tests/test_rpent_grpo.py tests/unit_tests/test_rollout_buffer.py -v
```
