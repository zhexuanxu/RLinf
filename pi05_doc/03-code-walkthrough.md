# 满血版 Pi0.5 代码走读

## 文件清单

### 新增文件

```
rlinf/models/embodiment/openpi/openpi_full_pi05_model.py   # 核心模型子类 (~530 行)
rlinf/models/embodiment/openpi/static_kv_cache.py          # KV cache (~170 行)
rlinf/models/embodiment/openpi/dataconfig/cot_transform.py  # CoT 数据变换 (~120 行)
rlinf/data/datasets/pi05_vlm_dataset.py                     # Robo2VLM 数据集 (~200 行)
examples/sft/config/robotwin_sft_openpi_pi05_vlm.yaml       # VLM-only SFT 配置
examples/sft/config/robotwin_sft_openpi_pi05_vlm_vla.yaml   # VLM+VLA SFT 配置
examples/embodiment/config/behavior_ppo_openpi_pi05_full.yaml  # PPO 配置
pi05_doc/                                                    # 文档目录
```

### 修改文件

```
rlinf/models/embodiment/openpi/openpi_action_model.py  # OpenPi0Config 增加 7 个字段
rlinf/models/embodiment/openpi/__init__.py             # get_model() 路由 + VLM skip norm_stats
rlinf/models/embodiment/openpi/openpi_full_pi05_model.py  # forward() 增加 GENERATE_LANGUAGE dispatch
rlinf/models/embodiment/base_policy.py                 # ForwardType 增加 GENERATE_LANGUAGE
rlinf/workers/sft/fsdp_vla_sft_worker.py               # 训练 + eval 实现（含 generate_language 调用）
rlinf/data/datasets/pi05_vlm_dataset.py                # eval_mode + 3-tuple + 配置驱动列名
rlinf/runners/sft_runner.py                            # val_at_step_0 训练前 baseline eval 钩子
examples/embodiment/config/model/pi0_5.yaml            # 增加 full_pi05 配置字段
examples/sft/config/model/pi0_5.yaml                   # 增加 full_pi05 配置字段
```

## 关键代码路径

### SFT VLM-only 完整代码路径

```
run_vla_sft.sh robotwin_sft_openpi_pi05_vlm
    ↓
examples/sft/train_vla_sft.py
    ├─ Hydra loads robotwin_sft_openpi_pi05_vlm.yaml
    ├─ Creates FSDPVlaSftWorker
    └─ Creates SFTRunner → runner.run()
    
FSDPVlaSftWorker.__init__()
    ├─ build_model(): get_model(cfg)
    │   ├─ full_pi05=True → OpenPi05FullForRLActionPrediction(config)
    │   ├─ forward_mode="vlm" → 跳过 norm_stats 加载
    │   └─ setup_wrappers([], [])  # 空 transforms
    │
    └─ build_dataloader(train_data_paths)
        ├─ full_pi05=True, forward_mode="vlm" → _build_pi05_vlm_dataloader()
        │   ├─ Pi05VLMDataset(data_dir, max_token_len=200, num_images=1)
        │   │   └─ 加载 Robo2VLM parquet，每条数据生成:
        │   │       image → 224x224 float32 [-1,1]
        │   │       question+answer → PaliGemma tokenized
        │   │       token_ar_mask, token_loss_mask, token_kv_cache_mask
        │   └─ DataLoader(collate_fn=pi05_vlm_collate_fn)
        └─ 返回 (data_loader, config)

Training loop (SFTRunner.run + FSDPSftWorker.run_training):
    # 训练前 baseline eval（如果 val_at_step_0=True）
    if start_step==0 and cfg.runner.val_at_step_0:
        actor.run_eval()  # 见下方 Eval 路径
    
    for step in range(max_steps):
        # 训练
        for each gradient_accumulation step:
            batch = next(data_iter)  # (observation_dict, None, meta_dict)
            ↓
            get_train_model_output(batch)
                # 兼容 2-tuple（openpi）和 3-tuple（pi05_vlm）
                if len(batch) == 3: observation, actions, _meta = batch
                else: observation, actions = batch
                token_kv_cache_mask = observation.pop("token_kv_cache_mask")
                observation → GPU tensors
                ↓
                model(forward_type=ForwardType.SFT, 
                      data={"observation": obs, "actions": None, 
                            "token_kv_cache_mask": kv_mask})
                ↓
                OpenPi05Full.sft_forward(data) → _forward_vlm → CE loss dict
            ↓
            loss.backward()
        optimizer.step()
        
        # 周期 eval
        if step % val_check_interval == 0:
            actor.run_eval()  # 见下方 Eval 路径
```

### SFT VLM-only Eval 路径（`val_check_interval > 0`）

```
SFTRunner.run() → actor.run_eval()
    ↓
FSDPVlaSftWorker.run_eval()
    ├── 重置 _eval_print_remaining = print_eval_samples (rank 0)
    └── super().run_eval()  # FSDPSftWorker
          ├── self.model.eval()
          └── for batch in eval_data_loader:
                correct += self.get_eval_model_output(batch)
                ↓
                # 1) 解包 3-tuple，pop kv_cache_mask, tensors → GPU
                observation, _, meta_list = batch
                obs_obj = _model.Observation.from_dict(observation)
                
                # 2) 通过外层 forward 分发 GENERATE_LANGUAGE（避免 FSDP 子模块 _is_root 冲突）
                with no_grad(), self.amp_context:
                    out_tokens, *_ = self.model(
                        forward_type=ForwardType.GENERATE_LANGUAGE,
                        observation=obs_obj,
                        max_new_tokens=cfg.openpi.max_language_len,
                        temperature=0.0,
                    )
                # → forward() 路由到 generate_language() → StaticKVCache 自回归
                
                # 3) 解码并对比
                for i, meta in enumerate(meta_list):
                    gen_ids = out_tokens[i].tolist()
                    if EOS in gen_ids: gen_ids = gen_ids[:gen_ids.index(EOS)]
                    pred_text = sentencepiece.decode(gen_ids)
                    pred_letter = re.search(r"[A-F]", pred_text)
                    if pred_letter == meta["correct_answer_letter"]:
                        correct += 1
                    if rank==0 and _eval_print_remaining > 0:
                        _print_qa_sample(meta, pred_text, pred_letter)
                        _eval_print_remaining -= 1
                
                return correct
          ↓
          metrics = {"eval_accuracy": correct/total} → all_reduce(AVG)
```

详细见 [04-sft-eval.md](./04-sft-eval.md)。

### StaticKVCache 工作原理

```python
# 初始化：预分配全部内存
cache = StaticKVCache(
    max_batch_size=B,
    max_cache_len=prefill_size + max_new_tokens,
    num_layers=18,
    num_key_value_heads=1,
    head_dim=256,
)

# Prefill：一次写入全部 prefix
cache.update(key_states, value_states, layer_idx)
# 内部：index_copy_(2, write_positions, key_states)

# Decode：逐 token 更新
for step in range(max_new_tokens):
    token = sample_from_logits(last_logits)
    if token == EOS: break  # EOS 不加入 cache！
    cache.update(new_key, new_value, layer_idx)

# Action sampling：读取 cache（不更新）
cached_k, cached_v = cache[layer_idx]  # 返回 [:seen_tokens] 的 clone
```

### _compute_ce_loss 详解

```
输入序列：[img_0, ..., img_M-1, tok_0, ..., tok_N-1]
输出位置 i 预测位置 i+1 的 token

VLM mode (truncated_input=True):
  输入: [img_0..img_M-1, tok_0..tok_N-2]  (截掉最后一个 token)
  输出: [img_0..img_M-1, tok_0..tok_N-2]
  lang_out = prefix_out[:, M:]  → 预测 tok_1..tok_N-1

VLA mode (truncated_input=False):
  输入: [img_0..img_M-1, tok_0..tok_N-1]  (完整输入)
  输出: [img_0..img_M-1, tok_0..tok_N-1]
  lang_out = prefix_out[:, M:-1]  → 预测 tok_1..tok_N-1

targets = lang_tokens[:, 1:]      # 去掉第一个 token
loss_mask = token_loss_mask[:, 1:] # 同步 shift
logits = lm_head(lang_out)
ce_loss = cross_entropy(logits, targets) * loss_mask / sum(loss_mask)
```

## 设计决策记录

### 1. 子类而非修改父类
- 原因：最大化复用残血版代码，不影响现有功能
- 效果：`full_pi05=False` 时行为完全不变

### 2. CoT 仅推理，PPO 不训练文本
- 原因：用户要求。PPO 只训练 action chunk
- 实现：`predict_action_batch` 调用 `sample_with_reasoning` 获取 action 交给环境，但 PPO 训练数据来自父类 `sample_actions`（不含 CoT context）

### 3. stop_gradient_to_vlm 暂不实现
- 原因：openpi 的 PaliGemmaWithExpertModel.forward() 不支持 detach_prefix_for_suffix
- 影响：flow matching 的梯度会回传到 VLM（当 train_expert_only=False 时）
- 后续：需要时可修改 openpi 包或用两次 forward 实现

### 4. token_kv_cache_mask 通过 data dict 传递
- 原因：openpi 的 Observation dataclass 没有这个字段，不修改第三方包
- 实现：SFT worker 从 batch dict 提取 kv_cache_mask，通过 data dict 传给 sft_forward
- 默认值：如果缺失，fallback 到 tokenized_prompt_mask（所有 valid token 都可见）

### 5. ForwardType.GENERATE_LANGUAGE —— eval 必须走外层 forward
- 原因：FSDP `auto_wrap_policy` 会把 `paligemma_with_expert` 单独包成 FSDP 单元。
  直接调 `model.generate_language()` → `paligemma_with_expert.forward()` 会绕过外层
  FSDP，触发该子模块 `_lazy_init` 把它标记为 `_is_root=True`。后续训练 forward 进入
  外层 FSDP 的 `_lazy_init`，发现非 root 子模块的 `_is_root` 已被设置 → 断言失败
- 实现：`ForwardType` 增加 `GENERATE_LANGUAGE`，模型 `forward()` 路由到 generate_language。
  Worker 用 `model(forward_type=GENERATE_LANGUAGE, ...)` 调用
- 经验法则：FSDP 模型上的 inference 都通过 forward 入口分发，不要直调子模块

### 6. Pi05VLMDataset 3-tuple + meta dict
- 原因：eval 需要原始问题/答案文本用于打印和字符串比较；又不能在 Observation 里塞字符串
  字段（`Observation.from_dict()` 会丢弃未识别字段，且 collate 时无法 stack）
- 实现：返回 `(observation, actions, meta)` 3-tuple。训练模式 `meta={}`；eval 模式
  `meta={"question", "choices", "correct_answer_letter", "prompt_text"}`
- Worker 兼容：`get_train_model_output` 同时支持 2-tuple（openpi 数据加载器）和 3-tuple

### 7. val_at_step_0 在 SFTRunner 实现
- 原因：默认的 `check_progress(step, ...)` 只在 step 是 `val_check_interval` 倍数时触发
  eval（且 step 从 1 开始）。要在训练前看 baseline，需要在主循环外手动跑一次 eval
- 实现：`SFTRunner.run()` 在 `for _step in range(...)` 之前判断 `val_at_step_0`，
  调一次 `actor.run_eval()` 并把结果记为 step 0 的 eval 指标
- 通用性：默认 `val_at_step_0=False`，对其他 SFT config 无影响
