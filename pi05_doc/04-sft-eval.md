# Pi0.5 VLM-only SFT 评估流程

## 目标

在训练过程中定期评估 pi0.5 在 Robo2VLM 多选题（MCQ）数据集上的准确率，并支持：

- **训练前 baseline eval**（step 0），看未训练模型的初始表现
- **每 N 步 eval 一次**（`val_check_interval`），跟踪准确率曲线
- **打印 K 条 QA 对到终端**（`print_eval_samples`），人工查看模型预测

## 配置开关

`examples/sft/config/robotwin_sft_openpi_pi05_vlm.yaml`：

```yaml
runner:
  max_epochs: 20000
  val_check_interval: 200       # 每 200 步 eval 一次（必须 > 0 才启用 eval）
  save_interval: 2000           # 必须能被 val_check_interval 整除
  val_at_step_0: True           # 训练前 baseline eval
  print_eval_samples: 5         # rank 0 打印的 QA 对数量

data:
  type: vlm
  dataset_name: "pi05_robo2vlm"
  train_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/data"
  val_data_paths: "/mnt/public/xzxuan/data/Robo2VLM/eval_data"   # eval 数据
  prompt_key: "question"
  choice_key: "choices"
  answer_key: "correct_answer"
  image_keys: ["image"]
  max_token_len: 200
  num_workers: 4

actor:
  micro_batch_size: 4
  eval_batch_size: 4            # eval 时的 per-rank batch size
  global_batch_size: 256
```

## 数据格式（统一 3-tuple）

为了支持 eval 模式下携带原始问题/答案文本（用于打印和对比），`Pi05VLMDataset` 改为返回 **3-tuple**：

```python
# Pi05VLMDataset.__getitem__ 返回
(observation, actions, meta)

# pi05_vlm_collate_fn 返回
(batched_obs, None, list_of_meta_dicts)
```

| 模式 | observation | actions | meta |
|------|-------------|---------|------|
| `eval_mode=False` (训练) | tokens=`[prompt+answer+EOS+pad]`，`loss_mask` 覆盖 answer+EOS | `None` | `{}` |
| `eval_mode=True` (评估) | tokens=`[prompt+pad]`，`loss_mask` 全 False，`ar_mask` 全 0（双向 prefix） | `None` | `{question, choices, correct_answer_letter, prompt_text}` |

**Worker 兼容**：`get_train_model_output` 同时兼容 2-tuple（openpi 数据加载器）和 3-tuple（Pi05VLMDataset）。

## 调用链

### Step 0 baseline eval

```
SFTRunner.run()
  ├── if start_step==0 and val_at_step_0 and val_check_interval>0:
  │     actor.run_eval()             # 主进程
  │       └── FSDPVlaSftWorker.run_eval()  # 每个 rank
  │             ├── reset _eval_print_remaining = print_eval_samples (rank 0 only)
  │             └── super().run_eval()  # FSDPSftWorker
  │                   ├── self.model.eval()
  │                   ├── for batch in eval_data_loader:
  │                   │     correct += self.get_eval_model_output(batch)
  │                   └── return {"eval_accuracy": correct/total}  (all_reduce AVG)
  │
  └── for step in range(max_steps):
        actor.run_training()
        if step % val_check_interval == 0:
            actor.run_eval()         # 同上路径
```

### `get_eval_model_output(batch)` —— 核心评估逻辑

```python
1. observation, _, meta_list = batch    # 3-tuple

2. observation.pop("token_kv_cache_mask")  # 不在 Observation 中
3. observation = tree_map(to_tensor, observation)
4. obs_obj = _model.Observation.from_dict(observation)

5. with no_grad(), self.amp_context:
    # 通过 forward dispatch 调用 generate_language —— 见下方 FSDP 注意事项
    out_tokens, *_ = self.model(
        forward_type=ForwardType.GENERATE_LANGUAGE,
        observation=obs_obj,
        max_new_tokens=cfg.openpi.max_language_len,
        temperature=0.0,
    )

6. for i, meta in enumerate(meta_list):
    gen_ids = out_tokens[i].tolist()
    if EOS in gen_ids:
        gen_ids = gen_ids[:gen_ids.index(EOS)]
    pred_text = sentencepiece_tokenizer.decode(gen_ids)
    pred_letter = re.search(r"[A-F]", pred_text)  # 或 ""
    if pred_letter == meta["correct_answer_letter"]:
        correct += 1
    if rank==0 and _eval_print_remaining > 0:
        _print_qa_sample(meta, pred_text, pred_letter)
        _eval_print_remaining -= 1

7. return correct  # 父类 run_eval 累加，最后除以总数得到 eval_accuracy
```

## ForwardType.GENERATE_LANGUAGE —— FSDP 不能绕过外层

### 问题

最初实现是直接调用 `self.model.generate_language(obs_obj, ...)`。这导致训练第 1 步报错：

```
AssertionError: Non-root FSDP instance's `_is_root` should not have been set yet
```

### 原因

FSDP `auto_wrap_policy` 会把模型的内部子模块（比如 `paligemma_with_expert`）也包成 FSDP 单元。`generate_language()` 内部直接调用 `self.paligemma_with_expert.forward(...)`，**绕过了外层 FSDP 的 forward**。

- 第一次调用是 eval 阶段：直接调子模块 → 该子模块的 FSDP `_lazy_init` 把它本身标记为 `_is_root=True`
- 紧接着训练 forward `self.model(...)`：外层 FSDP `_lazy_init` 被触发，遍历子模块时发现"非 root 子模块的 `_is_root` 已被设置"，断言失败

### 解决

在 `ForwardType` 枚举里加一个 `GENERATE_LANGUAGE = "generate_language"`，模型 `forward()` 路由到该方法：

```python
# rlinf/models/embodiment/openpi/openpi_full_pi05_model.py
def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
    ...
    elif forward_type == ForwardType.GENERATE_LANGUAGE:
        return self.generate_language(**kwargs)
```

Worker 调用方式：

```python
self.model(forward_type=ForwardType.GENERATE_LANGUAGE, observation=obs_obj, ...)
```

这样调用走的是**最外层 FSDP 的 forward**，`_lazy_init` 在顶层正确触发，子模块被标记为非 root。后续训练 forward 不再冲突。

### 经验法则

> 任何在 FSDP 包装的模型上需要执行的 inference / generation，都应该通过 `model.forward(...)` 入口分发，**不要直接调用模型的内部方法或子模块**。

## 终端打印格式

eval 期间，rank 0 进程会按 `print_eval_samples` 数量打印 QA 对（仅文字，无图）：

```
================ EVAL SAMPLE (WRONG | gold=C | pred=?) ================
Q:       The robot is to remove the black marker from the pot and put it on the table. Has the robot successfully completed the task?
Choices: ['Task was not attempted' 'Yes' 'No' 'Cannot be determined']
Gold:    C
Pred:        (raw: 'No.')
============================================================
```

字段含义：
- `verdict`: `CORRECT` / `WRONG`，由 `pred_letter == gold_letter` 决定
- `Q:` / `Choices:` / `Gold:`: 来自 `meta`（dataset 在 eval 模式下保留的原始文本）
- `Pred:`: 从模型输出文本中正则提取的第一个 `[A-F]` 字母（无则显示空）
- `Raw:`: 模型实际生成的解码文本（包含可能的 "No."、"Answer: B"、空字符串等）

## 度量

每次 eval 通过 `all_reduce(AVG)` 聚合所有 rank：

```python
metrics = {"eval_accuracy": float(correct / total)}
# total = eval_step * eval_batch_size （per rank）
```

写入 TensorBoard：`eval/eval_accuracy`。

## 实现要点

### 1. `_pi05_tokenizer` 必须在 `super().__init__()` 之前初始化

```python
class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg):
        # parent __init__ 会调用 build_dataloader → _build_pi05_vlm_dataloader
        # 后者读 self._pi05_tokenizer，所以必须先于 super 设置
        self._pi05_tokenizer = None
        super().__init__(cfg)
        ...
```

### 2. `run_eval` 重置打印计数器

```python
def run_eval(self):
    self._eval_print_remaining = (
        cfg.runner.print_eval_samples if self._rank == 0 else 0
    )
    return super().run_eval()
```

每次 eval pass 开始时重置，所以每次 eval 都会打印 K 条样本（不会"打完一次就没了"）。

### 3. eval 模式下的 prompt-only tokens

```python
# Pi05VLMDataset.__getitem__ (eval_mode=True)
tokens[:prompt_len] = prompt_tokens   # 不拼 answer，不加 EOS
token_mask[:prompt_len] = True
ar_mask[:] = 0                         # 全双向（prefix）
loss_mask[:] = False                   # eval 不计算 loss
```

`generate_language()` 把这些 prompt tokens 当作生成的 prefix，从 prompt 末尾开始自回归。

### 4. EOS 截断

`generate_language` 可能在某些 rank 提前生成 EOS、其他 rank 继续。所以解码时按 per-sample 找到第一个 EOS 截断：

```python
gen_ids = out_tokens[i].tolist()
if eos_id in gen_ids:
    gen_ids = gen_ids[:gen_ids.index(eos_id)]
```

### 5. step 0 baseline 在 runner 实现

`rlinf/runners/sft_runner.py` 在主训练循环 `for _step in range(...)` **之前**插入一次 eval：

```python
def run(self):
    start_step = self.global_step
    if (start_step == 0 and cfg.runner.val_at_step_0
            and cfg.runner.val_check_interval > 0):
        eval_handle = self.actor.run_eval()
        eval_metrics = eval_handle.wait()
        self.metric_logger.log({f"eval/{k}": v for k, v in eval_metrics[0].items()}, 0)
        logger.info(f"[step 0 baseline eval] {evaluate_metrics}")
    ...
```

通用且对其他 SFT config 无影响（默认 `val_at_step_0=False`）。

## 常见问题

### Q: eval 跑完后训练第 1 步报 `_is_root` 断言？
说明你直接调用了 `model.generate_language(...)` 或其他子模块方法。改为 `model(forward_type=ForwardType.GENERATE_LANGUAGE, ...)`。

### Q: 打印的 EVAL SAMPLE 数量不对？
`print_eval_samples` 是 per-eval-pass 的预算。每次 eval 重置，rank 0 打印前 K 条。在 tee 日志里这些 print 可能与 tqdm 进度条 `\r` 重叠，肉眼看像被吞了，用 `grep -oc "EVAL SAMPLE"` 校验真实计数。

### Q: 为什么 baseline eval 准确率比随机基线还低？
未训练的 base 模型对"Answer:"提示后多倾向于继续生成自然语言（"No."、"Cannot be determined" 等），而不是单字母 A-F。我们的 `_extract_first_letter` 只看 `[A-F]` 出现，所以多数样本提取不到 letter（pred=`?`），自然命中率很低。训练几百步后模型会学到只输出单字母。

### Q: VLM+VLA 模式或残血版能否复用这套 eval？
当前实现仅支持 `full_pi05=True + forward_mode="vlm"`。其他模式调 `get_eval_model_output` 会显式抛 `NotImplementedError`。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `rlinf/models/embodiment/base_policy.py` | `ForwardType` 增加 `GENERATE_LANGUAGE` |
| `rlinf/models/embodiment/openpi/openpi_full_pi05_model.py` | `forward()` 路由 `GENERATE_LANGUAGE → generate_language` |
| `rlinf/data/datasets/pi05_vlm_dataset.py` | `eval_mode` 参数 + 3-tuple 输出 + 配置驱动列名 + `tokenizer` 公开 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | `_build_pi05_vlm_dataloader(eval_dataset)` + `get_eval_model_output()` + `run_eval()` 重置 + `get_train_model_output` 兼容 3-tuple |
| `rlinf/runners/sft_runner.py` | `val_at_step_0` 钩子（在主循环前 eval 一次） |
| `examples/sft/config/robotwin_sft_openpi_pi05_vlm.yaml` | 加 `val_data_paths`、`val_check_interval`、`val_at_step_0`、`print_eval_samples`、`eval_batch_size` 等字段 |
