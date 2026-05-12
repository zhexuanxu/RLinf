# Pi0.5 SFT 评估说明

## 当前状态

VLA SFT 的 `get_eval_model_output` 目前返回 `NotImplementedError`，训练效果通过 loss 曲线监控。

如需评估 VLM 能力（如技能预测准确率），使用双系统评估流程，详见 `dual_system/vlm_vla_eval.md`。

## ForwardType.GENERATE_LANGUAGE —— FSDP 安全的推理路由

### 背景

pi0.5 满血版支持自回归语言生成（`generate_language()`），用于 CoT 推理和 VLM 评估。在 FSDP 环境下，必须通过特定方式调用。

### 问题

直接调用 `self.model.generate_language(obs_obj, ...)` 会导致训练报错：

```
AssertionError: Non-root FSDP instance's `_is_root` should not have been set yet
```

### 原因

FSDP `auto_wrap_policy` 会把模型内部子模块（如 `paligemma_with_expert`）也包成 FSDP 单元。`generate_language()` 内部直接调用 `self.paligemma_with_expert.forward(...)`，**绕过了外层 FSDP 的 forward**。

- eval 阶段直接调子模块 → 该子模块的 FSDP `_lazy_init` 把它标记为 `_is_root=True`
- 训练 forward `self.model(...)` → 外层 FSDP `_lazy_init` 发现非 root 子模块的 `_is_root` 已被设置 → 断言失败

### 解决方案

在 `ForwardType` 枚举里加 `GENERATE_LANGUAGE`，模型 `forward()` 路由到该方法：

```python
# rlinf/models/embodiment/openpi/openpi_full_pi05_model.py
def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
    ...
    elif forward_type == ForwardType.GENERATE_LANGUAGE:
        return self.generate_language(**kwargs)
```

调用方式：

```python
self.model(forward_type=ForwardType.GENERATE_LANGUAGE, observation=obs_obj, ...)
```

### 经验法则

> 任何在 FSDP 包装的模型上需要执行的 inference / generation，都应该通过 `model.forward(...)` 入口分发，**不要直接调用模型的内部方法或子模块**。

## 实现要点

### `_pi05_tokenizer` 初始化顺序

```python
class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg):
        # parent __init__ 会调用 build_dataloader → _build_pi05_vlm_dataloader
        # 后者读 self._pi05_tokenizer，所以必须先于 super 设置
        self._pi05_tokenizer = None
        super().__init__(cfg)
```

### Step 0 baseline eval 钩子

`rlinf/runners/sft_runner.py` 支持在主训练循环之前插入一次 eval（`val_at_step_0=True`）：

```python
def run(self):
    start_step = self.global_step
    if (start_step == 0 and cfg.runner.val_at_step_0
            and cfg.runner.val_check_interval > 0):
        eval_handle = self.actor.run_eval()
        eval_metrics = eval_handle.wait()
        ...
```

默认 `val_at_step_0=False`，对其他 SFT config 无影响。

## 涉及文件

| 文件 | 说明 |
|------|------|
| `rlinf/models/embodiment/base_policy.py` | `ForwardType` 包含 `GENERATE_LANGUAGE` |
| `rlinf/models/embodiment/openpi/openpi_full_pi05_model.py` | `forward()` 路由 `GENERATE_LANGUAGE → generate_language` |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | VLM-only 数据加载 + 训练 3-tuple 兼容 |
| `rlinf/runners/sft_runner.py` | `val_at_step_0` 钩子 |
