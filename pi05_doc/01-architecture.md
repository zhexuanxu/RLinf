# 满血版 Pi0.5 架构设计

## 类继承关系

```
PI0Pytorch (openpi 库)
    └── OpenPi0ForRLActionPrediction (残血版 + RL 基础设施)
            └── OpenPi05FullForRLActionPrediction (满血版)
```

满血版继承了残血版的所有 RL 功能：
- Value head（PPO 价值函数）
- 多种 noise method（flow_sde, flow_ode, flow_cps, flow_noise）
- DSRL 组件
- NFT 支持
- FSDP 分布式训练

## 三种训练模式

模式通过数据自动检测，无需手动指定：

| 模式 | 条件 | 损失函数 |
|------|------|---------|
| VLM-only | `actions=None`, `token_loss_mask.any()=True` | CE loss on language |
| VLA-only | `actions≠None`, `token_loss_mask.any()=False` | Flow matching on actions |
| VLM+VLA | `actions≠None`, `token_loss_mask.any()=True` | CE + flow matching |

## 三个关键 Mask

| Mask | 形状 | 含义 |
|------|------|------|
| `token_ar_mask` | `[B, L]` int | 0=双向注意力（prompt），1=因果注意力（CoT response） |
| `token_loss_mask` | `[B, L]` bool | True=对该 token 计算 CE loss |
| `token_kv_cache_mask` | `[B, L]` bool | True=该 token 在 action expert 的 KV cache 中可见，EOS 设为 False |

### 注意力模式示例

```
tokens:  [IMG, IMG, "pick", "up", "ans", "red", "cup", EOS]
ar_mask: [  0,   0,     0,    0,     1,     1,     1,    1]
cumsum:  [  0,   0,     0,    0,     1,     2,     3,    4]

注意力规则：token i 可以看到 token j 当且仅当 cumsum[j] <= cumsum[i]
- 所有 prompt tokens (cumsum=0) 彼此可见（双向）
- "ans" (cumsum=1) 看到所有 prompt 但开始因果链
- 后续 response tokens 因果自回归
```

### EOS 处理

- **训练时**：EOS 有 `loss_mask=True`（模型学习生成 EOS），但 `kv_cache_mask=False`（action expert 不 attend 到 EOS）
- **推理时**：`generate_language()` 生成 EOS 后停止，但 EOS **不加入** KV cache，确保 action sampling 时一致

## 核心方法

### `sft_forward(data)` — 统一 SFT 入口

```python
def sft_forward(self, data, **kwargs):
    observation = data["observation"]
    actions = data.get("actions", None)
    token_kv_cache_mask = data.get("token_kv_cache_mask", None)
    
    # 自动检测模式
    if not has_actions:
        return self._forward_vlm(...)    # VLM-only
    return self._forward_vla_full(...)   # VLA 或 VLM+VLA
```

### `_forward_vlm(...)` — VLM-only 前向

1. `embed_prefix_with_ar_mask()` 嵌入 prefix（图像 + 语言 token，带 ar_mask）
2. 截断最后一个 token（next-token prediction）
3. 通过 PaliGemma forward，`inputs_embeds=[prefix_embs, None]`（无 suffix）
4. `_compute_ce_loss()` 计算 CE loss

### `_forward_vla_full(...)` — VLA / VLM+VLA 前向

1. 采样 noise、time，计算 `x_t` 和 `u_t`
2. `_compute_velocity_with_prefix_out()` 做一次联合 prefix+suffix forward
3. Flow matching loss: `MSE(u_t, v_t)`
4. 可选 CE loss: `_compute_ce_loss(prefix_out, ...)`
5. 总 loss: `language_loss_weight * ce_loss + action_loss_weight * flow_loss`

### `generate_language(observation, ...)` — 自回归生成

1. 嵌入 prefix，使用 `left_to_right_align` 处理 batch
2. 初始化 `StaticKVCache`（预分配内存）
3. Prefill：一次 forward 填充 cache
4. Decode loop：逐 token 生成，更新 cache
5. EOS 不加入 cache，返回 `(tokens, past_kv, full_pad_mask, full_ar_mask)`

### `sample_with_reasoning(...)` — 先想再做

1. `generate_language()` → CoT tokens + KV cache（不含 EOS）
2. `_euler_sample_with_cache()` → 基于 KV cache 采样 action

### `predict_action_batch(env_obs, ...)` — RL 推理

- `forward_mode="vla"`: 直接调用父类（残血版行为）
- `forward_mode="vlm_vla"`:
  1. `sample_with_reasoning()` 生成 CoT + action（用于环境交互）
  2. `super().sample_actions()` 收集 PPO 训练数据（不含 CoT context）
  3. PPO 只训练 action，不训练 CoT 文本

## ForwardType 枚举

`rlinf/models/embodiment/base_policy.py` 的 `ForwardType` 用于在 FSDP-wrapped 模型上
分发不同行为（SFT 训练、PPO 推理、语言生成等）。**关键约束**：所有需要 forward 的
路径都必须通过 `model(forward_type=...)` 入口，不能直接调用模型方法或子模块的
`forward()`，否则会破坏 FSDP `_lazy_init` 的根状态判定。

| ForwardType | 用途 | 实现方法 |
|-------------|------|---------|
| `DEFAULT` | 默认 | `default_forward()` |
| `SFT` | SFT 训练（VLM/VLA/VLM+VLA） | `sft_forward(data)` |
| `NFT` | NFT 训练 | `forward_nft()` |
| `SAC` / `SAC_Q` | SAC 训练 | `sac_forward()` / `sac_q_forward()` |
| `GENERATE_LANGUAGE` | **VLM eval** 自回归生成 | `generate_language(observation, max_new_tokens, temperature)` |

详细原理见 `04-sft-eval.md` 中 "ForwardType.GENERATE_LANGUAGE —— FSDP 不能绕过外层" 一节。
