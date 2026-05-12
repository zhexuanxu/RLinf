# BEHAVIOR-1K 评估文档

本目录包含在 RLinf 框架中对 BEHAVIOR-1K 家务操控基准进行评估的完整文档。

## 快速入门

选择评估方案：

1. **独立 VLA** -- 直接从图像和任务描述进行策略学习
   - 配置：`behavior_ppo_openpi_pi05.yaml`
   - 单模型，推理快速，端到端学习
   - [阅读指南 ->](vla_eval.md)

2. **VLM+VLA（双系统）** -- 带子任务生成的层次化规划
   - 配置：`behavior_ppo_openpi_agentic.yaml`
   - 两阶段流水线：VLM 生成子任务，VLA 执行子任务
   - 可选语言记忆模块用于多步推理
   - [阅读指南 ->](vlm_vla_eval.md)

## 文档结构

### 核心指南

| 文档 | 内容 |
|------|------|
| [vla_eval.md](vla_eval.md) | **独立 VLA 评估**<br/>端到端策略学习、执行流程、任务切换、场景初始化 |
| [vlm_vla_eval.md](vlm_vla_eval.md) | **VLM+VLA 评估**<br/>双系统架构、记忆模块、轨迹日志、VLM 模型选择 |
| [behavior_config.md](behavior_config.md) | **配置参考手册**<br/>全部 50 个 BEHAVIOR-1K 任务、场景初始化模式、YAML 配置项 |
| [data.md](data.md) | **数据参考手册**<br/>任务实例、挑战赛演示数据、目录结构、数据格式 |

### 关键主题

**切换任务**
- [VLA 指南：切换任务](vla_eval.md#切换任务)
- [VLM+VLA 指南：切换任务](vlm_vla_eval.md#切换任务)

**场景初始化**
- [disabled 模式（固定场景）](behavior_config.md#模式-1disabled默认--固定实例)
- [offline 模式（缓存场景）](behavior_config.md#模式-2offline--从缓存中随机采样)
- [online 模式（实时 BDDL 采样）](behavior_config.md#模式-3online--实时-bddl-采样)
- [下载与生成初始化样本](behavior_config.md#下载与生成初始化样本)

**配置**
- [全部配置字段](behavior_config.md#配置参考)
- [示例：多任务评估](behavior_config.md#示例多任务-offline-评估)

## 启动命令

### 独立 VLA（Pi0.5）

```bash
# 默认：turning_on_radio，固定初始化
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05

# 切换任务（通过 CLI 覆盖）
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture

# offline 采样场景
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.instance_resample_mode=offline \
  env.eval.omni_config.task.activity_instance_dir=/path/to/instances/
```

### VLM+VLA（双系统，Qwen2.5-VL）

```bash
# 默认：启用记忆模块
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic

# 切换任务
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=picking_up_trash

# 禁用记忆（无记忆 VLM 模式）
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.enable_memory=False

# 切换到 Qwen3-VL-Thinking
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  vlm.model.model_type=qwen3_vl_embodied \
  vlm.model.model_path=/path/to/Qwen3-VL-4B-Thinking
```

## 环境准备

### 前置条件

```bash
# 设置所需环境变量
export ISAAC_PATH=/path/to/isaac-sim
export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets

# 安装 Isaac Sim（如尚未安装）
curl https://download.isaacsim.omniverse.nvidia.com/isaac-sim-standalone-4.5.0-linux-x86_64.zip \
  -o isaac-sim.zip && unzip isaac-sim.zip

# 下载 BEHAVIOR 资源
python -c "from omnigibson.utils.asset_utils import download_behavior_1k_assets; download_behavior_1k_assets(accept_license=True)"

# 下载挑战赛任务实例（可选，用于 offline 评估）
python -c "from omnigibson.utils.asset_utils import download_2025_challenge_task_instances; download_2025_challenge_task_instances()"
```

### 模型下载

**OpenPI pi0.5**（VLA 和 VLM+VLA 均需要）：
```bash
huggingface-hub download RLinf/RLinf-Pi0-Behavior --local-dir RLinf-Pi0-Behavior
```

**Qwen2.5-VL**（VLM+VLA 需要）：
```bash
# 从 Hugging Face Hub 下载
huggingface-hub download Qwen/Qwen2.5-VL-3B-Instruct --local-dir Qwen2.5-VL
```

**Qwen3-VL-Thinking**（可选，用于 VLM+VLA）：
```bash
# 从 Hugging Face Hub 下载
huggingface-hub download Qwen/Qwen3-VL-4B-Thinking --local-dir Qwen3-VL-4B-Thinking
```

## 文件结构

```
dual_system/
├── README.md                 # 本文件
├── vla_eval.md              # 独立 VLA 评估指南
├── vlm_vla_eval.md          # VLM+VLA 评估指南
├── behavior_config.md       # 完整配置参考手册
└── data.md                  # 数据参考手册
```

## 常用任务示例

### Task 0: Turning On Radio
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_pi05 \
  env.eval.omni_config.task.activity_name=turning_on_radio
```

### Task 8: Rearranging Kitchen Furniture
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=rearranging_kitchen_furniture
```

### Task 20: Sorting Vegetables
```bash
bash examples/embodiment/eval_embodiment.sh behavior_ppo_openpi_agentic \
  env.eval.omni_config.task.activity_name=sorting_vegetables
```

完整的 50 个任务列表见 [behavior_config.md](behavior_config.md)。

## 故障排查

**OmniGibson 超时问题**
-> 见 [vla_eval.md: 故障排查](vla_eval.md#故障排查)

**VLM 内存溢出**
-> 见 [vlm_vla_eval.md: 故障排查](vlm_vla_eval.md#故障排查)

**找不到任务**
-> 在 [behavior_config.md](behavior_config.md#支持的任务049) 中核对任务名称

## 架构对比

| 方面 | VLA | VLM+VLA |
|------|-----|---------|
| **模型数量** | 1（OpenPI） | 2（Qwen VLM + OpenPI） |
| **推理速度** | 快（约 100 ms/步） | 较慢（约 500 ms/步） |
| **可解释性** | 低 | 高（子任务有日志记录） |
| **记忆模块** | 无 | 可选 MEM 模块 |
| **轨迹日志** | 仅指标 | 完整输入输出 + 图像 |
| **批量推理** | 单次 VLA 调用 | VLM 支持逐样本 prompt |

## 参考资料

- **BEHAVIOR-1K**: https://behavior.stanford.edu/
- **OpenPI**: https://huggingface.co/openvla/openvla-7b
- **Qwen VLM 系列**: https://huggingface.co/Qwen

---

**快捷链接**: [VLA 指南](vla_eval.md) | [VLM+VLA 指南](vlm_vla_eval.md) | [配置参考](behavior_config.md) | [数据参考](data.md)
