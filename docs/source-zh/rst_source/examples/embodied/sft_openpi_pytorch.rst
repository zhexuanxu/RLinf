基于 PyTorch OpenPI (Pi0.5) 的 BEHAVIOR 监督微调
================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 45%

   用于 PyTorch OpenPI SFT 的 π₀ 模型系列。来源：`Physical
   Intelligence <https://www.physicalintelligence.company/blog/pi0>`_。

在 BEHAVIOR 示范数据上微调数值对齐的 PyTorch π₀.₅ 实现。你可以训练仅输出动作的
VLA，也可以训练完整的 VLM-to-VLA 模型：模型先预测子任务，再让动作专家基于该预测
生成动作。

概览
----

选择单任务或 50 任务配方，并从同一个 π₀.₅ 基础模型开始训练。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      π₀.₅ VLA · π₀.₅ VLM-to-VLA

   .. grid-item-card:: 方法
      :text-align: center

      流匹配 · language CE

   .. grid-item-card:: 数据
      :text-align: center

      BEHAVIOR task 0 · 全部 50 个任务

   .. grid-item-card:: 硬件
      :text-align: center

      FSDP · bf16 计算

| **你将完成：** 准备数据与统计量 → 选择配置 → 启动 SFT → 转换 checkpoint 用于评估。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · 新格式 π₀.₅ 基础 checkpoint · BEHAVIOR 示范数据。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - 配方
     - 配置
     - 监督信号
   * - 仅动作
     - ``behavior_pi05_vla``
     - 主任务 prompt 与 32 步动作块。
   * - 单任务 VLM-to-VLA
     - ``behavior_pi05_vlm_vla``
     - 主任务 prompt、逐帧子任务 response 与动作。
   * - 50 任务 VLM-to-VLA
     - ``behavior_50tasks_pi05_vlm_vla``
     - 全部 50 个任务的逐 episode 子任务语言与动作。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 规格
   * - Observation
     - 头部 RGB、左右腕部 RGB 与 R1 Pro proprioception。
   * - Action
     - 32 步 BEHAVIOR 动作块，补齐到 32 个模型通道。
   * - Reward
     - SFT 阶段不使用。
   * - Prompt
     - 主任务；``vlm_vla`` 额外监督逐 episode 的局部子任务。

准备数据
--------

让两个通用数据集字段指向所选的 LeRobot 数据集：

.. code-block:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
     repo_id: behavior-1k/2025-challenge-demos
     modalities: [rgb]
     num_workers: 8
     hf_cache_dir: /path/to/large-cache/hf_datasets
     tasks: [turning_on_radio]

使用 ``vlm_vla`` 时，设置 ``fine_grained_level: 1``。加载器保留主任务作为
``prompt``，并从每个 episode 的技能标注解析 ``response``。加载器会在视频解码前跳过
有效标注区间之外的帧。50 任务缓存可能需要约 140 GB，因此请将 ``hf_cache_dir``
放在大容量文件系统中。

.. warning::

   不要在 50 任务训练中使用固定的逐任务子任务列表。技能序列与对象标识会随 episode
   变化。

配置模型
--------

下载模型
~~~~~~~~

在本地准备新格式的 π₀.₅ PyTorch 基础 checkpoint。目录中必须包含
``model.safetensors`` 和 ``config.json``。将 ``model_path`` 指向该目录；仓库中的
验证配方使用 ``/mnt/public/xzxuan/models/pi05_base_pytorch_new``。

无路径模型模板位于 ``examples/sft/config/model/pi0_5_pytorch.yaml``。在
``actor.model`` 下设置实验路径与完整 π₀.₅ 行为：

.. code-block:: yaml

   actor:
     model:
       model_path: /path/to/pi05_base_pytorch_new
       openpi:
         mode: vlm_vla
         state_token: abs_joint_old
         assets_dir: /path/to/norm-stats
         asset_id: behavior
         language_loss_weight: 1.0
         action_loss_weight: 10.0
         stop_gradient_to_vlm: false
         max_new_tokens: 24
         language_temperature: 0.0

使用 ``state_token: none`` 可生成无状态语言前缀。其他取值会把归一化状态离散化为语言
token 并注入 prompt。OpenPI 会自动将 PaliGemma tokenizer 下载到缓存；可设置
``OPENPI_DATA_HOME`` 来选择共享的可写缓存目录。当任务、状态与 response 超过
``max_token_len`` 时，tokenizer 会直接报错，不会截断 response 或 EOS 监督。
50 任务配置使用 288 个 token。

归一化统计从 ``{assets_dir}/{asset_id}/norm_stats.json`` 读取。SFT 与评估必须使用
相同状态/动作表示对应的统计量。

精度
~~~~

SFT 模板加载 fp32 optimizer-master 参数。FSDP 使用 bf16 计算、fp32 梯度归约，并启用
non-reentrant 梯度检查点。请保持加载 dtype 与计算 dtype 独立，避免较小的 warmup
更新因舍入而丢失。

安装
----

安装 OpenPI 模型与 BEHAVIOR 环境：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior

该命令会安装 RLinf OpenPI fork、BEHAVIOR runtime，以及共享
``build_openpi_transforms`` 流程所需的依赖。

运行
----

在仓库根目录启动任一 SFT 配方：

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh behavior_pi05_vla
   bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_vla
   bash examples/sft/run_vla_sft.sh behavior_50tasks_pi05_vlm_vla

第一条命令训练现有的仅动作模型；其余命令联合训练语言损失与动作损失。checkpoint
写入 ``runner.logger.log_path/checkpoints/global_step_<N>/``。

转换 Checkpoint
---------------

把 FSDP SFT checkpoint 转换为评估使用的裸新格式：

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/norm_stats.json \
       --output-model /path/to/pi05_sft_pytorch_new \
       --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json

转换器会移除 wrapper/FSDP 前缀，把模型张量转换为 bf16，并复制所选归一化统计。使用
:doc:`BEHAVIOR 评估 <../../evaluations/guides/behavior>` 中对应的独立评估流程。

可视化与结果
------------

让 TensorBoard 读取 ``runner.logger.log_path``。使用 ``vlm_vla`` 时，请同时查看
``action_loss``、``language_loss``、``language_acc`` 与总 ``loss``。共享日志约定参见
:doc:`训练指标 <../../reference/metrics>`。
