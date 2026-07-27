基于 PyTorch OpenPI (Pi0.5) 的 BEHAVIOR 监督微调
================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 45%

   用于 PyTorch OpenPI SFT 的 π₀ 模型系列。来源：`Physical
   Intelligence <https://www.physicalintelligence.company/blog/pi0>`_。

在 BEHAVIOR 示范数据上微调数值对齐的 PyTorch π₀.₅ 实现。你可以训练仅输出动作的
VLA，也可以训练完整的 VLM-to-VLA 模型：模型先预测子任务，再让动作专家基于该预测
生成动作。两条路径都支持绝对/增量关节控制以及绝对/增量末端执行器（EEF）控制。

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
     - 32 步、23 通道关节或 21 通道 EEF 动作块，补齐到 32 个模型通道。
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

转换控制表示
~~~~~~~~~~~~

仓库中的 SFT YAML 使用统一的控制入口。通过
``actor.model.openpi.control_mode`` 选择表示，不要添加按模式区分的数据集或
统计量字段。

.. list-table::
   :header-rows: 1
   :widths: 20 14 30 36

   * - ``control_mode``
     - 维度
     - 手臂目标
     - BEHAVIOR 控制器
   * - ``abs_joint``
     - 23
     - 绝对 7-DoF 关节目标
     - ``JointController`` （绝对）
   * - ``delta_joint``
     - 23
     - 实际到达的 ``qpos[t+1] - qpos[t]``
     - ``JointController`` （增量）
   * - ``abs_eef``
     - 21
     - 基座坐标系下的位置和轴角姿态
     - IK ``absolute_pose``
   * - ``delta_eef``
     - 21
     - 基座坐标系下的位置与相对旋转增量
     - IK ``pose_delta_ori``

原始数据集已经包含 ``abs_joint`` 动作。使用统一转换脚本生成任一表示，也可以只
生成用于验证的小子集：

.. code-block:: bash

   bash toolkits/behavior/convert_openpi_control_mode.sh \
       --source-dataset-root /path/to/native-behavior-data \
       --output-dataset-root /path/to/behavior-delta-eef \
       --control-mode delta_eef \
       --tasks turning_on_radio

省略 ``--tasks`` 会转换全部 50 个任务。``--episodes 10`` 按绝对 LeRobot
episode ID 限制范围，适合快速验证。转换器不会修改源数据；它会写入新的 Parquet
和 ``meta/control_mode.json``，并链接视频及与动作无关的标注。
旧版 ``RLinf`` 转换数据使用 ``meta/eef_delta_provenance.json`` 和旧模式别名，
必须从原生 ``abs_joint`` 数据重新生成。转换器会拒绝这些数据，避免把同为 23
通道的 ``delta_joint`` 源数据错误标记为 ``abs_joint``。

随后为相同表示计算归一化统计量：

.. code-block:: bash

   bash toolkits/behavior/compute_openpi_norm_stats.sh \
       --behavior-dataset-root /path/to/behavior-delta-eef \
       --assets-dir /path/to/behavior-norm-stats \
       --asset-id turning_on_radio_delta_eef \
       --control-mode delta_eef \
       --state-token abs_eef \
       --tasks turning_on_radio

四元数到轴角的映射是非线性的，因此 ``abs_eef`` 状态统计会扫描原始帧。关节状态
默认使用更快的 episode 元数据路径；需要精确逐帧聚合时可增加
``--raw-state-stats``。

仅通过原有通用字段选择生成的数据和统计量：

.. code-block:: yaml

   data:
     train_data_paths: /path/to/behavior-delta-eef
     behavior_dataset_root: ${data.train_data_paths}

   actor:
     model:
       openpi:
         control_mode: delta_eef
         state_token: abs_eef
         assets_dir: /path/to/behavior-norm-stats
         asset_id: turning_on_radio_delta_eef

模型模板会根据 ``control_mode`` 推导环境动作维度，同时保持模型填充维度为 32。
系统会在使用前校验数据集来源、统计量元数据、模型维度和评估控制器。

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
         control_mode: abs_joint
         assets_dir: /path/to/norm-stats
         asset_id: behavior
         language_loss_weight: 1.0
         action_loss_weight: 10.0
         stop_gradient_to_vlm: false
         max_new_tokens: 24
         language_temperature: 0.0

PaliGemma SentencePiece 模型会自动下载到 OpenPI 缓存。若默认的
``~/.cache/openpi`` 不合适，或多个 worker 需要共享同一个可写缓存，请在启动前设置
``OPENPI_DATA_HOME``。

``state_token`` 只接受以下四个值：

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - 值
     - 状态行为
   * - ``none``
     - 不向语言前缀注入离散状态；预处理仅为归一化和 checkpoint 兼容保留
       ``abs_joint_old`` tensor 布局，但模型不会使用其归一化数值。
   * - ``abs_joint_old``
     - 注入预训练关节布局，两个夹爪都位于末尾。
   * - ``abs_joint``
     - 注入与动作对齐的关节布局，左夹爪位于索引 14。
   * - ``abs_eef``
     - 注入基座坐标系下的 EEF 位置和轴角状态。

``abs_joint_old``、``abs_joint`` 和 ``abs_eef`` 需要匹配的状态统计量。使用
``none`` 时模型不消费状态 token，因此忽略状态布局元数据，但仍校验动作布局和
任务覆盖范围。当任务、状态与 response 超过 ``max_token_len`` 时，tokenizer
会直接报错，不会截断 response 或 EOS 监督。50 任务配置使用 288 个 token。

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
