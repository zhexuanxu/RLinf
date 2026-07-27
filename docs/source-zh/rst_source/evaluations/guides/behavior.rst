BEHAVIOR-1K 评测
================

BEHAVIOR-1K 是基于 OmniGibson 与 Isaac Sim 的大规模家居场景仿真基准，控制双臂 R1 Pro 机器人完成抓取、放置、整理等家庭操作任务。RLinf 支持在 BEHAVIOR 环境中并行评测 OpenPI 等 VLA 策略，并输出 ``eval/success_once`` 等指标。

相关训练文档：:doc:`../../examples/embodied/behavior` 和
:doc:`../../examples/embodied/sft_openpi_pytorch`

环境准备
--------

**安装依赖**

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior
   source .venv/bin/activate

``evaluations/behavior/`` 提供 OpenPI π₀.₅、自包含 PyTorch VLA 与
VLM-to-VLA 评测，以及确定性数据集 replay。训练侧还支持 OpenVLA-OFT，可从
``examples/embodiment/config/`` 中的训练配置派生评测 YAML（见
:doc:`../reference/configuration`）。

**硬件与 Isaac Sim**

BEHAVIOR 依赖 Isaac Sim 4.5，对 GPU 与驱动有额外要求，详见训练文档中的 `Isaac Sim 依赖说明 <https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html>`_。要点如下：

- 建议配备支持 Ray Tracing 的 GPU（如 RTX 30/40 系列）；A100、H100 等无 RT 能力的卡渲染质量较差，画面可能出现马赛克或模糊。
- Hopper 及以上架构 GPU 需使用 570 及以上版本 NVIDIA 驱动。

也可使用官方 Docker 镜像 ``rlinf/rlinf:agentic-rlinf0.3-behavior`` 运行评测，详见 :doc:`../../examples/embodied/behavior`。

**环境变量**

每次评测前须设置 ``ISAAC_PATH`` 与 OmniGibson 数据路径（``run_eval.sh`` 会自动补全 ``OMNIGIBSON_DATASET_PATH`` 等衍生变量，以及 ``EXP_PATH``、``CARB_APP_PATH``）：

.. code-block:: bash

   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

BEHAVIOR 资源体积超过 30 GB，下载与 license 配置步骤见 :doc:`../../examples/embodied/behavior` 中的「资源下载」一节。

示例配置
--------

``evaluations/behavior/`` 目录下已有以下示例：

.. list-table::
   :header-rows: 1
   :widths: 52 24 24

   * - 配置文件
     - 环境 preset
     - 模型
   * - ``behavior_openpi_pi05_eval.yaml``
     - ``behavior_r1pro``
     - π₀.₅
   * - ``behavior_openpi_pi05_pytorch_eval.yaml``
     - ``behavior_r1pro``
     - PyTorch π₀.₅ VLA
   * - ``behavior_openpi_pi05_pytorch_vlm_vla_eval.yaml``
     - ``behavior_r1pro``
     - PyTorch π₀.₅ VLM-to-VLA
   * - ``behavior_openpi_pi05_pytorch_vlm_vla_eval_test.yaml``
     - ``behavior_r1pro``
     - 在有序缓存 instance 测试集上评测 PyTorch π₀.₅ VLM-to-VLA
   * - ``behavior_openpi_pi05_pytorch_replay.yaml``
     - ``behavior_r1pro``
     - 数据集 replay（无模型）

若 ``evaluations/behavior/<config>.yaml`` 不存在，``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下同名配置（例如 ``behavior_ppo_openpi_pi05``）。回退配置包含 ``actor`` / ``algorithm`` 等训练段，并需要设置 ``runner.only_eval=true``。

完整评测流程
------------

**Step 1：激活环境并设置路径**

.. code-block:: bash

   source .venv/bin/activate
   export ISAAC_PATH=/path/to/isaac-sim
   export OMNIGIBSON_DATA_PATH=/path/to/BEHAVIOR-1K-datasets
   export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/
   export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson.key
   export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/

**Step 2：准备模型**

推荐预训练权重：`RLinf/RLinf-Pi0-Behavior <https://huggingface.co/RLinf/RLinf-Pi0-Behavior>`_（下载命令见训练文档）。若使用第三方 OpenPI 权重（如 OpenPI-Comet），需先转换为 PyTorch 格式后再填入 ``rollout.model.model_path``。自包含 PyTorch 配置要求使用 :doc:`../../examples/embodied/sft_openpi_pytorch` 中说明的新格式目录。

**Step 3：编辑配置**

复制或编辑目标 YAML，至少修改 ``rollout.model.model_path``。VLM-to-VLA 会自动将
PaliGemma SentencePiece 模型下载到 OpenPI 缓存；若默认的
``~/.cache/openpi`` 不合适，请设置 ``OPENPI_DATA_HOME``。通用
``env.eval`` 字段见 :doc:`../reference/configuration` 中的
:ref:`env-eval-fields`；BEHAVIOR 特有字段与评测协议见下文
:ref:`behavior-eval-config`。

OpenPI 参数必须与训练保持一致。对于自包含 PyTorch 路径，
``rollout.model.openpi.state_token`` 与 ``control_mode`` 必须匹配
checkpoint、数据集和归一化统计量。仓库中的评测示例使用
``control_mode: abs_joint``；不注入 prompt 状态的 VLM-to-VLA checkpoint
使用 ``state_token: none``。

**Step 4：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model

完整 PyTorch 模型的启动方式如下：

.. code-block:: bash

   bash evaluations/run_eval.sh behavior \
     behavior_openpi_pi05_pytorch_vlm_vla_eval \
     rollout.model.model_path=/path/to/new-format-model

**Step 5：查看结果**

终端输出 ``eval/success_once``；日志与视频见 :doc:`../reference/results`。

.. _behavior-eval-config:

评测配置详解
------------

BEHAVIOR 评测 **每次运行对应单个任务** （由 ``omni_config.task.activity_name`` 指定），不会在一次启动中自动遍历全部 50 个任务。以下字段共同决定并行规模、单条轨迹长度与初始场景实例。

评测协议概述
~~~~~~~~~~~~

BEHAVIOR-1K 共 50 个 household 任务（任务名列表见 ``rlinf/envs/behavior/behavior_task.jsonl``）。``behavior_r1pro`` preset 默认任务为 ``turning_on_radio``，场景为 ``house_double_floor_lower``。

每条评测轨迹由以下配置唯一确定：

- ``omni_config.task.activity_name``：任务名称，决定语言指令与 BDDL 定义；
- ``omni_config.task.activity_definition_id``：任务定义变体编号（通常为 ``0``）；
- ``omni_config.task.activity_instance_id`` 与 ``instance_resample_mode``：初始物体布局与机器人位姿。

``instance_resample_mode`` 支持三种模式：

- ``disabled`` （默认）：每次 reset 加载标量 ``activity_instance_id`` 对应的固定实例；若设置了 ``activity_instance_dir``，则从该目录读取对应 JSON。
- ``offline``：按确定性的全局轮询顺序分配缓存实例。列表形式的 ``activity_instance_id`` 会限制 instance 池并保留给定顺序；标量形式则允许使用目录中发现的全部文件。需先下载官方 ``2025-challenge-task-instances`` 或用 ``instance_generator.py`` 生成。
- ``online``：reset 时在线重采样物体布局（需 ``online_object_sampling: True`` 且 ``use_presampled_robot_pose: False``，启动较慢）。

.. note::

   单次启动不会自动遍历全部任务。要评测多个任务，需分别修改
   ``activity_name`` 并多次运行，或编写批量脚本。评测 instance 测试集时，请使用
   ``instance_resample_mode: offline`` 与 ID 列表。每次 reset 消耗下一个全局位置，
   因此 ``total_num_envs × rollout_epoch`` 决定覆盖次数与重复次数。

``env.eval`` 通用字段
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - 字段
     - BEHAVIOR 设置建议
   * - ``total_num_envs``
     - 全局并行环境数。单个 BEHAVIOR 环境约占 **10 GiB** 显存；模型评测配置使用 ``8``，replay 使用 ``1``。
   * - ``rollout_epoch``
     - 评测轮次，相同配置下重复运行并取指标平均；JAX、PyTorch 与 replay 配置分别使用 ``2``、``4`` 和 ``1``。
   * - ``max_episode_steps``
     - 单条轨迹步数上限。模型评测使用 ``4096``，replay 使用 ``2560``；preset 默认 ``2000`` 偏短，长 horizon 任务建议调大。
   * - ``max_steps_per_rollout_epoch``
     - 每轮 rollout 的总交互步数，**必须能被** ``rollout.model.num_action_chunks`` **整除**。无 ``auto_reset`` 时通常等于 ``max_episode_steps``。
   * - ``num_env_subprocess``
     - 单个 env worker 内划分的 Isaac 仿真子进程数（默认 ``1``）。适当增大可缓解步进瓶颈，但会成倍增加显存与进程开销；``total_num_envs`` 须能被 ``num_env_subprocess × pipeline_stage_num`` 整除。
   * - ``skip_intermediate_obs_in_chunk``
     - 设为 ``True`` 时跳过 chunk 内中间 observation，显著加速环境步进；保存的视频仅含 chunk 边界帧。

``omni_config`` 关键字段
~~~~~~~~~~~~~~~~~~~~~~~~

以下字段位于 ``env.eval.omni_config`` 下（继承自 ``examples/embodiment/config/env/behavior_r1pro.yaml``，可在评测 YAML 中覆盖）：

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_name: turning_on_radio
           activity_definition_id: 0
           activity_instance_id: 0
           activity_instance_dir: null          # 指向缓存 instance JSON 目录
           instance_file_format: tro_state        # template | tro_state
           instance_resample_mode: disabled       # disabled | offline | online
         scene:
           scene_model: house_double_floor_lower
           partial_scene_load: true               # 仅加载任务相关房间，缩短启动时间

更完整的字段说明（``partial_scene_load``、``instance_generator.py`` 用法等）见 :doc:`../../examples/embodied/behavior` 中的「behavior_r1pro.yaml 关键配置说明」。

GPU 与集群布局
~~~~~~~~~~~~~~

BEHAVIOR 环境步进较慢，通常建议给 env 分配足够 GPU，并与 rollout 共享或分离 placement：

.. code-block:: yaml

   cluster:
     component_placement:
       rollout,env: all          # env 与 rollout 共享全部 GPU（示例默认）

也可将 env 与 rollout 分到不同 GPU 以缓解显存压力，详见训练文档「关键集群配置」一节。

进阶用法
--------

**切换评测任务**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.omni_config.task.activity_name=picking_up_trash

**使用确定性离线 instance 测试集**

.. code-block:: yaml

   env:
     eval:
       omni_config:
         task:
           activity_instance_dir: ${oc.env:OMNIGIBSON_DATA_PATH}/2025-challenge-task-instances
           activity_instance_id: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
           instance_file_format: tro_state
           instance_resample_mode: offline
       rollout_epoch: 5

**调整并行规模**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_openpi_pi05_eval \
     rollout.model.model_path=/path/to/model \
     env.eval.total_num_envs=4 \
     env.eval.num_env_subprocess=2

**从训练配置评测**

.. code-block:: bash

   bash evaluations/run_eval.sh behavior behavior_ppo_openpi_pi05 \
     rollout.model.model_path=/path/to/model \
     runner.only_eval=true

Replay 已记录 Episode
---------------------

Replay 会把数据集中记录的动作直接发送给 BEHAVIOR 控制器，用于在不依赖模型推理的
情况下验证数据转换和控制器语义。Replay worker 不加载 checkpoint、tokenizer、
transform、归一化统计量或 demonstration 中记录的 observation state。

显式指定一组缓存 instance 与数据集 episode：

.. code-block:: bash

   bash toolkits/behavior/run_behavior_replay.sh \
     abs_joint none \
     /path/to/behavior-dataset \
     1 10

位置参数依次为 ``control_mode``、``state_token``、``dataset_root``、
``activity_instance_id`` 和 ``episode_index``。使用 ``delta_joint``、
``abs_eef`` 或 ``delta_eef`` 时，将 ``dataset_root`` 指向
``convert_openpi_control_mode.sh`` 生成的数据。启动脚本使用专用入口
``evaluations/behavior/replay_embodied_agent.py`` 和统一的
``behavior_openpi_pi05_pytorch_replay.yaml``，并接受后续 Hydra 覆盖参数：

.. code-block:: bash

   bash toolkits/behavior/run_behavior_replay.sh \
     delta_eef abs_eef /path/to/converted-data 1 10 \
     env.eval.max_episode_steps=3200 \
     env.eval.max_steps_per_rollout_epoch=3200

Replay 要求 ``instance_resample_mode: disabled``，并要求显式
``activity_instance_id`` 与所选缓存 JSON 一致；它不会根据 instance ID 推导
episode；两者由调用者显式配对，系统仅检查 episode 属于配置的 activity。
``max_steps_per_rollout_epoch`` 必须能被 32 步动作 horizon 整除。
启动脚本会同时覆盖 ``env.eval.replay`` 和 ``env.eval.omni_config.task`` 中的
instance ID；若直接启动 YAML，请将两个字段覆盖为同一个值。

常见问题
--------

- **数据下载：** BEHAVIOR 资源体积较大，请按 :doc:`../../examples/embodied/behavior` 完成 Isaac Sim、OmniGibson 资产与 license 配置后再评测。
- **ISAAC_PATH 未设置：** ``run_eval.sh`` 默认值为 ``/path/to/isaac-sim``，未正确设置会导致 Isaac Sim 无法启动。
- **Headless 模式：** ``run_eval.sh`` 默认设置 ``OMNIGIBSON_HEADLESS=1``。
- **显存不足：** 降低 ``total_num_envs`` 或 ``num_env_subprocess``；单个环境约占 10 GiB 显存。
- **渲染马赛克/模糊：** 当前 GPU 无 Ray Tracing 能力，建议换用 RTX 30/40 系列或更高。
- **启动极慢：** 首次加载大场景耗时较长；保持 ``partial_scene_load: true`` 可只加载任务相关房间。
- **视频帧数少于预期：** ``skip_intermediate_obs_in_chunk: True`` 会跳过 chunk 内中间帧，仅保留策略实际消费的 observation 对应帧。
- **instance 加载失败：** ``activity_instance_dir`` 中的 JSON 文件名须与 ``activity_name``、``activity_definition_id``、``scene_model`` 匹配；详见 ``rlinf/envs/behavior/instance_loader.py``。
- **步数校验失败：** ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除。
- **Replay 模式不匹配：** 转换后的数据集带有 ``meta/control_mode.json``；其中的模式和动作维度必须与 ``rollout.model.openpi.control_mode`` 一致。
