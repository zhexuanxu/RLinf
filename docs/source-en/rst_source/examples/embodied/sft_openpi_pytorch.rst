Supervised Fine-Tuning with PyTorch OpenPI (Pi0.5) on BEHAVIOR
==============================================================

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/pi0_icon.jpg
   :align: center
   :width: 45%

   The π₀ model family used for PyTorch OpenPI SFT. Source: `Physical
   Intelligence <https://www.physicalintelligence.company/blog/pi0>`_.

Fine-tune the numerically aligned PyTorch π₀.₅ implementation on BEHAVIOR
demonstrations. You can train the action-only VLA or the full VLM-to-VLA model,
which first predicts a subtask and then conditions the action expert on that
prediction. Both paths support absolute or delta joint control and absolute or
delta end-effector (EEF) control.

Overview
--------

Choose a single-task or 50-task recipe and train from the same π₀.₅ base model.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      π₀.₅ VLA · π₀.₅ VLM-to-VLA

   .. grid-item-card:: Methods
      :text-align: center

      Flow matching · language CE

   .. grid-item-card:: Data
      :text-align: center

      BEHAVIOR task 0 · all 50 tasks

   .. grid-item-card:: Hardware
      :text-align: center

      FSDP · bf16 compute

| **You'll do:** prepare data and stats → select a config → launch SFT → convert the checkpoint for evaluation.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · a new-format π₀.₅ base checkpoint · BEHAVIOR demonstrations.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Recipe
     - Config
     - Supervision
   * - Action-only
     - ``behavior_pi05_vla``
     - Main-task prompt and a 32-step action chunk.
   * - Single-task VLM-to-VLA
     - ``behavior_pi05_vlm_vla``
     - Main-task prompt, frame-local subtask response, and actions.
   * - 50-task VLM-to-VLA
     - ``behavior_50tasks_pi05_vlm_vla``
     - Episode-specific subtask language across all 50 tasks and actions.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Specification
   * - Observation
     - Head RGB, left/right wrist RGB, and R1 Pro proprioception.
   * - Action
     - A 32-step, 23-channel joint or 21-channel EEF action chunk, padded to 32
       model channels.
   * - Reward
     - Not used during SFT.
   * - Prompt
     - Main task; ``vlm_vla`` additionally supervises an episode-local subtask.

Prepare the Data
----------------

Point both generic dataset fields at the selected LeRobot dataset:

.. code-block:: yaml

   data:
     train_data_paths: /path/to/2025-challenge-demos
     behavior_dataset_root: /path/to/2025-challenge-demos
     repo_id: behavior-1k/2025-challenge-demos
     modalities: [rgb]
     num_workers: 8
     hf_cache_dir: /path/to/large-cache/hf_datasets
     tasks: [turning_on_radio]

For ``vlm_vla``, set ``fine_grained_level: 1``. The loader keeps the main task
as ``prompt`` and resolves ``response`` from each episode's skill annotation.
Frames outside the valid annotation interval are skipped before video decoding.
The 50-task cache can require about 140 GB, so place ``hf_cache_dir`` on a large
filesystem.

.. warning::

   Do not use a fixed per-task subtask list for 50-task training. Skill
   sequences and object identifiers vary by episode.

Convert Control Representations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The checked-in SFT YAMLs use one control surface. Select the representation with
``actor.model.openpi.control_mode``; do not add mode-specific dataset or asset
fields.

.. list-table::
   :header-rows: 1
   :widths: 20 14 30 36

   * - ``control_mode``
     - Width
     - Arm target
     - BEHAVIOR controller
   * - ``abs_joint``
     - 23
     - Absolute 7-DoF joint targets
     - ``JointController`` (absolute)
   * - ``delta_joint``
     - 23
     - Achieved ``qpos[t+1] - qpos[t]``
     - ``JointController`` (delta)
   * - ``abs_eef``
     - 21
     - Base-frame position and axis-angle pose
     - IK ``absolute_pose``
   * - ``delta_eef``
     - 21
     - Base-frame position and relative-rotation delta
     - IK ``pose_delta_ori``

The native dataset already contains ``abs_joint`` actions. Create any selected
representation, or a small validation subset, with the generic converter:

.. code-block:: bash

   bash toolkits/behavior/convert_openpi_control_mode.sh \
       --source-dataset-root /path/to/native-behavior-data \
       --output-dataset-root /path/to/behavior-delta-eef \
       --control-mode delta_eef \
       --tasks turning_on_radio

Omit ``--tasks`` to convert all 50 tasks. ``--episodes 10`` restricts the
conversion to absolute LeRobot episode IDs and is useful for validation. The
converter writes new Parquet files and ``meta/control_mode.json`` without
modifying the source; videos and action-independent annotations are linked.
Older ``RLinf`` conversions that carry ``meta/eef_delta_provenance.json`` use
legacy mode aliases and must be regenerated from the native ``abs_joint``
dataset. The converter rejects those artifacts so a 23-channel
``delta_joint`` source cannot be silently relabeled as ``abs_joint``.

Compute matching normalization statistics in the same representation:

.. code-block:: bash

   bash toolkits/behavior/compute_openpi_norm_stats.sh \
       --behavior-dataset-root /path/to/behavior-delta-eef \
       --assets-dir /path/to/behavior-norm-stats \
       --asset-id turning_on_radio_delta_eef \
       --control-mode delta_eef \
       --state-token abs_eef \
       --tasks turning_on_radio

``abs_eef`` state statistics are computed from raw frames because quaternion to
axis-angle conversion is nonlinear. Joint-state statistics use the faster
episode-metadata path by default; pass ``--raw-state-stats`` when exact
frame-level aggregation is required.

Use only the original generic path fields to select the generated data and
asset:

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

The model template derives the semantic action width from ``control_mode`` and
keeps the padded model width at 32. Dataset provenance, norm-stat metadata,
model width, and evaluation controllers are validated before use.

Configure the Model
-------------------

Download the Model
~~~~~~~~~~~~~~~~~~

Stage a new-format π₀.₅ PyTorch base checkpoint locally. The directory must
contain ``model.safetensors`` and ``config.json``. Set ``model_path`` to that
directory; the validation recipes in this repository use
``/mnt/public/xzxuan/models/pi05_base_pytorch_new``.

The path-free model template is
``examples/sft/config/model/pi0_5_pytorch.yaml``. Set experiment paths and the
full π₀.₅ behavior under ``actor.model``:

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

The PaliGemma SentencePiece model is downloaded automatically into OpenPI's
cache. Set ``OPENPI_DATA_HOME`` before launching when the default
``~/.cache/openpi`` location is unsuitable or when several workers should share
one writable cache.

``state_token`` accepts exactly four values:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Value
     - State behavior
   * - ``none``
     - Do not inject discrete state into the language prefix. Preprocessing
       retains the ``abs_joint_old`` tensor layout for checkpoint compatibility,
       but the model does not consume its normalized values.
   * - ``abs_joint_old``
     - Inject the pretrained joint layout with both grippers at the tail.
   * - ``abs_joint``
     - Inject the action-aligned joint layout with the left gripper at index 14.
   * - ``abs_eef``
     - Inject base-frame EEF position and axis-angle state.

``abs_joint_old``, ``abs_joint``, and ``abs_eef`` require matching state
statistics. With ``none``, state-layout metadata is ignored because no state
token is consumed, while action layout and task coverage are still validated.
The tokenizer raises when the task, state, and response exceed
``max_token_len``; it never truncates away response or EOS supervision. The
50-task config uses 288 tokens.

Norm stats resolve from
``{assets_dir}/{asset_id}/norm_stats.json``. SFT and evaluation must use stats
from the same state/action representation.

Precision
~~~~~~~~~

The SFT template loads fp32 optimizer-master parameters. FSDP computes in bf16,
reduces gradients in fp32, and enables non-reentrant gradient checkpointing.
Keep these load and compute dtypes independent so small warmup updates are not
rounded away.

Installation
------------

Install the OpenPI model and BEHAVIOR environment:

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env behavior

What this does: it installs the RLinf OpenPI fork, the BEHAVIOR runtime, and the
dependencies used by the shared ``build_openpi_transforms`` pipeline.

Run It
------

Launch one of the SFT recipes from the repository root:

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh behavior_pi05_vla
   bash examples/sft/run_vla_sft.sh behavior_pi05_vlm_vla
   bash examples/sft/run_vla_sft.sh behavior_50tasks_pi05_vlm_vla

The first command trains the existing action-only model. The other commands
train language and action losses together. Checkpoints are written beneath
``runner.logger.log_path/checkpoints/global_step_<N>/``.

Convert a Checkpoint
--------------------

Convert an FSDP SFT checkpoint into the bare new-format layout used by
evaluation:

.. code-block:: bash

   python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
       --ckpt /path/to/checkpoints/global_step_30000 \
       --input-norm-stats /path/to/norm_stats.json \
       --output-model /path/to/pi05_sft_pytorch_new \
       --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json

The converter removes wrapper/FSDP prefixes, casts model tensors to bf16, and
copies the selected norm stats. Use the matching standalone workflow in
:doc:`BEHAVIOR evaluation <../../evaluations/guides/behavior>`.

Visualization and Results
-------------------------

Launch TensorBoard against ``runner.logger.log_path``. For ``vlm_vla``, inspect
``action_loss``, ``language_loss``, and ``language_acc`` together with the total
``loss``. See :doc:`Training metrics <../../reference/metrics>` for the shared
logging contract.
