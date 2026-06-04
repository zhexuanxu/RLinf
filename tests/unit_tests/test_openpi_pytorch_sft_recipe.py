# Copyright (c) 2025, RLinf contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`use_skill: false` SFT recipe alignment vs the reference (AC-11, CPU half).

These tests pin every CPU-verifiable training knob of the BEHAVIOR SFT path to the
reference run ``pi05_b1k-task0000_sft_pytorch_mixed`` (in
``openpi-comet-pytorch-mixed``). The expected values below are the REFERENCE values,
each annotated with the reference file:line it comes from, so a drift on either side
(RLinf config OR a future reference change) turns the snapshot red. The full
knob-by-knob audit lives in ``docs/sft-recipe-audit.md``; the LR-schedule shape is
covered exactly by ``test_openpi_pytorch_sft_schedule.py``; the loss-curve match
itself is the GPU-evidenced tier (DEC-1/DEC-5) and is out of scope here.
"""

from __future__ import annotations

import pathlib

import pytest
from omegaconf import OmegaConf

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SFT_MODEL = _REPO / "examples/sft/config/model/pi0_5_pytorch.yaml"
_SFT_EXP = _REPO / "examples/sft/config/behavior_pi05_vla.yaml"

# Reference recipe values (openpi-comet-pytorch-mixed). Each comment cites the
# reference source so this test is a true cross-reference, not a self-snapshot.
_REF_PER_DEVICE_BATCH = 32  # config.py:875  batch_size=8*32 over 8 ranks
_REF_GLOBAL_BATCH = 256  # config.py:875  8*32
_REF_SEED = 42  # config.py:540  default seed
_REF_NUM_TRAIN_STEPS = 30000  # config.py:866
_REF_PEAK_LR = 2.5e-5  # config.py:868 / optimizer.py:20
_REF_WARMUP_STEPS = 1000  # optimizer.py:19
_REF_MIN_LR = 0.0  # optimizer.py:23  decay_lr=0.0
_REF_BETA1 = 0.9  # optimizer.py:70
_REF_BETA2 = 0.95  # optimizer.py:71
_REF_EPS = 1e-8  # optimizer.py:72
_REF_WEIGHT_DECAY = 1e-10  # optimizer.py:74
_REF_CLIP_GRAD = 1.0  # optimizer.py:75
_REF_ACTION_HORIZON = 32  # config.py:855  Pi0Config(action_horizon=32)
_REF_MODEL_ACTION_DIM = 32  # pi0_config.py:27  default action_dim
_REF_PALIGEMMA_VARIANT = "gemma_2b"  # pi0_config.py:22
_REF_ACTION_EXPERT_VARIANT = "gemma_300m"  # pi0_config.py:23
_REF_MAX_TOKEN_LEN = 200  # pi0_config.py:42  pi05 __post_init__
_REF_TASKS = ["turning_on_radio"]  # config.py:862
_REF_NUM_EPISODES = 200  # config.py:860  episodes_index=list(range(200))
_REF_WEIGHT_PATH = "/mnt/public/xzxuan/models/pi05_base_pytorch_new"  # config.py:881


def _load(path):
    return OmegaConf.load(path)


def test_sft_optimizer_matches_reference():
    """AdamW + warmup-cosine knobs equal the reference optimizer config."""
    optim = _load(_SFT_EXP).actor.optim
    assert optim.adam_beta1 == pytest.approx(_REF_BETA1)
    assert optim.adam_beta2 == pytest.approx(_REF_BETA2)
    assert optim.adam_eps == pytest.approx(_REF_EPS)
    assert optim.weight_decay == pytest.approx(_REF_WEIGHT_DECAY)
    assert optim.clip_grad == pytest.approx(_REF_CLIP_GRAD)
    assert optim.lr == pytest.approx(_REF_PEAK_LR)
    assert optim.lr_warmup_steps == _REF_WARMUP_STEPS
    assert optim.total_training_steps == _REF_NUM_TRAIN_STEPS
    assert optim.min_lr == pytest.approx(_REF_MIN_LR)
    # The reference-exact warmup init (peak/(warmup+1), not 0) requires the
    # openpi_cosine mode; the plain HF cosine would start at 0 (see sft_schedule).
    assert optim.lr_scheduler == "openpi_cosine"


def test_sft_batch_seed_steps_match_reference():
    """Per-device / global batch, seed, and training length equal the reference."""
    cfg = _load(_SFT_EXP)
    assert cfg.actor.micro_batch_size == _REF_PER_DEVICE_BATCH
    assert cfg.actor.global_batch_size == _REF_GLOBAL_BATCH
    # The reference reaches global batch 256 in a SINGLE step (no accumulation) only
    # on 8 ranks: 256 / (32 * 8) = 1. Pin the equivalence so the constants stay
    # consistent (the run itself must use 8 GPUs -- see docs/sft-recipe-audit.md).
    assert _REF_GLOBAL_BATCH == _REF_PER_DEVICE_BATCH * 8
    assert cfg.actor.seed == _REF_SEED
    assert cfg.runner.max_steps == _REF_NUM_TRAIN_STEPS


def test_sft_data_and_prompt_source_match_reference():
    """Single task-0000 task, main-task prompt (use_skill false)."""
    cfg = _load(_SFT_EXP)
    assert list(cfg.data.tasks) == _REF_TASKS
    # use_skill:false -> the reference prompt_from_task=True (main-task text).
    assert cfg.data.use_skill is False


def test_sft_model_shape_and_weights_match_reference():
    """Model shape (template) + the fp32 new-format base weights (experiment)."""
    merged = OmegaConf.merge(_load(_SFT_MODEL), _load(_SFT_EXP).actor.model)
    assert merged.model_path == _REF_WEIGHT_PATH
    assert merged.num_action_chunks == _REF_ACTION_HORIZON
    assert merged.openpi.model_action_dim == _REF_MODEL_ACTION_DIM
    assert merged.openpi.paligemma_variant == _REF_PALIGEMMA_VARIANT
    assert merged.openpi.action_expert_variant == _REF_ACTION_EXPERT_VARIANT
    assert merged.openpi.max_token_len == _REF_MAX_TOKEN_LEN
    # bf16 compute (fp32 load + cast) under FSDP, matching mp_bfloat16.
    assert merged.precision == "bf16"
    assert merged.load_for_training is True


def test_sft_fsdp_full_shard_matches_reference():
    """FSDP FULL_SHARD with bf16 mixed precision, matching fsdp1 mp_bfloat16."""
    fsdp = _load(_SFT_EXP).actor.fsdp_config
    assert fsdp.sharding_strategy == "full_shard"
    # The mixed-precision dtypes interpolate the model precision (bf16).
    assert "mixed_precision" in fsdp


# --------------------------------------------------------------------------- #
# Episode-set alignment on the REAL dataset: the reference episodes_index=
# list(range(200)) positionally selects the first 200 episodes of the task, and
# turning_on_radio (task id 0) has exactly 200 episodes -- so the RLinf task
# filter loads the identical set. This skip-gated gate fails if a future data or
# filter change breaks that equivalence.
# --------------------------------------------------------------------------- #
_DATA_ROOT = pathlib.Path("/mnt/public/xzxuan/data/2025-challenge-demos")
_ASSETS_DIR = pathlib.Path("/mnt/public/xzxuan/models/pi05-b1kpt50-cs32/assets")
_NORM_STATS = _ASSETS_DIR / "behavior-1k/2025-challenge-demos/norm_stats.json"


@pytest.mark.skipif(
    not (_DATA_ROOT.is_dir() and _NORM_STATS.is_file()),
    reason="real BEHAVIOR dataset / canonical task-0000 norm stats not available",
)
def test_turning_on_radio_loads_reference_episode_count():
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        build_behavior_sft_dataloader,
    )

    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {
                    "model_type": "openpi_pytorch",
                    "num_action_chunks": 32,
                    "openpi": {
                        "assets_dir": str(_ASSETS_DIR),
                        "asset_id": "behavior-1k/2025-challenge-demos",
                        "model_action_dim": 32,
                        "max_token_len": 200,
                    },
                },
                "micro_batch_size": 1,
                "eval_batch_size": 1,
                "seed": 42,
            },
            "data": {
                "train_data_paths": str(_DATA_ROOT),
                "num_workers": 0,
                "tasks": ["turning_on_radio"],
                "use_skill": False,
            },
        }
    )
    loader, _ = build_behavior_sft_dataloader(cfg, 1, 0, str(_DATA_ROOT))

    # The underlying streaming dataset loads exactly the reference's range(200).
    episodes = loader.torch_loader.dataset._dataset.episodes
    assert len(episodes) == _REF_NUM_EPISODES
    # All selected episodes belong to task id 0 (turning_on_radio).
    assert all(ep // 10000 == 0 for ep in episodes)
