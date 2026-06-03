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

"""Validation guardrails for the OpenPI PyTorch eval-only model."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from rlinf.config import (
    _validate_openpi_pytorch_eval_cfg,
    validate_sft_cfg,
)


def _embodied_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "runner": {"only_eval": True},
            "actor": {
                "model": {
                    "model_type": "openpi_pytorch",
                    "precision": None,
                    "add_value_head": False,
                    "openpi": {"config_name": "pi05_behavior"},
                }
            },
            "rollout": {"model": {"precision": None}},
            "env": {
                "train": {"env_type": "behavior"},
                "eval": {"env_type": "behavior"},
            },
        }
    )
    for path, value in overrides.items():
        OmegaConf.update(cfg, path.replace("__", "."), value, merge=True)
    return cfg


def test_openpi_pytorch_validation_accepts_behavior_eval_only():
    _validate_openpi_pytorch_eval_cfg(_embodied_cfg(), task_type="embodied")


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("runner__only_eval", False, "eval-only"),
        ("env__eval__env_type", "libero", "BEHAVIOR"),
        ("actor__model__add_value_head", True, "add_value_head"),
        ("actor__model__openpi__full_pi05", True, "full_pi05"),
        ("actor__model__openpi__use_dsrl", True, "use_dsrl"),
        ("actor__model__precision", "fp32", "precision"),
    ],
)
def test_openpi_pytorch_validation_rejects_unsupported_embodied_paths(
    path, value, match
):
    cfg = _embodied_cfg(**{path: value})
    with pytest.raises(AssertionError, match=match):
        _validate_openpi_pytorch_eval_cfg(cfg, task_type="embodied")


def _sft_cfg(**overrides):
    cfg = OmegaConf.create(
        {
            "actor": {
                "global_batch_size": 256,
                "micro_batch_size": 32,
                "model": {
                    "model_type": "openpi_pytorch",
                    "precision": None,
                    "add_value_head": False,
                    "openpi": {
                        "config_name": "pi05_behavior_b1k_local",
                        "full_pi05": False,
                    },
                },
            },
            "data": {"train_data_paths": "/tmp/data"},
            "runner": {},
        }
    )
    for path, value in overrides.items():
        OmegaConf.update(cfg, path.replace("__", "."), value, merge=True)
    return cfg


def test_validate_sft_cfg_accepts_openpi_pytorch_behavior():
    # openpi_pytorch SFT on a BEHAVIOR data config is now supported.
    validate_sft_cfg(_sft_cfg())


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        ("actor__model__openpi__full_pi05", True, "full_pi05"),
        ("actor__model__openpi__use_dsrl", True, "use_dsrl"),
        ("actor__model__add_value_head", True, "add_value_head"),
        ("actor__model__openpi__config_name", "pi05_libero", "BEHAVIOR"),
        ("actor__model__precision", "fp32", "precision"),
    ],
)
def test_validate_sft_cfg_rejects_unsupported_openpi_pytorch(path, value, match):
    cfg = _sft_cfg(**{path: value})
    with pytest.raises(AssertionError, match=match):
        validate_sft_cfg(cfg)
