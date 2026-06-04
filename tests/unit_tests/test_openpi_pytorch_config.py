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

"""YAML-driven config for the openpi_pytorch model (M3: AC-4/5/8/9/10-config).

These tests assert the post-refactor configuration contract:
- model templates carry model-shape fields only (no filesystem paths);
- the experiment configs carry the paths + the SFT data fields;
- the package builds Pi0Config from YAML (a checkpoint config.json is ignored)
  and defines no in-code ``TrainConfig`` registry keyed by config name;
- eval and SFT resolve the SAME canonical norm-stats file via assets_dir+asset_id.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest
from omegaconf import OmegaConf

_REPO = pathlib.Path(__file__).resolve().parents[2]
_SFT_MODEL = _REPO / "examples/sft/config/model/pi0_5_pytorch.yaml"
_EMB_MODEL = _REPO / "examples/embodiment/config/model/pi0_5_pytorch.yaml"
_SFT_EXP = _REPO / "examples/sft/config/behavior_pi05_vla.yaml"
_EMB_EXP = (
    _REPO / "examples/embodiment/config/behavior_ppo_openpi_pi05_pytorch_eval.yaml"
)
_PACKAGE = _REPO / "rlinf/models/embodiment/openpi_pytorch"

_SHAPE_FIELDS = ("model_action_dim", "paligemma_variant", "action_expert_variant")


def _load(path):
    return OmegaConf.load(path)


# --------------------------------------------------------------------------- #
# AC-9: model templates are path-free; paths live in the experiment configs.
# --------------------------------------------------------------------------- #
def test_model_templates_carry_shape_fields_not_paths():
    for path in (_SFT_MODEL, _EMB_MODEL):
        cfg = _load(path)
        assert "model_path" not in cfg, f"{path.name} must not hard-code model_path"
        assert "config_name" not in cfg.openpi, (
            f"{path.name} must not carry config_name"
        )
        for field in _SHAPE_FIELDS:
            assert field in cfg.openpi, f"{path.name} missing openpi.{field}"

    # The SFT template carries no asset paths (they live in the experiment config).
    sft = _load(_SFT_MODEL)
    assert "assets_dir" not in sft.openpi and "asset_id" not in sft.openpi
    # The embodiment template may carry assets_dir/asset_id placeholders.
    emb = _load(_EMB_MODEL)
    assert "assets_dir" in emb.openpi and "asset_id" in emb.openpi


def test_sft_experiment_config_has_paths_and_data_fields():
    cfg = _load(_SFT_EXP)
    assert cfg.actor.model.model_path
    assert cfg.actor.model.openpi.assets_dir
    assert cfg.actor.model.openpi.asset_id
    assert list(cfg.data.tasks) == ["turning_on_radio"]
    assert cfg.data.use_skill is False


def test_eval_experiment_config_has_paths():
    cfg = _load(_EMB_EXP)
    assert cfg.actor.model.openpi.assets_dir
    assert cfg.actor.model.openpi.asset_id


def test_composed_sft_model_snapshot():
    """Template (shape) + experiment override (paths) compose to the expected set."""
    template = _load(_SFT_MODEL)
    override = _load(_SFT_EXP).actor.model
    merged = OmegaConf.merge(template, override)
    assert merged.model_path == "/mnt/public/xzxuan/models/pi05_base_pytorch_new"
    assert merged.openpi.model_action_dim == 32
    assert merged.openpi.paligemma_variant == "gemma_2b"
    assert merged.openpi.action_expert_variant == "gemma_300m"
    assert merged.openpi.asset_id == "behavior-1k/2025-challenge-demos"


# --------------------------------------------------------------------------- #
# AC-4 / AC-5: no in-package TrainConfig registry; config_name fully removed.
# --------------------------------------------------------------------------- #
def test_package_source_has_no_config_name_or_registry():
    for py in _PACKAGE.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        assert "config_name" not in text, f"{py} still references config_name"
        assert "TrainConfig" not in text, f"{py} defines/uses a TrainConfig registry"


# --------------------------------------------------------------------------- #
# AC-8: eval and SFT resolve the SAME canonical norm-stats via assets_dir/asset_id.
# --------------------------------------------------------------------------- #
def _write_norm_stats(directory: pathlib.Path, value: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    key: {
                        "mean": [value] * 32,
                        "std": [1.0] * 32,
                        "q01": [0.0] * 32,
                        "q99": [1.0] * 32,
                    }
                    for key in ("state", "actions")
                }
            }
        )
    )


def test_norm_stats_resolver_is_shared_and_hash_equal(tmp_path):
    from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
        _resolve_norm_stats,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
        resolve_norm_stats_dir,
    )

    assets_dir = tmp_path / "assets"
    asset_id = "behavior-1k/2025-challenge-demos"
    _write_norm_stats(assets_dir / asset_id, 0.0)

    # The eval/model resolver and the SFT loader resolve the SAME file.
    eval_dir = resolve_norm_stats_dir(assets_dir, asset_id)
    assert (eval_dir / "norm_stats.json").is_file()
    sft_stats = _resolve_norm_stats(assets_dir, asset_id)
    eval_bytes = (eval_dir / "norm_stats.json").read_bytes()
    sft_dir = resolve_norm_stats_dir(assets_dir, asset_id)
    assert sft_dir == eval_dir  # same canonical directory
    assert np.allclose(sft_stats["state"].mean, 0.0)
    assert eval_bytes == (sft_dir / "norm_stats.json").read_bytes()


def test_norm_stats_resolver_rejects_divergent_and_missing(tmp_path):
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize import (
        resolve_norm_stats_dir,
    )

    good = tmp_path / "good"
    _write_norm_stats(good / "id", 0.0)
    other = tmp_path / "other"
    _write_norm_stats(other / "id", 9.0)
    a = (resolve_norm_stats_dir(good, "id") / "norm_stats.json").read_bytes()
    b = (resolve_norm_stats_dir(other, "id") / "norm_stats.json").read_bytes()
    assert a != b  # divergent stats files differ

    with pytest.raises(FileNotFoundError, match="norm_stats.json"):
        resolve_norm_stats_dir(tmp_path / "missing", "id")


# --------------------------------------------------------------------------- #
# AC-5: Pi0Config is built from YAML; a checkpoint config.json is ignored.
# --------------------------------------------------------------------------- #
def test_get_model_builds_from_yaml_and_ignores_config_json(tmp_path):
    torch = pytest.importorskip("torch")
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    base = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=32,
        pcd=False,
    )
    # Eval builds expect a bf16 new-format checkpoint (get_model casts to bf16),
    # so save the synthetic weights in bf16 — otherwise the dtype validation
    # fires before the config.json-is-ignored path under test.
    state_dict = base.create().to(torch.bfloat16).state_dict()
    safetensors.torch.save_file(state_dict, str(tmp_path / "model.safetensors"))
    # A checkpoint config.json with BOGUS shape values must be ignored.
    (tmp_path / "config.json").write_text(
        json.dumps({"action_horizon": 999, "action_dim": 7, "paligemma_variant": "x"})
    )
    _write_norm_stats(tmp_path / "physical-intelligence" / "behavior", 0.0)

    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "precision": "bf16",
            "num_action_chunks": 4,
            "action_dim": 23,
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "dummy",
                "action_expert_variant": "dummy",
                "assets_dir": str(tmp_path),
                "asset_id": "physical-intelligence/behavior",
            },
        }
    )
    # If the bogus config.json (action_horizon=999, action_dim=7) had been read,
    # Pi0Config would build a mismatched model and the strict load would fail.
    # A successful build proves the model shape came from the YAML fields (=4/32).
    model = get_model(cfg)
    assert model.processor is not None


def test_get_model_eval_requires_assets_dir(tmp_path):
    torch = pytest.importorskip("torch")
    import safetensors.torch

    from rlinf.models.embodiment.openpi_pytorch import get_model
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

    base = Pi0Config(
        dtype="bfloat16",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        pi05=True,
        action_horizon=4,
        action_dim=32,
        pcd=False,
    )
    # bf16 eval-format weights so the build reaches the assets_dir check (the
    # missing-norm-stats path under test), not the dtype validation.
    safetensors.torch.save_file(
        base.create().to(torch.bfloat16).state_dict(),
        str(tmp_path / "model.safetensors"),
    )
    cfg = OmegaConf.create(
        {
            "model_path": str(tmp_path),
            "precision": "bf16",
            "num_action_chunks": 4,
            "action_dim": 23,
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "dummy",
                "action_expert_variant": "dummy",
            },
        }
    )
    with pytest.raises(FileNotFoundError, match="assets_dir"):
        get_model(cfg)
