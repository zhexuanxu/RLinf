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

"""Self-containment and smoke tests for the vendored PyTorch OpenPI 0.5 core.

These tests guard the invariant that ``openpi_pytorch`` runs the eval /
action-generation path without the externally installed ``openpi`` package:

* import isolation: an AST scan finds zero external ``openpi`` imports;
* importability: every vendored core module imports;
* smoke: a tiny (``dummy`` variant) Pi0 produces a finite action tensor of the
  expected shape from ``sample_actions`` on a synthetic observation.

The smoke uses the ``dummy`` gemma variant on CPU so it stays fast and does not
require a GPU or the multi-GB BEHAVIOR checkpoint.
"""

from __future__ import annotations

import importlib

import pytest

from rlinf.models.embodiment.openpi_pytorch._import_isolation import (
    find_external_openpi_imports,
)

_CORE_MODULES = [
    "rlinf.models.embodiment.openpi_pytorch.utils.utils",
    "rlinf.models.embodiment.openpi_pytorch.utils.lora",
    "rlinf.models.embodiment.openpi_pytorch.utils.pointnet",
    "rlinf.models.embodiment.openpi_pytorch.utils.model",
    "rlinf.models.embodiment.openpi_pytorch.utils.siglip",
    "rlinf.models.embodiment.openpi_pytorch.utils.gemma",
    "rlinf.models.embodiment.openpi_pytorch.utils.pi0_config",
    "rlinf.models.embodiment.openpi_pytorch.utils.pi0",
    "rlinf.models.embodiment.openpi_pytorch.utils.checkpoint_format",
]


def test_no_external_openpi_imports():
    """The package must not import the externally installed ``openpi`` package."""
    violations = find_external_openpi_imports()
    assert violations == [], "External `openpi` imports found:\n" + "\n".join(
        f"  {v.file}:{v.lineno}: {v.statement}" for v in violations
    )


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_core_modules_import(module_name):
    """Every vendored core module imports cleanly."""
    importlib.import_module(module_name)


def test_sample_actions_smoke():
    """A tiny Pi0 returns a finite action tensor of the expected shape."""
    import torch

    from rlinf.models.embodiment.openpi_pytorch.utils.pi0_config import Pi0Config

    batch_size = 1
    action_dim = 32
    action_horizon = 4
    cfg = Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=action_dim,
        action_horizon=action_horizon,
        pi05=True,
        pcd=False,
    )
    model = cfg.create().eval()
    obs = cfg.fake_obs(batch_size)

    with torch.no_grad():
        actions = model.sample_actions(obs, num_steps=2)

    assert tuple(actions.shape) == (batch_size, action_horizon, action_dim)
    assert torch.isfinite(actions).all(), "sample_actions produced non-finite values"
