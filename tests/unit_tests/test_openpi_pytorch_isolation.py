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

* import isolation: an AST scan of the package AND the relocated BEHAVIOR data
  modules (``rlinf/data/datasets/behavior/``) finds zero external ``openpi``
  imports;
* importability: every vendored core module imports;
* smoke: a tiny (``dummy`` variant) Pi0 produces a finite action tensor of the
  expected shape from ``sample_actions`` on a synthetic observation.

The AST-based isolation logic lives here (previously a standalone
``_import_isolation.py`` module) so the package itself ships no test-only code;
the guarantee is enforced by this test. It uses ``ast`` (not text matching) so
the string ``"openpi"`` in a comment, docstring, or logger name is never
mistaken for a dependency.

The smoke uses the ``dummy`` gemma variant on CPU so it stays fast and does not
require a GPU or the multi-GB BEHAVIOR checkpoint.
"""

from __future__ import annotations

import ast
import importlib
from dataclasses import dataclass
from pathlib import Path

import pytest

import rlinf.models.embodiment.openpi_pytorch as _openpi_pytorch_pkg

# Root of the self-contained package.
_PACKAGE_ROOT = Path(_openpi_pytorch_pkg.__file__).resolve().parent
# repo_root/rlinf/models/embodiment/openpi_pytorch -> parents[2] is the `rlinf` dir.
_RLINF_ROOT = _PACKAGE_ROOT.parents[2]
# The relocated BEHAVIOR SFT data modules must also stay free of external openpi.
_BEHAVIOR_DATA_ROOT = _RLINF_ROOT / "data" / "datasets" / "behavior"

# A module name is an external-openpi dependency when its top-level package is
# exactly ``openpi``. ``openpi_pytorch`` does NOT match (it starts with
# ``openpi_``, not ``openpi`` or ``openpi.``).
_FORBIDDEN_TOP_LEVEL = "openpi"


def _is_external_openpi(module_name: str | None) -> bool:
    """Return True when ``module_name``'s top-level package is exactly ``openpi``."""
    if not module_name:
        return False
    return module_name == _FORBIDDEN_TOP_LEVEL or module_name.startswith(
        _FORBIDDEN_TOP_LEVEL + "."
    )


@dataclass(frozen=True)
class Violation:
    file: str
    lineno: int
    statement: str


def _violations_in_source(source: str, filename: str) -> list[Violation]:
    """AST-scan one source string for real external ``openpi`` imports."""
    violations: list[Violation] = []
    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_external_openpi(alias.name):
                    violations.append(
                        Violation(filename, node.lineno, f"import {alias.name}")
                    )
        elif isinstance(node, ast.ImportFrom):
            # node.level > 0 means a relative import (always allowed).
            if node.level == 0 and _is_external_openpi(node.module):
                violations.append(
                    Violation(filename, node.lineno, f"from {node.module} import ...")
                )
    return violations


def find_external_openpi_imports(*roots: Path) -> list[Violation]:
    """Return every statement importing the external ``openpi`` package under ``roots``."""
    violations: list[Violation] = []
    for root in roots:
        if not root.exists():
            continue
        for py_file in sorted(root.rglob("*.py")):
            source = py_file.read_text(encoding="utf-8")
            rel = py_file.relative_to(_RLINF_ROOT.parent)
            violations.extend(_violations_in_source(source, str(rel)))
    return violations


_CORE_MODULES = [
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.utils",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.lora",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.pointnet",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.model",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.siglip",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.gemma",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.normalize",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.processing",
    "rlinf.models.embodiment.openpi_pytorch.pi0_model.tokenizer",
    "rlinf.models.embodiment.openpi_pytorch.utils.old_to_new",
    "rlinf.models.embodiment.openpi_pytorch.utils.new_to_old",
    "rlinf.models.embodiment.openpi_pytorch.utils.jax_to_new_pytorch",
    "rlinf.models.embodiment.openpi_pytorch.utils.export_sft_checkpoint",
    "rlinf.models.embodiment.openpi_pytorch.utils.image_tools",
]


def test_no_external_openpi_imports():
    """The package and relocated BEHAVIOR data modules import no external ``openpi``."""
    violations = find_external_openpi_imports(_PACKAGE_ROOT, _BEHAVIOR_DATA_ROOT)
    assert violations == [], "External `openpi` imports found:\n" + "\n".join(
        f"  {v.file}:{v.lineno}: {v.statement}" for v in violations
    )


def test_ast_detects_injected_external_import():
    """A real ``import openpi`` / ``from openpi import`` is flagged (negative gate)."""
    assert _violations_in_source("import openpi\n", "synthetic.py")
    assert _violations_in_source("from openpi.models import x\n", "synthetic.py")


def test_ast_ignores_openpi_token_outside_imports():
    """The token ``openpi`` in comments/strings/relative imports is NOT a violation."""
    source = (
        "# this references openpi in a comment\n"
        '"""openpi appears in a docstring"""\n'
        "logger_name = 'openpi.thing'\n"
        "from . import sibling  # relative import, allowed\n"
        "import openpi_pytorch_like\n"  # starts with openpi_ but is not openpi
    )
    assert _violations_in_source(source, "synthetic.py") == []


@pytest.mark.parametrize("module_name", _CORE_MODULES)
def test_core_modules_import(module_name):
    """Every vendored core module imports cleanly."""
    importlib.import_module(module_name)


def test_sample_actions_smoke():
    """A tiny Pi0 returns a finite action tensor of the expected shape."""
    import torch

    from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0_config import Pi0Config

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
