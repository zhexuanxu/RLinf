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

"""Package-layout guardrails for the reorganized ``openpi_pytorch`` package.

After the Phase-3 reorganization, the package top level holds only
``openpi_action_model.py`` (plus ``__init__.py``); the Pi0 model internals and
preprocessing live under ``pi0_model/`` and the checkpoint / image tooling lives
under ``utils/``. These tests assert that end-state and that no source file
anywhere references a pre-refactor import path.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import rlinf.models.embodiment.openpi_pytorch as _pkg

_PACKAGE_ROOT = Path(_pkg.__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parents[3]

# The directional checkpoint converters + the SFT export + image tooling that
# the new ``utils/`` package must expose (AC-1 layout end-state).
_REQUIRED_UTILS_MODULES = (
    "old_to_new",
    "new_to_old",
    "jax_to_new_pytorch",
    "export_sft_checkpoint",
    "image_tools",
)

# Pre-refactor / pre-rename import paths that must no longer appear anywhere in
# the source. New valid paths interpose ``pi0_model.`` / the directional utility
# names and so do not match.
_FORBIDDEN_OLD_PATHS = re.compile(
    r"openpi_pytorch\.(?:normalize|processing|tokenizer|image_tools|"
    r"convert_checkpoint|export_checkpoint|_import_isolation)\b"
    r"|openpi_pytorch\.utils\.(?:model|pi0|pi0_config|gemma|siglip|pointnet|lora|utils|"
    r"convert_checkpoint|export_checkpoint|checkpoint_format)\b"
)


def test_required_utility_modules_importable():
    """The AC-1 directional converter + export + image-tool modules all import."""
    for mod in _REQUIRED_UTILS_MODULES:
        importlib.import_module(
            f"rlinf.models.embodiment.openpi_pytorch.utils.{mod}"
        )


def test_top_level_is_only_action_model():
    """The package top level contains exactly the action model + package marker."""
    top_level_py = {p.name for p in _PACKAGE_ROOT.glob("*.py")}
    assert top_level_py == {"openpi_action_model.py", "__init__.py"}, top_level_py


def test_expected_subpackages_exist():
    """The reorganized subpackages are present."""
    for sub in ("pi0_model", "utils", "dataconfig", "policies"):
        assert (_PACKAGE_ROOT / sub / "__init__.py").is_file(), sub


def test_no_old_import_paths_anywhere():
    """No source file under rlinf/, tests/, examples/ references an old import path."""
    offenders: list[str] = []
    for sub in ("rlinf", "tests", "examples"):
        root = _REPO_ROOT / sub
        if not root.exists():
            continue
        for py_file in root.rglob("*.py"):
            if py_file.resolve() == Path(__file__).resolve():
                continue  # this file names the forbidden patterns as regex literals
            text = py_file.read_text(encoding="utf-8")
            for m in _FORBIDDEN_OLD_PATHS.finditer(text):
                lineno = text.count("\n", 0, m.start()) + 1
                rel = py_file.relative_to(_REPO_ROOT)
                offenders.append(f"{rel}:{lineno}: {m.group(0)}")
    assert not offenders, "Old import paths still present:\n" + "\n".join(offenders)
