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

"""Static import-isolation check for the self-contained PyTorch OpenPI 0.5 package.

The package must not depend on the externally installed top-level ``openpi``
distribution: the whole point of vendoring is that the embodied eval path runs
without that dependency (and without patching ``transformers``).

This module uses the ``ast`` module (not text matching) so that the string
``"openpi"`` appearing in comments, docstrings, or logger names is never
mistaken for a dependency. Only real ``import openpi`` / ``from openpi import``
statements that resolve to the top-level ``openpi`` package are reported.
Relative imports (``from . import x``) and absolute imports of this package
(``rlinf.models.embodiment.openpi_pytorch...``) are allowed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent

# A module name is an external-openpi dependency when its top-level package is
# exactly ``openpi``. Note that ``openpi_pytorch`` does NOT match, because it is
# neither equal to ``openpi`` nor prefixed by ``openpi.`` (it starts with
# ``openpi_``).
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


def find_external_openpi_imports(package_root: Path = PACKAGE_ROOT) -> list[Violation]:
    """Return every statement that imports the external ``openpi`` package."""
    violations: list[Violation] = []
    for py_file in sorted(package_root.rglob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        rel = py_file.relative_to(package_root.parent)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_external_openpi(alias.name):
                        violations.append(
                            Violation(str(rel), node.lineno, f"import {alias.name}")
                        )
            elif isinstance(node, ast.ImportFrom):
                # node.level > 0 means a relative import (always allowed).
                if node.level == 0 and _is_external_openpi(node.module):
                    violations.append(
                        Violation(str(rel), node.lineno, f"from {node.module} import ...")
                    )
    return violations


def main() -> int:
    violations = find_external_openpi_imports()
    if violations:
        print("Import-isolation FAILED: external `openpi` imports found:")
        for v in violations:
            print(f"  {v.file}:{v.lineno}: {v.statement}")
        return 1
    print("Import-isolation OK: no external `openpi` imports under the package.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
