# Copyright 2026 The RLinf Authors.
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

"""Lazy public exports for the BEHAVIOR OpenPI PyTorch data package.

Dataset conversion and norm-stat CLIs intentionally depend only on NumPy,
PyArrow, and SciPy. Loading the SFT classes eagerly here would pull the full
training stack before their argument parsers can even display ``--help``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .behavior_sft_data_loader import (
        BehaviorSftDataConfig,
        BehaviorSftDataLoader,
        build_behavior_sft_dataloader,
        create_behavior_sft_data_loader,
    )
    from .behavior_sft_dataset import BehaviorSftDataset

_EXPORT_MODULES = {
    "BehaviorSftDataConfig": ".behavior_sft_data_loader",
    "BehaviorSftDataLoader": ".behavior_sft_data_loader",
    "build_behavior_sft_dataloader": ".behavior_sft_data_loader",
    "create_behavior_sft_data_loader": ".behavior_sft_data_loader",
    "BehaviorSftDataset": ".behavior_sft_dataset",
}

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftDataset",
    "build_behavior_sft_dataloader",
    "create_behavior_sft_data_loader",
]


def __getattr__(name: str) -> Any:
    """Import a public SFT symbol only when a training caller requests it."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
