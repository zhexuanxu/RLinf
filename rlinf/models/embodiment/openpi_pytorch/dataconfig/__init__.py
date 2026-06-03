# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Self-contained BEHAVIOR-1K SFT data pipeline for the PyTorch OpenPI package.

Ports the old (installed-``openpi``) BEHAVIOR streaming SFT pipeline into the
vendored ``openpi_pytorch`` package with zero installed-``openpi`` imports. All
preprocessing primitives (state extraction, image resize/pad, quantile
normalization, and the PaliGemma tokenizer) are reused from the vendored
``openpi_pytorch`` modules.
"""

from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_data_loader import (
    BehaviorSftDataConfig,
    BehaviorSftDataLoader,
    collate_behavior_sft_items,
    create_behavior_sft_data_loader,
)
from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.models.embodiment.openpi_pytorch.dataconfig.behavior_sft_transform import (
    BehaviorSftTransform,
    transform_behavior_sft_item,
)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftDataset",
    "BehaviorSftTransform",
    "collate_behavior_sft_items",
    "create_behavior_sft_data_loader",
    "transform_behavior_sft_item",
]
