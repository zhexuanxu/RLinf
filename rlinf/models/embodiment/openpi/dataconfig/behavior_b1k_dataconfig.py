# Copyright 2025 The RLinf Authors.
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
"""
LeRobotB1KDataConfig — data config factory matching openpi-comet's LeRobotB1KDataConfig.

Uses the same repack keys and B1kInputs transform as openpi-comet to ensure
identical data preprocessing.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies.behavior_policy import (
    B1kInputs,
    B1kOutputs,
)


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data config factory matching openpi-comet's LeRobotB1KDataConfig.

    The fields ``behavior_dataset_root``, ``tasks``, ``fine_grained_level``,
    and ``tolerance_s`` are stored on the factory (not on DataConfig) because
    the installed openpi DataConfig in .venv_pi does not have them.  The
    behavior-specific data loader reads them directly from this factory.
    """

    action_sequence_keys: tuple[str, ...] = ("action",)
    # Fields consumed by the behavior data loader (not part of DataConfig)
    behavior_dataset_root: str = ""
    tasks: list[str] = dataclasses.field(default_factory=list)
    fine_grained_level: int = 0
    tolerance_s: float = 1e-4
    modalities: list[str] = dataclasses.field(default_factory=lambda: ["rgb"])

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/egocentric_camera": "observation.images.rgb.head",
                        "observation/wrist_image_left": "observation.images.rgb.left_wrist",
                        "observation/wrist_image_right": "observation.images.rgb.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                B1kInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[B1kOutputs(action_dim=23)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            use_quantile_norm=True,
        )
