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

"""Loader-level migration validation for the BEHAVIOR SFT ``data:`` section.

Pins the config-surface contract: the removed weighted-skill keys fail loudly,
while ``skill_list`` is accepted exactly in its no-op forms (absent, ``None``,
or ``["all"]``).
"""

import pytest
from omegaconf import OmegaConf

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
    validate_behavior_data_config_migration,
)


def _data_cfg(**entries):
    return OmegaConf.create({"fine_grained_level": 0, "tasks": ["t"], **entries})


class TestRemovedKeysAreRejected:
    @pytest.mark.parametrize(
        ("key", "value"),
        [("use_skill", False), ("allow_left", 0), ("allow_right", 100)],
    )
    def test_removed_key_rejected_even_with_inert_value(self, key, value):
        with pytest.raises(ValueError, match=f"data.{key} was removed"):
            validate_behavior_data_config_migration(_data_cfg(**{key: value}))

    def test_weighted_skill_list_rejected(self):
        with pytest.raises(ValueError, match="Weighted skill sampling was removed"):
            validate_behavior_data_config_migration(
                _data_cfg(skill_list=["press radio:2.0"])
            )

    def test_partial_skill_list_rejected(self):
        with pytest.raises(ValueError, match="no-op values"):
            validate_behavior_data_config_migration(
                _data_cfg(skill_list=["all", "press radio:2.0"])
            )


class TestNoOpValuesAreAccepted:
    def test_absent_skill_list_passes(self):
        validate_behavior_data_config_migration(_data_cfg())

    def test_all_skill_list_passes(self):
        validate_behavior_data_config_migration(_data_cfg(skill_list=["all"]))

    def test_null_skill_list_passes(self):
        validate_behavior_data_config_migration(_data_cfg(skill_list=None))
