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

"""BEHAVIOR data-config package for the self-contained PyTorch OpenPI 0.5 model.

The BEHAVIOR streaming SFT pipeline (dataset / transform / data loader) now lives
under ``rlinf.data.datasets.behavior``; import it from there. This package is
reserved for the YAML-driven BEHAVIOR data config that mirrors
``openpi/dataconfig`` (no hard-coded ``TrainConfig`` registry).
"""
