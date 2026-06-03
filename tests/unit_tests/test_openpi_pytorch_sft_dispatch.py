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

"""SFT-worker dispatch for the relocated BEHAVIOR data loader (AC-6).

The BEHAVIOR SFT data code now lives under ``rlinf.data.datasets.behavior`` and
the worker dispatches to ``build_behavior_sft_dataloader`` the same way it
dispatches DreamZero — a direct import + call with
``(cfg, world_size, rank, data_paths, eval_dataset)``. These tests assert that
dispatch contract and that the old inline builder / old data import are gone.
"""

from __future__ import annotations

import inspect
import pathlib

from omegaconf import OmegaConf

import rlinf.data.datasets.behavior as behavior_pkg
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def test_worker_dispatches_openpi_pytorch_to_behavior_builder(monkeypatch):
    """An OPENPI_PYTORCH config routes to build_behavior_sft_dataloader verbatim."""
    captured = {}

    def fake_builder(cfg, world_size, rank, data_paths, eval_dataset):
        captured["args"] = (cfg, world_size, rank, data_paths, eval_dataset)
        return ("loader", "data_config")

    monkeypatch.setattr(behavior_pkg, "build_behavior_sft_dataloader", fake_builder)

    worker = FSDPVlaSftWorker.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"model_type": "openpi_pytorch"}}}
    )
    worker._world_size = 8
    worker._rank = 3

    result = worker.build_dataloader("data/path/x", eval_dataset=True)

    assert result == ("loader", "data_config")
    # Exactly (cfg, world_size, rank, data_paths, eval_dataset), DreamZero-style.
    assert captured["args"] == (worker.cfg, 8, 3, "data/path/x", True)


def test_worker_has_no_inline_builder_or_old_data_import():
    """The inline `_build_openpi_pytorch_dataloader` and old data import are gone."""
    src = inspect.getsource(FSDPVlaSftWorker)
    assert "_build_openpi_pytorch_dataloader" not in src
    worker_text = pathlib.Path(inspect.getfile(FSDPVlaSftWorker)).read_text()
    assert "openpi_pytorch.dataconfig" not in worker_text
