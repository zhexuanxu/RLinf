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
Behavior-specific data loader mirroring openpi-comet's create_behavior_data_loader.
"""

import logging
import multiprocessing
import os
import typing

import jax
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

from rlinf.models.embodiment.openpi.dataconfig.behavior_b1k_dataconfig import (
    LeRobotB1KDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.behavior_dataset import (
    BehaviorLeRobotDataset,
    PromptFromLeRobotItem,
)

logger = logging.getLogger(__name__)


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    shuffle: bool = True,
    skip_norm_stats: bool = False,
    seed: int = 0,
):
    """Create a PyTorch data loader that mirrors openpi-comet's behavior data loader.

    Args:
        config: The TrainConfig whose ``data`` field is a LeRobotB1KDataConfig factory.
        shuffle: Whether to shuffle chunks.
        skip_norm_stats: Whether to skip normalization.
        seed: Random seed.

    Returns:
        DataLoaderImpl wrapping the behaviour dataset with all transforms applied.
    """
    data_factory: LeRobotB1KDataConfig = config.data
    data_config = data_factory.create(config.assets_dirs, config.model)

    dataset = BehaviorLeRobotDataset(
        repo_id=data_factory.base_config.repo_id or "behavior-1k/2025-challenge-demos",
        root=data_factory.behavior_dataset_root,
        tolerance_s=data_factory.tolerance_s,
        tasks=data_factory.tasks or None,
        modalities=data_factory.modalities,
        local_only=True,
        delta_timestamps={
            key: [t / 30.0 for t in range(config.model.action_horizon)]
            for key in data_factory.action_sequence_keys
        },
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=data_factory.fine_grained_level,
    )

    # Apply prompt transform (extracts 'task' → 'prompt')
    dataset = _data_loader.TransformedDataset(dataset, [PromptFromLeRobotItem()])

    # Apply repack + data_transforms + normalize + model_transforms
    dataset = _data_loader.transform_dataset(
        dataset, data_config, skip_norm_stats=skip_norm_stats
    )

    # Build PyTorch DataLoader with DistributedSampler
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = config.batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = config.batch_size

    logger.info(f"Behavior B1K data loader: local_batch_size={local_batch_size}")

    mp_context = None
    num_workers = config.num_workers
    if num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

    generator = torch.Generator()
    generator.manual_seed(seed)

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, dataset),
        batch_size=local_batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_collate_fn,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    return _BehaviorDataLoaderImpl(data_config, torch_loader)


def _collate_fn(items):
    """Collate batch elements into batched numpy arrays."""
    return jax.tree.map(
        lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items
    )


def _worker_init_fn(worker_id: int) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class _TorchDataLoaderShim:
    """Shim matching the openpi TorchDataLoader interface so
    ``_openpi_pytorch_dataloader`` can unwrap to the inner torch DataLoader."""

    def __init__(self, torch_loader):
        self._data_loader = torch_loader
        self.torch_loader = torch_loader


class _BehaviorDataLoaderImpl:
    """Wraps a PyTorch DataLoader with an infinite iteration loop,
    matching the openpi TorchDataLoader behavior."""

    def __init__(self, data_config, torch_loader):
        self._data_config = data_config
        self._torch_loader = torch_loader
        # Expose _data_loader matching openpi DataLoaderImpl interface
        # so get_max_steps_per_epoch can call len() on the inner torch loader.
        self._data_loader = _TorchDataLoaderShim(torch_loader)

    def data_config(self):
        return self._data_config

    def __iter__(self):
        while True:
            data_iter = iter(self._torch_loader)
            while True:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                # Convert to (Observation, actions) 2-tuple, matching
                # openpi's DataLoaderImpl.__iter__ behaviour.
                # Ensure actions are torch tensors (the batch comes as numpy).
                actions = torch.as_tensor(batch["actions"])
                yield _model.Observation.from_dict(batch), actions

    def __len__(self):
        return len(self._torch_loader)

    def __len__(self):
        return len(self._torch_loader)
