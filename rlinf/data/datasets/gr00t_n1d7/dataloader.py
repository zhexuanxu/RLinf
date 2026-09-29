# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Stateful sequence loading through the standard RLinf VLA SFT worker."""

import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import dataset_settings

from .sequence_dataset import (
    EpisodeDataset,
    EpisodeSampler,
    SequenceCollator,
    build_processor,
    data_metadata,
)


def build_gr00t_sequence_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    if eval_dataset:
        raise ValueError(
            "This protocol uses all episodes for training; no validation split"
        )
    settings = dataset_settings(cfg)
    if str(data_paths) != settings["data"]["path"]:
        raise ValueError("Dataloader path differs from the resolved training data")
    processor, loader, bundle = build_processor(settings)
    dataset = EpisodeDataset(settings, processor, loader)
    if (
        cfg.data.get("expected_episodes") is not None
        and len(dataset) != cfg.data.expected_episodes
    ):
        raise ValueError("Unexpected episode count")
    sampler = EpisodeSampler(len(dataset), cfg.actor.seed, 0, rank, world_size)
    options = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": cfg.actor.micro_batch_size,
        "collate_fn": SequenceCollator(processor),
        "num_workers": cfg.data.num_workers,
        "pin_memory": True,
        "drop_last": False,
        "generator": torch.Generator().manual_seed(cfg.actor.seed + rank),
    }
    if cfg.data.num_workers:
        options.update(
            prefetch_factor=cfg.data.prefetch_factor, multiprocessing_context="spawn"
        )
    dataloader = StatefulDataLoader(**options)
    return dataloader, {
        "settings": settings,
        "processor": bundle,
        "data": data_metadata(settings),
    }
