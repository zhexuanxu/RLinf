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

"""Self-contained BEHAVIOR-1K SFT data loader for the PyTorch pi05 path.

Streams the BEHAVIOR dataset (:class:`~.behavior_sft_dataset.BehaviorSftDataset`),
applies the per-sample :class:`BehaviorSftTransform` (state extraction, image
resize/pad, quantile normalization, pi05 discrete-state tokenization), collates
samples into a batched :class:`Observation` plus an actions tensor of shape
``[batch, action_horizon, action_dim]``, and yields ``(Observation, actions)``.

The streaming dataset partitions its keyframe chunks per ``(rank, worker)``
internally (see :meth:`BehaviorSftDataset.__getitem__`), so a
``DistributedSampler`` is intentionally *not* used: a sampler only reorders the
ignored ``idx`` values and would otherwise give every distributed rank identical
data. Every loader parameter is read directly from YAML.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import typing
from pathlib import Path

import numpy as np
import torch

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.lerobot_paths import (
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    CONTROL_MODE_ACTION_ENV_DIM,
    CONTROL_MODES,
    STATE_ORDERS,
    BehaviorInputs,
    resolve_state_token,
)
from rlinf.models.embodiment.openpi_pytorch.utils.image_tools import resize_with_pad
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    NormStats,
    load_norm_stats,
    normalize_quantile,
    validate_norm_stats_for_control_mode,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import PaligemmaTokenizer

logger = logging.getLogger(__name__)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftTransform",
    "build_behavior_sft_dataloader",
    "collate_behavior_sft_items",
    "create_behavior_sft_data_loader",
]

# Camera views resolved by the BEHAVIOR pi05 transform.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_IMAGE_SIZE = 224

# Repack mapping: BehaviorInputs key -> raw LeRobot frame key. Mirrors the old
# LeRobotB1KDataConfig RepackTransform (with the BehaviorInputs key names).
_REPACK_KEYS = {
    "observation/image": "observation.images.rgb.head",
    "observation/left_wrist_image": "observation.images.rgb.left_wrist",
    "observation/right_wrist_image": "observation.images.rgb.right_wrist",
    "observation/state": "observation.state",
}


def _pad_to_dim(x: np.ndarray, target_dim: int, value: float = 0.0) -> np.ndarray:
    """Zero-pad the last axis of ``x`` up to ``target_dim`` (no-op if already >=)."""
    current_dim = x.shape[-1]
    if current_dim >= target_dim:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[-1] = (0, target_dim - current_dim)
    return np.pad(x, pad_width, constant_values=value)


def _repack(frame: dict) -> dict:
    """Map raw LeRobot keys onto the names ``BehaviorInputs`` expects.

    Images arrive as ``(C, H, W)`` float tensors from the streaming dataset;
    ``BehaviorInputs`` (via its ``_parse_image``) handles the channel order and
    the float-to-uint8 conversion, so they are passed through as numpy arrays.
    """
    data: dict = {}
    for dst, src in _REPACK_KEYS.items():
        data[dst] = np.asarray(frame[src])

    actions = frame.get("action")
    if actions is not None:
        data["actions"] = np.asarray(actions)

    prompt = frame.get("prompt", frame.get("task"))
    if prompt is None:
        raise ValueError(
            "BEHAVIOR SFT frame is missing both 'prompt' and 'task'; the streaming "
            "dataset must set the per-frame task text."
        )
    if not isinstance(prompt, str):
        prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
    data["prompt"] = prompt

    response = frame.get("response")
    if response is not None:
        data["response"] = response if isinstance(response, str) else str(response)
    return data


@dataclasses.dataclass
class BehaviorSftTransform:
    """Map a raw BEHAVIOR LeRobot frame to pi05 model inputs + padded actions.

    Reproduces the old chain (repack -> ``BehaviorInputs`` -> quantile-Normalize
    state/actions -> resize images -> pi05 discrete-state tokenize -> pad state and
    actions to ``action_dim``). Images stay ``uint8`` through resize; the final
    ``uint8 -> float[-1, 1]`` conversion happens in :meth:`Observation.from_dict`.

    Args:
        norm_stats: Quantile normalization statistics keyed by ``"state"`` and
            ``"actions"`` (as loaded from the checkpoint ``norm_stats.json``).
        tokenizer_path: Filesystem path to the PaliGemma SentencePiece model.
        action_dim: Model action dimension to pad the state and actions to.
        max_token_len: Maximum tokenized-prompt length.
        image_size: Target square image resolution.
        vlm_vla: Tokenize with the subtask-supervision template (prompt prefix +
            subtask response + EOS, with per-token attention/loss/KV-cache
            masks) instead of the action-only prompt. Frames must carry a
            ``response`` text in this mode.
        discrete_state_input: Whether to discretize normalized state into the
            language prompt before ``Action:`` / ``Subtask:``.
        state_order: Proprio->state channel ordering passed to
            :func:`extract_state_from_proprio` (``"comet"`` reference order or
            ``"align"`` action-aligned). Must match the ``state_order`` used to
            generate ``norm_stats.json``.
        tokenizer: Optional pre-built tokenizer. A new
            :class:`PaligemmaTokenizer` is created lazily per worker when ``None``
            so the (non-picklable) SentencePiece processor is not shared across
            ``spawn`` workers.
    """

    norm_stats: dict[str, NormStats]
    tokenizer_path: str
    action_dim: int = 32
    max_token_len: int = 200
    image_size: int = _IMAGE_SIZE
    vlm_vla: bool = False
    discrete_state_input: bool = True
    state_order: str = "comet"
    tokenizer: PaligemmaTokenizer | None = None

    def __post_init__(self):
        self._behavior_inputs = BehaviorInputs(
            extract_state_from_proprio=True,
            use_all_wrist_images=True,
            state_order=self.state_order,
        )

    def _get_tokenizer(self) -> PaligemmaTokenizer:
        if self.tokenizer is None:
            self.tokenizer = PaligemmaTokenizer(
                self.tokenizer_path, max_len=self.max_token_len
            )
        return self.tokenizer

    def __call__(self, frame: dict) -> dict:
        """Transform a single raw LeRobot frame into the model-input dict."""
        # Repack LeRobot keys -> BehaviorInputs keys (+ prompt), then run
        # BehaviorInputs (23-dim state extraction, image-key mapping, masks).
        repacked = _repack(frame)
        response = repacked.pop("response", None)
        inputs = self._behavior_inputs(repacked)

        # Resize each camera image to image_size x image_size (uint8 in/out).
        images = {
            key: resize_with_pad(
                np.asarray(inputs["image"][key]), self.image_size, self.image_size
            )
            for key in _IMAGE_KEYS
        }

        # Quantile-normalize the state (23-dim proprio) and the actions to
        # [-1, 1] before padding. The action width depends on control_mode
        # (23 for joint_absolute, 21 for eef_delta_pose); normalize_quantile
        # slices the stats to the input width, and the matching norm stats are
        # enforced upstream. When configured, the normalized state is also
        # discretized into the pi05 language prompt.
        state = np.asarray(inputs["state"], dtype=np.float32)
        state = normalize_quantile(state, self.norm_stats["state"]).astype(np.float32)
        actions = np.asarray(inputs["actions"], dtype=np.float32)
        actions = normalize_quantile(actions, self.norm_stats["actions"]).astype(
            np.float32
        )
        token_extras = {}
        prompt_state = state if self.discrete_state_input else None
        if self.vlm_vla:
            if response is None:
                raise ValueError(
                    "vlm_vla SFT requires a per-frame subtask response, but the "
                    "streaming dataset yielded a frame without one."
                )
            tokens, token_masks, ar_mask, loss_mask, kv_cache_mask = (
                self._get_tokenizer().tokenize_with_subtask(
                    inputs["prompt"], prompt_state, response
                )
            )
            token_extras = {
                "token_ar_mask": np.asarray(ar_mask),
                "token_loss_mask": np.asarray(loss_mask),
                "token_kv_cache_mask": np.asarray(kv_cache_mask),
            }
        else:
            tokens, token_masks = self._get_tokenizer().tokenize(
                inputs["prompt"], prompt_state
            )
        state = _pad_to_dim(state, self.action_dim).astype(np.float32)
        actions = _pad_to_dim(actions, self.action_dim).astype(np.float32)

        return {
            "image": images,
            "image_mask": {
                key: np.asarray(inputs["image_mask"][key]) for key in _IMAGE_KEYS
            },
            "state": state,
            "actions": actions,
            "tokenized_prompt": np.asarray(tokens),
            "tokenized_prompt_mask": np.asarray(token_masks),
            **token_extras,
        }


@dataclasses.dataclass(frozen=True)
class BehaviorSftDataConfig:
    """Metadata describing the BEHAVIOR SFT data pipeline.

    Exposed via :meth:`BehaviorSftDataLoader.data_config` so the SFT worker can
    read the resolved repo id, action dimension, action horizon, and the
    normalization statistics without reaching into the dataset internals.
    """

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    norm_stats: dict[str, NormStats]


class _TransformedStreamingDataset(torch.utils.data.Dataset):
    """Wrap the streaming dataset, applying the per-sample SFT transform.

    The transform holds a (non-picklable) SentencePiece tokenizer; it is built
    lazily inside each ``spawn`` worker on first use, so only the lightweight
    :class:`BehaviorSftTransform` config travels across the process boundary.
    """

    def __init__(self, dataset: BehaviorSftDataset, transform: BehaviorSftTransform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        # The streaming dataset ignores `idx` and partitions chunks internally;
        # `len` only drives torch's default index sampler so iteration proceeds.
        return len(self._dataset.hf_dataset)


def collate_behavior_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed items into ``(Observation, actions)``.

    Images are stacked as ``uint8`` ``[B, H, W, C]`` tensors and converted to
    ``float32`` in ``[-1, 1]`` by :meth:`Observation.from_dict` (matching the old
    path). State/actions/tokens are stacked into the appropriate torch dtypes;
    the returned actions tensor has shape ``[batch, action_horizon, action_dim]``.
    """
    if not items:
        raise ValueError("Cannot collate an empty BEHAVIOR SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    # Per-token attention/loss/KV-cache masks (subtask-supervision mode only).
    for mask_key in ("token_ar_mask", "token_loss_mask", "token_kv_cache_mask"):
        if mask_key in items[0]:
            batch[mask_key] = torch.from_numpy(
                np.stack([np.asarray(item[mask_key], dtype=np.bool_) for item in items])
            )
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker init hook (placeholder for worker-local environment setup)."""
    del worker_id


def validate_behavior_data_config_migration(data_cfg) -> None:
    """Reject ``data:`` keys left over from the removed weighted-skill recipe.

    A config still carrying them would otherwise silently train on different
    data than its author intended. ``use_skill`` / ``allow_left`` /
    ``allow_right`` always fail loudly; ``skill_list`` survives only as its
    no-op values — absent, ``None``, or ``["all"]`` — because weighted skill
    sampling lost its label source when the orchestrator machinery was removed.

    Args:
        data_cfg: The experiment's ``data:`` config section.

    Raises:
        ValueError: When a removed key is present, or ``skill_list`` carries
            anything but a no-op value.
    """
    for stale_key in ("use_skill", "allow_left", "allow_right"):
        if stale_key in data_cfg:
            raise ValueError(
                f"data.{stale_key} was removed. Use data.fine_grained_level "
                "(0 = main task only, 1 = main task + subtask response) with "
                "data.enable_gap and actor.model.openpi.mode instead."
            )
    if "skill_list" in data_cfg:
        skill_list = data_cfg.skill_list
        if skill_list is not None and list(skill_list) != ["all"]:
            raise ValueError(
                "Weighted skill sampling was removed; data.skill_list supports "
                f"only its no-op values (absent, null, or ['all']), got "
                f"{skill_list!r}. Use data.fine_grained_level with "
                "data.task_subtasks for subtask supervision instead."
            )


def _validate_task_names(tasks: list[str]) -> None:
    """Validate ``data.tasks`` for subtask-supervised (``fine_grained_level=1``) SFT.

    The task list must be explicit and known: an empty list is a configuration
    error (it is NOT an implicit "all tasks"), and every name must be a key of the
    authoritative ``TASK_NAMES_TO_INDICES`` map. Unknown names are reported by
    name so a typo is easy to spot.

    Args:
        tasks: The resolved ``data.tasks`` list.

    Raises:
        ValueError: If ``tasks`` is empty or names any task absent from
            ``TASK_NAMES_TO_INDICES``.
    """
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        TASK_NAMES_TO_INDICES,
    )

    if not tasks:
        raise ValueError(
            "openpi_pytorch BEHAVIOR SFT at fine_grained_level=1 requires a "
            "non-empty data.tasks listing the task(s) to train on; an empty list "
            "is not treated as 'all tasks'. To train on every task, list all "
            "names explicitly (e.g. via the behavior_task_names() helper)."
        )
    unknown = [name for name in tasks if name not in TASK_NAMES_TO_INDICES]
    if unknown:
        raise ValueError(
            f"data.tasks contains unknown BEHAVIOR task name(s): {unknown}. Valid "
            "names are the keys of TASK_NAMES_TO_INDICES in behavior_sft_dataset."
        )


def create_behavior_sft_data_loader(
    *,
    behavior_dataset_root: str,
    assets_dir: str,
    asset_id: str,
    tokenizer_path: str,
    repo_id: str,
    tasks: list[str],
    modalities: list[str],
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    fine_grained_level: int,
    tolerance_s: float,
    shuffle: bool,
    seed: int,
    subtask_labels: dict[int, str] | None,
    enable_gap: bool,
    vlm_vla: bool,
    discrete_state_input: bool,
    state_order: str,
    dist_rank: int,
    dist_world_size: int,
    hf_cache_dir: str | None = None,
) -> "BehaviorSftDataLoader":
    """Build the BEHAVIOR-1K SFT data loader yielding ``(Observation, actions)``.

    Args:
        behavior_dataset_root: Local root of the LeRobot BEHAVIOR dataset.
        assets_dir: Directory holding the checkpoint assets (norm stats).
        asset_id: Sub-directory under ``assets_dir`` for the norm stats
            (``{assets_dir}/{asset_id}/norm_stats.json``).
        tokenizer_path: Filesystem path to the PaliGemma SentencePiece model.
        repo_id: LeRobot dataset repo id (used for metadata bookkeeping).
        tasks: BEHAVIOR task names to include.
        modalities: Observation modalities to load (e.g. ``["rgb"]``).
        action_dim: Model action dimension to pad state/actions to.
        action_horizon: Number of future action steps per sample.
        max_token_len: Maximum tokenized-prompt length.
        batch_size: Per-rank batch size.
        num_workers: Number of ``DataLoader`` workers (``> 0`` uses ``spawn``).
        fine_grained_level: Per-frame text granularity (``0`` = main task only;
            ``1`` = main task + subtask response).
        tolerance_s: Frame-timestamp sync tolerance.
        shuffle: Whether the streaming dataset shuffles its chunk order.
        seed: Base seed for the streaming chunk partition.
        subtask_labels: Per-task subtask labels keyed by ``skill_idx`` (required
            at ``fine_grained_level=1``).
        enable_gap: Assign gap frames to the next skill window (True) or skip
            them (False); only consulted at ``fine_grained_level=1``.
        vlm_vla: Tokenize with the subtask-supervision template (per-token
            masks for the VLM CE loss) instead of the action-only prompt.
        discrete_state_input: Whether to inject normalized state as pi05
            discrete language tokens.
        state_order: Proprio->state channel ordering (``"comet"`` or
            ``"align"``); must match the ordering used to generate the norm stats.
        dist_rank: This rank's id, threaded into the per-rank chunk partition.
        dist_world_size: Total ranks, threaded into the per-rank chunk partition.
        hf_cache_dir: Directory for the on-disk Arrow cache that ``load_dataset``
            materializes from the parquet. Must be on a disk with room for the
            whole low-dim dataset; when ``None`` the dataset defaults it to a
            sibling of the dataset root (see :class:`BehaviorSftDataset`).

    Returns:
        A loader whose iteration yields ``(Observation, actions)`` 2-tuples.
    """
    norm_stats = load_norm_stats(assets_dir, asset_id)
    logger.info("Loaded BEHAVIOR norm stats from %s/%s", assets_dir, asset_id)

    dataset = BehaviorSftDataset(
        repo_id=repo_id,
        root=behavior_dataset_root,
        tolerance_s=tolerance_s,
        tasks=tasks or None,
        modalities=modalities or ["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(action_horizon)]},
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=fine_grained_level,
        subtask_labels=subtask_labels,
        enable_gap=enable_gap,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
        hf_cache_dir=hf_cache_dir,
    )

    transform = BehaviorSftTransform(
        norm_stats=norm_stats,
        tokenizer_path=tokenizer_path,
        action_dim=action_dim,
        max_token_len=max_token_len,
        vlm_vla=vlm_vla,
        discrete_state_input=discrete_state_input,
        state_order=state_order,
    )
    source = _TransformedStreamingDataset(dataset, transform)

    # The streaming dataset partitions chunks per (rank, worker) on its own, so a
    # DistributedSampler is intentionally omitted: it would only reorder the
    # ignored `idx` values and give every distributed rank identical data.
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "BEHAVIOR SFT data loader: batch_size=%d, num_workers=%d, action_horizon=%d",
        batch_size,
        num_workers,
        action_horizon,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_behavior_sft_items,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    data_config = BehaviorSftDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        norm_stats=norm_stats,
    )
    return BehaviorSftDataLoader(torch_loader, data_config)


class BehaviorSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over the BEHAVIOR SFT dataset.

    Re-iterates the underlying ``torch`` ``DataLoader`` forever. Each batch is
    already collated into an :class:`Observation` plus an actions tensor of shape
    ``[batch, action_horizon, action_dim]`` by :func:`collate_behavior_sft_items`.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: BehaviorSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> BehaviorSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Expose the underlying ``torch`` ``DataLoader``."""
        return self._torch_loader

    def __iter__(self):
        while True:
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_behavior_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    """Build the self-contained BEHAVIOR SFT data loader for the SFT worker.

    The streaming dataset partitions chunks per ``(rank, worker)``; ``rank`` /
    ``world_size`` are captured here (in the main process) and threaded into the
    dataset so that SPAWNED DataLoader workers -- which cannot read
    ``torch.distributed`` -- still partition by the correct per-rank id (otherwise
    every rank replicates rank 0's chunks, collapsing the effective batch to one
    rank's micro-batch). Every parameter is read directly from YAML (no hidden
    defaults). Returns ``(loader, loader.data_config())``.
    """
    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_pytorch BEHAVIOR SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    validate_behavior_data_config_migration(data_cfg)

    # Norm stats + tokenizer resolve STRICTLY from YAML (no checkpoint-relative
    # fallback); load_norm_stats rejects a blank assets_dir/asset_id the same way
    # the eval model factory does, so neither path can silently load non-task-0000
    # stats. assets_dir/asset_id are resolved from control_mode below (the switch);
    # only the tokenizer is mode-independent and read here.
    tokenizer_path = model_cfg.openpi.paligemma_tokenizer

    # The model mode and the data granularity are one decision: action-only
    # training (`vla`) reads main-task prompts (level 0), while VLM subtask
    # supervision (`vlm_vla`) requires the subtask response labels (level 1).
    # Any other combination is a configuration error, rejected here at build.
    mode = str(model_cfg.openpi.get("mode", "vla"))
    if mode not in ("vla", "vlm_vla"):
        raise ValueError(
            f"actor.model.openpi.mode must be 'vla' or 'vlm_vla', got {mode!r}."
        )
    discrete_state_input = model_cfg.openpi.get("discrete_state_input", True)
    if not isinstance(discrete_state_input, bool):
        raise TypeError(
            "actor.model.openpi.discrete_state_input must be a boolean, got "
            f"{discrete_state_input!r}."
        )
    # Prompt state channel layout. Canonical key is openpi.state_token
    # (abs_joint_old / abs_joint / abs_eef); the legacy openpi.state_order
    # (comet / align) still resolves. resolve_state_token normalizes + validates.
    raw_state_token = model_cfg.openpi.get(
        "state_token", model_cfg.openpi.get("state_order", "abs_joint_old")
    )
    state_order = resolve_state_token(str(raw_state_token))
    # Robot action space. "joint_absolute" is the original 23-dim absolute-joint
    # dataset; "eef_delta_pose" is the 21-dim delta-EEF converted dataset. The
    # field is REQUIRED for BEHAVIOR pi0.5 SFT (every shipped template declares
    # it); a missing key is a malformed/outdated config and fails loudly rather
    # than silently defaulting. The checks reject a config whose control_mode
    # disagrees with EITHER action_dim or action_env_dim, so SFT can never
    # silently normalize 21-dim actions against 23-dim stats.
    if "control_mode" not in model_cfg.openpi:
        raise ValueError(
            "actor.model.openpi.control_mode is required for BEHAVIOR pi0.5 SFT "
            f"(one of {CONTROL_MODES}); it is declared in the shared model "
            "template model/pi0_5_pytorch.yaml. A missing key indicates a "
            "malformed or outdated config."
        )
    control_mode = str(model_cfg.openpi.control_mode)
    if control_mode not in CONTROL_MODES:
        raise ValueError(
            f"actor.model.openpi.control_mode must be one of {CONTROL_MODES}, got "
            f"{control_mode!r}."
        )
    expected_env_dim = CONTROL_MODE_ACTION_ENV_DIM[control_mode]
    action_dim_cfg = int(model_cfg.action_dim)
    action_env_dim = int(model_cfg.openpi.get("action_env_dim", action_dim_cfg))
    if action_dim_cfg != expected_env_dim or action_env_dim != expected_env_dim:
        raise ValueError(
            f"actor.model.openpi.control_mode={control_mode!r} expects a semantic "
            f"action dim of {expected_env_dim}, but actor.model.action_dim="
            f"{action_dim_cfg} / openpi.action_env_dim={action_env_dim}. Set "
            f"action_dim to {expected_env_dim} for this control mode (the model "
            f"still pads to openpi.model_action_dim)."
        )
    # control_mode is the switch: resolve the dataset root + norm-stats asset for
    # this mode. resolve_behavior_paths prefers optional mode-specific YAML fields
    # (e.g. behavior_dataset_root_eef_delta / asset_id_eef_delta) and otherwise
    # falls back to the base fields, so one config can carry both modes' paths and
    # flipping control_mode selects the matching data/stats. Paths stay in YAML
    # (no hardcoded filesystem defaults in code); the validations below reject any
    # residual mode/dataset/stats mismatch.
    from rlinf.data.datasets.openpi_pytorch.behavior.convert_to_eef_delta import (
        resolve_behavior_paths,
        validate_converted_dataset,
    )

    resolved = resolve_behavior_paths(data_cfg, model_cfg.openpi, control_mode)
    behavior_dataset_root = str(resolved["behavior_dataset_root"])
    assets_dir = resolved["assets_dir"]
    asset_id = resolved["asset_id"]
    # Reject a norm-stats asset whose manifest disagrees with this run's mode/dim
    # (delta-EEF assets carry a manifest; legacy joint stats without one pass only
    # for joint_absolute).
    validate_norm_stats_for_control_mode(
        assets_dir, asset_id, control_mode, expected_env_dim,
        model_action_dim=int(model_cfg.openpi.model_action_dim),
        state_token=state_order,
    )
    # Reject a dataset root that does not match the control mode: for
    # eef_delta_pose the root must be a converted delta-EEF dataset (action
    # shape 21 + a matching provenance covering data.tasks); for joint_absolute
    # it must be the original 23-dim dataset. This closes the hole where a delta
    # config could stream the old 23-dim parquet while passing config-dim checks.
    validate_converted_dataset(behavior_dataset_root, control_mode, list(data_cfg.tasks))
    fine_grained_level = int(data_cfg.fine_grained_level)
    if fine_grained_level not in (0, 1):
        raise ValueError(
            f"data.fine_grained_level must be 0 or 1, got {fine_grained_level!r}."
        )
    if mode == "vla" and fine_grained_level != 0:
        raise ValueError(
            "Subtask supervision (data.fine_grained_level=1) is only allowed with "
            "actor.model.openpi.mode=vlm_vla; vla mode trains on level 0."
        )
    if mode == "vlm_vla" and fine_grained_level != 1:
        raise ValueError(
            "actor.model.openpi.mode=vlm_vla requires data.fine_grained_level=1 "
            "(the VLM CE loss needs a subtask response to supervise)."
        )
    enable_gap = bool(data_cfg.get("enable_gap", True))

    tasks = list(data_cfg.tasks)
    if fine_grained_level == 1:
        # Subtask supervision no longer needs a per-task label list: each frame's
        # subtask text is resolved at runtime from that frame's own episode
        # annotation (see BehaviorSftDataset._resolve_subtask_text), which is
        # correct even for the many tasks whose skill sequence varies per episode.
        # `data.tasks` must name the tasks to train on explicitly; an empty list
        # is a configuration error rather than an implicit "all tasks".
        _validate_task_names(tasks)

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=behavior_dataset_root,
        assets_dir=str(assets_dir),
        asset_id=asset_id,
        tokenizer_path=str(tokenizer_path),
        repo_id=str(data_cfg.repo_id),
        tasks=tasks,
        modalities=list(data_cfg.modalities),
        action_dim=int(model_cfg.openpi.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(model_cfg.openpi.max_token_len),
        batch_size=int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size),
        num_workers=int(data_cfg.num_workers),
        fine_grained_level=fine_grained_level,
        tolerance_s=float(data_cfg.tolerance_s),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        # Subtask text is resolved per frame from each episode's own annotation;
        # no per-task label list is threaded through.
        subtask_labels=None,
        enable_gap=enable_gap,
        vlm_vla=(mode == "vlm_vla"),
        discrete_state_input=discrete_state_input,
        state_order=state_order,
        dist_rank=rank,
        dist_world_size=world_size,
        # Steer the on-disk Arrow cache onto a large disk. ``load_dataset`` writes a
        # full re-encoded copy of the low-dim parquet (~140 GB for 50 tasks) before
        # training; the HuggingFace default lands under HF_HOME (often a small
        # container overlay) and overflows with ENOSPC. Read from YAML, else fall
        # back to a sibling of the dataset root so it follows the dataset's disk.
        hf_cache_dir=str(
            data_cfg.get("hf_cache_dir", None)
            or (Path(str(data_cfg.behavior_dataset_root)).parent / ".hf_datasets_cache")
        ),
    )
    return loader, loader.data_config()
