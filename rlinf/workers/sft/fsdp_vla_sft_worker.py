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
import os
from typing import Any

import torch
from omegaconf import DictConfig
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.config import SupportedModel
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.utils import get_rng_state, set_rng_state
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        # Initialize before super().__init__() — the parent's init calls
        # self.build_dataloader(), which reads self._pi05_tokenizer.
        self._pi05_tokenizer = None
        super().__init__(cfg)
        self._full_pi05_loss = None

    def _is_pi05_vlm_only(self) -> bool:
        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.OPENPI:
            return False
        full_pi05 = getattr(self.cfg.actor.model.openpi, "full_pi05", False)
        forward_mode = getattr(self.cfg.actor.model.openpi, "forward_mode", "vla")
        return bool(full_pi05) and forward_mode == "vlm"

    def build_dataloader(self, data_paths: Any, eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # Check for pi0.5 VLM-only mode — uses custom dataset instead of openpi data loader
            if self._is_pi05_vlm_only():
                return self._build_pi05_vlm_dataloader(data_paths, eval_dataset=eval_dataset)
            repo_id = resolve_lerobot_repo_id(data_paths)
            if repo_id is None:
                raise ValueError(
                    "OpenPI SFT requires data.train_data_paths to be set to a local "
                    "dataset path or LeRobot repo id."
                )

            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )

            # Use behavior-specific data loader for B1K configs (BehaviorLeRobotDataset
            # with chunk streaming), matching openpi-comet's data pipeline exactly.
            from rlinf.models.embodiment.openpi.dataconfig.behavior_b1k_dataconfig import (
                LeRobotB1KDataConfig,
            )
            if isinstance(config.data, LeRobotB1KDataConfig):
                from rlinf.models.embodiment.openpi.dataconfig.behavior_data_loader import (
                    create_behavior_data_loader,
                )
                data_loader = create_behavior_data_loader(
                    config, shuffle=True, seed=self.cfg.actor.get("seed", 42),
                )
                return data_loader, data_loader.data_config()

            # Standard openpi data loader path (non-behavior datasets)
            # Force pyav video backend — torchcodec may be importable but broken
            # at runtime (missing FFmpeg libs or PyTorch version mismatch).
            # Also increase video timestamp tolerance for datasets with imprecise
            # frame alignment (e.g., BEHAVIOR videos at 30fps need >=1/30s tolerance).
            try:
                import lerobot.common.datasets.video_utils as _vutils
                _orig_codec = _vutils.get_safe_default_codec
                def _pyav_fallback():
                    try:
                        from torchcodec.decoders import VideoDecoder  # noqa: F401
                        return _orig_codec()
                    except Exception:
                        return "pyav"
                _vutils.get_safe_default_codec = _pyav_fallback
            except ImportError:
                pass
            try:
                import lerobot.common.datasets.lerobot_dataset as _lrd
                _orig_lrd_init = _lrd.LeRobotDataset.__init__
                def _init_with_tolerance(self_ds, *args, **kwargs):
                    kwargs.setdefault("tolerance_s", 1.0)
                    _orig_lrd_init(self_ds, *args, **kwargs)
                _lrd.LeRobotDataset.__init__ = _init_with_tolerance
            except ImportError:
                pass

            import openpi.training.data_loader as openpi_data_loader
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.DREAMZERO
        ]:
            from rlinf.data.datasets.dreamzero import (
                build_dreamzero_sft_dataloader,
            )

            return build_dreamzero_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths, eval_dataset
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _build_pi05_vlm_dataloader(self, data_paths, eval_dataset: bool = False):
        """Build dataloader for pi0.5 VLM-only SFT (BEHAVIOR skill prediction)."""
        data_dir = data_paths[0] if isinstance(data_paths, list) else data_paths
        data_cfg = self.cfg.data
        dataset_name = data_cfg.get("dataset_name", "behavior_skill_pi05")
        max_token_len = data_cfg.get(
            "max_token_len",
            getattr(self.cfg.actor.model.openpi, "max_token_len", 200),
        )
        num_images = getattr(self.cfg.actor.model.openpi, "num_images_in_input", 1)

        if dataset_name == "behavior_skill_pi05":
            from rlinf.models.embodiment.openpi.dataconfig.behavior_vlm_data_loader import (
                create_behavior_vlm_data_loader,
            )

            batch_size = (
                self.eval_batch_size if eval_dataset else self.micro_batch_size
            )
            num_workers = data_cfg.get("num_workers", 4)
            data_loader, tokenizer = create_behavior_vlm_data_loader(
                data_root=data_dir,
                tasks=["turning_on_radio"],
                max_token_len=max_token_len,
                num_images=num_images,
                eval_mode=eval_dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                seed=self.cfg.actor.get("seed", 42),
            )
            if self._pi05_tokenizer is None:
                self._pi05_tokenizer = tokenizer
            split = "eval" if eval_dataset else "train"
            return data_loader, {
                "dataset": "behavior_skill_pi05",
                "split": split,
                "num_samples": len(data_loader.dataset),
            }
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}")

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: Any) -> tuple[torch.Tensor, dict[str, Any]]:
        with self.amp_context:
            output = self.model(forward_type=ForwardType.SFT, data=batch)

        if isinstance(output, torch.Tensor):
            loss = output
        else:
            loss = output["loss"]

        step_metrics = {"loss": loss.detach().item()}
        if isinstance(output, dict) and output.get("dynamics_loss", None) is not None:
            step_metrics.update(
                {
                    "dynamics_loss": output["dynamics_loss"].detach().item(),
                    "action_loss": output["action_loss"].detach().item(),
                }
            )
        return loss, step_metrics

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        super().save_checkpoint(save_path, step)

        if isinstance(self.data_loader, StatefulDataLoader):
            state = self.data_loader.state_dict()

            all_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_states, state)

            if self._rank == 0:
                torch.save(all_states, os.path.join(save_path, "data.pt"))

            torch.distributed.barrier()

            rng_state = get_rng_state()
            all_rng_states = [None] * self._world_size
            torch.distributed.all_gather_object(all_rng_states, rng_state)
            if self._rank == 0:
                torch.save(all_rng_states, os.path.join(save_path, "rng.pt"))

            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str) -> None:
        super().load_checkpoint(load_path)

        if isinstance(self.data_loader, StatefulDataLoader):
            all_states = torch.load(
                os.path.join(load_path, "data.pt"), weights_only=False
            )
            state = all_states[self._rank]
            self.data_loader.load_state_dict(state)
            self.data_iter = iter(self.data_loader)

            rng_path = os.path.join(load_path, "rng.pt")
            if os.path.exists(rng_path):
                all_rng_states = torch.load(rng_path, weights_only=False)
                set_rng_state(all_rng_states[self._rank])

            torch.distributed.barrier()

    def get_max_steps_per_epoch(self):
        if self.data_loader is None:
            return 0
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            # VLM-only datasets return a plain PyTorch DataLoader, not an
            # openpi DataLoaderImpl.  Fall back to len(data_loader) directly.
            if self._is_pi05_vlm_only():
                import torch.utils.data

                dl = self.data_loader
                if isinstance(dl, torch.utils.data.DataLoader):
                    return max(1, len(dl) // self.gradient_accumulation)
                # Try unwrapping tuple if still stored as (loader, config)
                if isinstance(dl, tuple):
                    dl = dl[0]
                return max(1, len(dl) // self.gradient_accumulation)
            num_batches = len(self._openpi_pytorch_dataloader(self.data_loader))
            return max(1, num_batches // self.gradient_accumulation)
        return super().get_max_steps_per_epoch()

    @staticmethod
    def _openpi_pytorch_dataloader(openpi_dataloader: Any):
        """Unwrap OpenPI `DataLoaderImpl` to the inner PyTorch DataLoader.

        OpenPI torch path:
          DataLoaderImpl._data_loader -> TorchDataLoader
          TorchDataLoader._data_loader / .torch_loader -> torch.utils.data.DataLoader

        """
        torch_data_loader = getattr(openpi_dataloader, "_data_loader", None)
        pytorch_dl = getattr(torch_data_loader, "_data_loader", None) or getattr(
            torch_data_loader, "torch_loader", None
        )
        if pytorch_dl is None:
            raise TypeError(
                "OpenPI dataloader does not expose an inner torch DataLoader; cannot infer steps per epoch from len()."
            )
        return pytorch_dl
