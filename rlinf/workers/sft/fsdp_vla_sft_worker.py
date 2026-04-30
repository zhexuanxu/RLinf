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
import logging
import re
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        # Initialize before super().__init__() — the parent's init calls
        # self.build_dataloader(), which reads self._pi05_tokenizer.
        self._pi05_tokenizer = None
        super().__init__(cfg)
        self._full_pi05_loss = None
        self._eval_print_remaining = 0

    def _is_pi05_vlm_only(self) -> bool:
        if SupportedModel(self.cfg.actor.model.model_type) != SupportedModel.OPENPI:
            return False
        full_pi05 = getattr(self.cfg.actor.model.openpi, "full_pi05", False)
        forward_mode = getattr(self.cfg.actor.model.openpi, "forward_mode", "vla")
        return bool(full_pi05) and forward_mode == "vlm"

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            # Check for pi0.5 VLM-only mode — uses custom dataset instead of openpi data loader
            if self._is_pi05_vlm_only():
                return self._build_pi05_vlm_dataloader(data_paths, eval_dataset=eval_dataset)

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

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=getattr(self.cfg.actor, "openpi_data", None),
            )
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
            self._dreamzero_loss = None
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
        """Build dataloader for pi0.5 VLM-only SFT with Robo2VLM data.

        ``eval_dataset=True`` builds a prompt-only loader for accuracy
        evaluation (no answer concatenated; raw text returned in meta).
        """
        import torch.distributed as dist
        from torch.utils.data import DataLoader, DistributedSampler

        from rlinf.data.datasets.pi05_vlm_dataset import (
            Pi05VLMDataset,
            pi05_vlm_collate_fn,
        )
        data_dir = data_paths[0] if isinstance(data_paths, list) else data_paths
        data_cfg = self.cfg.data
        # Prefer cfg.data.max_token_len; fall back to openpi.max_token_len for
        # back-compat with configs that placed it under the model block.
        max_token_len = data_cfg.get(
            "max_token_len",
            getattr(self.cfg.actor.model.openpi, "max_token_len", 200),
        )
        num_images = getattr(self.cfg.actor.model.openpi, "num_images_in_input", 1)
        image_keys = data_cfg.get("image_keys", ["image"])
        image_key = image_keys[0] if image_keys else "image"

        dataset = Pi05VLMDataset(
            data_dir=data_dir,
            max_token_len=max_token_len,
            num_images=num_images,
            prompt_key=data_cfg.get("prompt_key", "question"),
            choice_key=data_cfg.get("choice_key", "choices"),
            answer_key=data_cfg.get("answer_key", "correct_answer"),
            image_key=image_key,
            eval_mode=eval_dataset,
        )

        if self._pi05_tokenizer is None:
            self._pi05_tokenizer = dataset.tokenizer

        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(
                dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=not eval_dataset,
                drop_last=True,
            )
        else:
            sampler = None

        batch_size = (
            self.eval_batch_size if eval_dataset else self.micro_batch_size
        )
        num_workers = data_cfg.get("num_workers", 4)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None and not eval_dataset),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=pi05_vlm_collate_fn,
        )
        split = "eval" if eval_dataset else "train"
        logging.info(
            "Built pi0.5 VLM-only %s dataloader with %d samples from %s",
            split,
            len(dataset),
            data_dir,
        )
        return data_loader, {
            "dataset": "pi05_vlm",
            "split": split,
            "num_samples": len(dataset),
        }

    def run_eval(self):
        # Reset the per-eval-pass print counter on rank 0 only.
        self._eval_print_remaining = (
            int(self.cfg.runner.get("print_eval_samples", 0) or 0)
            if self._rank == 0
            else 0
        )
        return super().run_eval()

    def get_eval_model_output(self, batch: dict[str, Any]):
        if not self._is_pi05_vlm_only():
            raise NotImplementedError(
                "eval is only implemented for pi0.5 full_pi05 + forward_mode='vlm'"
            )

        from openpi.models import model as _model

        observation, _actions, meta_list = batch

        observation.pop("token_kv_cache_mask", None)
        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )

        obs_obj = _model.Observation.from_dict(observation)

        max_new_tokens = int(
            getattr(self.cfg.actor.model.openpi, "max_language_len", 10) or 10
        )
        with torch.no_grad(), self.amp_context:
            # Route through outer forward so FSDP's lazy_init runs at the
            # top level — calling self.model.generate_language() directly
            # ends up invoking a sub-FSDP's forward and breaks subsequent
            # training with the "_is_root should not have been set" assert.
            out_tokens, *_ = self.model(
                forward_type=ForwardType.GENERATE_LANGUAGE,
                observation=obs_obj,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
            )

        eos_id = self._pi05_tokenizer.eos_id() if self._pi05_tokenizer is not None else 1
        correct = 0
        for i, meta in enumerate(meta_list):
            # 每个meta就是一个问题
            gen_ids = out_tokens[i].tolist()
            if eos_id in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(eos_id)]
            pred_text = (
                self._pi05_tokenizer.decode(gen_ids).strip()
                if self._pi05_tokenizer is not None
                else ""
            )
            pred_letter = self._extract_first_letter(pred_text)
            gold_letter = meta.get("correct_answer_letter", "")
            if pred_letter and pred_letter == gold_letter:
                correct += 1

            if self._rank == 0 and self._eval_print_remaining > 0:
                self._print_qa_sample(meta, pred_text, pred_letter)
                self._eval_print_remaining -= 1

        return correct

    @staticmethod
    def _extract_first_letter(text: str) -> str:
        if not text:
            return ""
        m = re.search(r"[A-F]", text)
        return m.group(0) if m else ""

    def _print_qa_sample(self, meta: dict, pred_text: str, pred_letter: str):
        question = meta.get("question", "")
        choices = meta.get("choices", "")
        gold = meta.get("correct_answer_letter", "")
        verdict = "CORRECT" if pred_letter and pred_letter == gold else "WRONG"
        # Single block, single print() call so it doesn't interleave across ranks.
        print(
            "\n================ EVAL SAMPLE ({} | gold={} | pred={}) ================\n"
            "Q:       {}\n"
            "Choices: {}\n"
            "Gold:    {}\n"
            "Pred:    {}    (raw: {!r})\n"
            "============================================================".format(
                verdict,
                gold or "?",
                pred_letter or "?",
                question,
                choices,
                gold,
                pred_letter,
                pred_text,
            ),
            flush=True,
        )

    def get_train_model_output(self, batch: dict[str, Any]):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA,
            SupportedModel.DREAMZERO,
        ]:
            batch_data = _pytree.tree_map(
                lambda x: (
                    torch.as_tensor(x, device=self.device).contiguous().clone()
                    if isinstance(x, torch.Tensor)
                    else x
                ),
                batch,
            )
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch_data)
            if losses_dict.get("dynamics_loss", None) is not None:
                self._dreamzero_loss = {
                    "dynamics_loss": losses_dict["dynamics_loss"],
                    "action_loss": losses_dict["action_loss"],
                }
            return losses_dict["loss"]

        # Pi0.5 VLM dataset returns a 3-tuple (obs, actions, meta); openpi
        # data loader returns a 2-tuple (obs, actions). Tolerate both.
        if len(batch) == 3:
            observation, actions, _meta = batch
        else:
            observation, actions = batch

        # Extract token_kv_cache_mask before Observation.from_dict() drops it
        # (it's computed by cot_transform but not a recognized Observation field)
        token_kv_cache_mask = None
        if isinstance(observation, dict) and "token_kv_cache_mask" in observation:
            token_kv_cache_mask = observation.pop("token_kv_cache_mask")

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        if actions is not None:
            actions = actions.to(torch.float32)
            actions = actions.to(self.device)
        if token_kv_cache_mask is not None:
            token_kv_cache_mask = torch.as_tensor(
                token_kv_cache_mask, device=self.device
            ).contiguous().clone()

        with self.amp_context:
            data = {"observation": observation, "actions": actions}
            if token_kv_cache_mask is not None:
                data["token_kv_cache_mask"] = token_kv_cache_mask
            losses = self.model(
                forward_type=ForwardType.SFT,
                data=data,
            )

        # Full pi0.5 returns a dict with loss + extra metrics
        if isinstance(losses, dict):
            self._full_pi05_loss = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in losses.items()
                if k != "loss" and v is not None
            }
            return losses["loss"]

        return losses

    def run_training(self):
        train_metrics = super().run_training()
        if (
            SupportedModel(self.cfg.actor.model.model_type)
            in [SupportedModel.DREAMZERO]
            and self._dreamzero_loss is not None
        ):
            train_metrics.update(
                {
                    "dynamics_loss": self._dreamzero_loss["dynamics_loss"],
                    "action_loss": self._dreamzero_loss["action_loss"],
                }
            )
            self._dreamzero_loss = None
        # Full pi0.5 extra metrics (language_loss, action_loss, language_token_acc)
        if self._full_pi05_loss is not None:
            train_metrics.update(self._full_pi05_loss)
            self._full_pi05_loss = None
        return train_metrics
