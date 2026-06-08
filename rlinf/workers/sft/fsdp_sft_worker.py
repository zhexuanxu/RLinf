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

import logging
import os
from abc import abstractmethod
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory


class FSDPSftWorker(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # set the global batch size, micro batch size, eval batch size and gradient accumulation
        self.global_batch_size = self.cfg.actor.global_batch_size
        self.micro_batch_size = self.cfg.actor.micro_batch_size
        self.eval_batch_size = self.cfg.actor.get("eval_batch_size", 1)

        assert (
            self.global_batch_size % (self.micro_batch_size * self._world_size) == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"
        self.gradient_accumulation = (
            self.global_batch_size // self.micro_batch_size // self._world_size
        )

        self._init_train_dataloader()

        if self.cfg.data.get("val_data_paths") is not None:
            self.eval_data_loader, self.eval_data_config = self.build_dataloader(
                self.cfg.data.val_data_paths, eval_dataset=True
            )
        else:
            self.eval_data_loader = None

        self.global_step = 0
        # set the dataloader epoch and data iter offset
        self._data_epoch = 0
        self._data_iter_offset = 0

    def _init_train_dataloader(self):
        """Construct this rank's TRAIN dataloader and set ``self._reference_fanout``.

        Resolved BEFORE construction so the topology is correct per rank:
        ``reference_fanout`` is centralized — ONLY rank 0 builds + iterates the single
        worker-only-partition loader and scatters ``world_size*grad_accum`` micro-batches/step
        to the other ranks (`run_training`), so ranks > 0 build NOTHING for the train path
        (they receive via the fanout). ``per_rank_stream`` (default) keeps every rank building
        its own shard. Eval is handled separately (always decentralized).
        """
        from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
            PER_RANK_STREAM,
            REFERENCE_FANOUT,
            builds_train_loader,
        )

        eval_only = self.cfg.data.get("eval_only", None)
        train_disabled = (
            eval_only is not None or self.cfg.data.get("train_data_paths") is None
        )
        self._reference_fanout = (not train_disabled) and (
            str(self.cfg.data.get("loader_mode", PER_RANK_STREAM)) == REFERENCE_FANOUT
        )

        if train_disabled:
            logging.warning(
                "Eval-only mode (eval_only=%s): skipping train dataloader", eval_only
            )
            assert self.cfg.data.get("val_data_paths") is not None, (
                "val_data_paths must be set in eval-only mode"
            )
            self.data_loader = self.data_iter = self.data_config = None
            return

        mode = REFERENCE_FANOUT if self._reference_fanout else PER_RANK_STREAM
        if not builds_train_loader(mode, self._rank):
            # reference_fanout on a non-zero rank: rank 0 owns the loader and scatters.
            self.data_loader = self.data_iter = self.data_config = None
            return

        self.data_loader, self.data_config = self.build_dataloader(
            self.cfg.data.train_data_paths, eval_dataset=False
        )
        self.data_iter = iter(self.data_loader)

    def init_worker(self):
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def set_global_step(self, global_step):
        self.global_step = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0

    def run_eval(self):
        assert self.eval_data_loader is not None, "eval_data_loader is not set"

        # reset the eval_data_iter
        eval_data_iter = iter(self.eval_data_loader)

        with self.worker_timer():
            eval_step = len(eval_data_iter)
            eval_pbar = tqdm(
                initial=0,
                total=eval_step,
                desc="Evaluate Step",
                dynamic_ncols=True,
            )
            self.model.eval()
            total = 0
            correct = 0

            # get the next batch
            for _ in range(eval_step):
                batch = next(eval_data_iter)
                batch_size = batch["prompt"].size(0)
                correct += self.get_eval_model_output(batch)
                total += batch_size
                eval_pbar.update(1)

            metrics = {
                "eval_accuracy": float(correct / max(1, total)),
            }
            metrics = all_reduce_dict(metrics, op=torch.distributed.ReduceOp.AVG)
            return metrics

    def _fanout_send(self, batches, dst):
        """Send one rank's per-step micro-batches to ``dst`` (reference_fanout mode).

        Mirrors the reference ``_send_batches_to_rank`` (``send_object_list([batches])``).
        """
        torch.distributed.send_object_list([batches], dst=dst)

    def _fanout_recv(self, src):
        """Receive this rank's per-step micro-batches from ``src`` (reference_fanout mode)."""
        obj_list = [None]
        torch.distributed.recv_object_list(obj_list, src=src)
        return obj_list[0]

    def run_training(self):
        with self.worker_timer():
            self.model.train()

            metrics = {}
            # Non-invasive step-0 instrumentation (default OFF). When enabled, observe
            # the FIRST micro-batch's per-rank frame ids to prove the rank-disjoint
            # effective batch (the loader fix exercised at runtime) and pin the real
            # step-0 loss/grad-norm/lr. Read-only: it only hashes already-fetched tensors.
            _step0_instrument = self.global_step == 0 and bool(
                getattr(self.cfg.actor, "sft_step0_instrument", False)
            )
            _step0_batch = None

            # reference_fanout: fetch this rank's grad_accum micro-batches once per step
            # via the rank-0 fanout/scatter (the single loader is infinite, so no reset).
            _fanout_step_batches = None
            if self._reference_fanout:
                from rlinf.data.datasets.behavior.behavior_sft_data_loader import (
                    reference_fanout_micro_batches,
                )

                _fanout_step_batches = reference_fanout_micro_batches(
                    self.data_iter,
                    self._rank,
                    self._world_size,
                    self.gradient_accumulation,
                    self._fanout_send,
                    self._fanout_recv,
                )

            for idx in range(self.gradient_accumulation):
                # set the gradient accumulation backward_ctx
                backward_ctx = self.before_micro_batch(
                    self.model,
                    is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                )

                if self._reference_fanout:
                    batch = _fanout_step_batches[idx]
                else:
                    try:
                        batch = next(self.data_iter)
                        self._data_iter_offset += 1
                    except StopIteration:
                        self._data_epoch += 1
                        logging.info(
                            f"[INFO] data_iter exhausted, reset iterator self._data_epoch {self._data_epoch}"
                        )
                        if hasattr(self.data_loader, "sampler") and hasattr(
                            self.data_loader.sampler, "set_epoch"
                        ):
                            self.data_loader.sampler.set_epoch(self._data_epoch)
                        self.data_iter = iter(self.data_loader)
                        batch = next(self.data_iter)
                        self._data_iter_offset = 1

                if _step0_instrument and idx == 0:
                    _step0_batch = batch

                loss, step_metrics = self.get_train_model_output(batch)
                append_to_dict(metrics, step_metrics)

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

            # in one step do the optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Capture the LR used for the just-finished optimizer step (from
            # optimizer_step) before the scheduler advances to the next step's LR.
            lr_value = (
                float(lr_list[0])
                if lr_list
                else float(self.optimizer.param_groups[0]["lr"])
            )
            self.lr_scheduler.step()
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            if self.global_step > 0 and self.global_step % 1000 == 0:
                clear_memory()

            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            if _step0_instrument and _step0_batch is not None:
                self._record_step0_instrumentation(_step0_batch, train_metrics)

            return train_metrics

    def _record_step0_instrumentation(self, batch, train_metrics):
        """Observe the production step-0: per-rank frame-id disjointness (loader fix
        exercised at runtime) + the real step-0 loss/grad-norm/lr + run provenance.
        Gated OFF by default; read-only w.r.t. the loader."""
        try:
            import os

            from tools.sft_step0_instrument import capture_step0, dump_provenance

            observation, actions = batch[0], batch[1]
            out_dir = getattr(
                self.cfg.actor,
                "sft_step0_out_dir",
                "/mnt/public/xzxuan/tmp/sft_step0",
            )
            repo_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            try:
                from omegaconf import OmegaConf

                resolved_config = OmegaConf.to_container(self.cfg, resolve=True)
            except Exception:
                resolved_config = None
            dump_provenance(
                self.cfg,
                rank=self._rank,
                world_size=self._world_size,
                out_dir=out_dir,
                repo_root=repo_root,
                resolved_config=resolved_config,
                gradient_accumulation=self.gradient_accumulation,
            )
            capture_step0(
                observation,
                actions,
                rank=self._rank,
                world_size=self._world_size,
                out_dir=out_dir,
                loss=train_metrics.get("loss"),
                grad_norm=train_metrics.get("grad_norm"),
                lr=train_metrics.get("learning_rate"),
                global_batch_size=self.global_batch_size,
            )
        except Exception as exc:  # never let instrumentation break training
            logging.warning("step-0 instrumentation skipped: %s", exc)

    @abstractmethod
    def build_dataloader(self):
        raise NotImplementedError

    @abstractmethod
    def get_train_model_output(
        self, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_eval_model_output(self, batch: dict[str, Any]):
        raise NotImplementedError
