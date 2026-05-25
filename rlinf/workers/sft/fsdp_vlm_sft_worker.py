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

import json
import logging
import os
import re
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.agents.dualsystem.prompts import parse_vlm_output
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import generate_with_kv_cache
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker

# Qwen3-VL </think> token ID for token-level thinking strip.
_THINK_END_TOKEN_ID = 151668


class FSDPVlmSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._is_qwen3 = SupportedModel(cfg.actor.model.model_type) in [
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]

    def _save_data_state(self, save_path: str):
        state = {
            "data_epoch": self._data_epoch,
            "data_iter_offset": self._data_iter_offset,
        }
        with open(os.path.join(save_path, "data_state.json"), "w") as f:
            json.dump(state, f)

    def save_checkpoint(self, save_path: str, step: int = 0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            self._save_data_state(save_path)

    def _load_data_state(self, load_path: str):
        path = os.path.join(load_path, "data_state.json")
        if not os.path.exists(path):
            return
        if self.data_loader is None:
            return
        with open(path, "r") as f:
            state = json.load(f)
        self._data_epoch = int(state.get("data_epoch", 0))
        self._data_iter_offset = int(state.get("data_iter_offset", 0))

        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            self.data_loader.sampler.set_epoch(self._data_epoch)

        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)

    def load_checkpoint(self, load_path: str):
        super().load_checkpoint(load_path)
        self._load_data_state(load_path)

    def build_tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.actor.model.model_path,
        )
        # set the padding side to left for the tokenizer, QWEN 2.5 VL just use left padding
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.QWEN2_5_VL_SFT,
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            # vlm sft before load dataloader should build the tokenizer
            if not hasattr(self, "tokenizer"):
                self.tokenizer = self.build_tokenizer()

            dataset_name = self.cfg.data.get("dataset_name", "robo2vlmsft")

            # --- Behavior VLM SFT (skill or agentic) ---
            if dataset_name in ("behavior_skill_sft", "behavior_agentic_sft"):
                from transformers import AutoProcessor

                from rlinf.data.datasets.behavior_photo_vlm import (
                    create_behavior_photo_data_loader_qwen as create_behavior_vlm_data_loader_qwen,
                )

                data_dir = data_paths[0] if isinstance(data_paths, list) else data_paths
                processor = AutoProcessor.from_pretrained(
                    self.cfg.actor.model.model_path
                )
                batch_size = (
                    self.micro_batch_size
                    if not eval_dataset
                    else self.cfg.actor.get("eval_batch_size", 1)
                )

                # In eval-only mode, eval_split controls which data split to evaluate
                eval_only = self.cfg.data.get("eval_only", None)
                eval_split = eval_only if eval_dataset and eval_only else None

                task_names_cfg = self.cfg.data.get("task_names", ["turning_on_radio"])
                if task_names_cfg == "all":
                    from rlinf.data.datasets.behavior_photo_vlm import TASK_NAMES_TO_INDICES
                    task_names_cfg = list(TASK_NAMES_TO_INDICES.keys())

                # Convert OmegaConf containers to plain Python for pickling
                task_subtasks_raw = self.cfg.data.get("task_subtasks", None)
                if task_subtasks_raw is not None:
                    from omegaconf import OmegaConf
                    task_subtasks_raw = OmegaConf.to_container(task_subtasks_raw, resolve=True)
                skill_library_raw = self.cfg.data.get("skill_library", None)
                if skill_library_raw is not None:
                    from omegaconf import OmegaConf
                    skill_library_raw = OmegaConf.to_container(skill_library_raw, resolve=True)

                data_loader = create_behavior_vlm_data_loader_qwen(
                    data_root=data_dir,
                    tasks=task_names_cfg,
                    processor=processor,
                    tokenizer=self.tokenizer,
                    eval_mode=eval_dataset,
                    eval_ratio=self.cfg.data.get("eval_ratio", 0.1),
                    eval_split=eval_split,
                    batch_size=batch_size,
                    num_workers=self.cfg.data.get("num_workers", 4),
                    seed=self.cfg.data.get("seed", 42),
                    enable_reasoning=self.cfg.data.get("enable_reasoning", False),
                    enable_memory=self.cfg.data.get("enable_memory", False),
                    simple_skill=self.cfg.data.get("simple_skill", True),
                    sft_data_dir=self.cfg.data.get("sft_data_dir", None),
                    task_subtasks=task_subtasks_raw,
                    skill_library=skill_library_raw,
                    aug_option=str(self.cfg.data.get("skill_subtask_aug", "none")),
                )
                data_config = {
                    "dataset_name": dataset_name,
                    "num_samples": len(data_loader.dataset),
                }
                return data_loader, data_config

            # --- Standard VLM SFT: uses VLMDatasetRegistry ---
            from torch.utils.data import DataLoader, DistributedSampler

            from rlinf.data.datasets import sft_collate_fn
            from rlinf.data.datasets.vlm import VLMDatasetRegistry

            train_dataset = VLMDatasetRegistry.create(
                dataset_name,
                data_paths=data_paths,
                config=self.cfg,
                tokenizer=self.tokenizer,
                eval_dataset=eval_dataset,
            )

            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=self.cfg.data.get("shuffle", True),
                    seed=self.cfg.data.get("seed", 42),
                    drop_last=True,
                )
            else:
                sampler = None

            batch_size = (
                self.micro_batch_size
                if not eval_dataset
                else self.cfg.actor.get("eval_batch_size", 1)
            )
            data_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=sampler,
                shuffle=(sampler is None),
                num_workers=self.cfg.data.get("num_workers", 4),
                drop_last=True,
                collate_fn=sft_collate_fn,
            )
            logging.info(
                f"Build data loader from {data_paths} with {len(train_dataset)} samples"
            )

            data_config = {
                "dataset_name": dataset_name,
                "num_samples": len(train_dataset),
            }

            return data_loader, data_config

        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _normalize_text(self, s: str) -> str:
        return " ".join(str(s).strip().lower().split())

    def _strip_thinking_tokens(self, token_ids: torch.Tensor) -> str:
        """Strip ``<think>...</think>`` from Qwen3-VL output at token level.

        Uses the same logic as ``Qwen3_VLPolicy._strip_thinking``.
        """
        ids_list = token_ids.tolist()
        try:
            idx = len(ids_list) - 1 - ids_list[::-1].index(_THINK_END_TOKEN_ID)
            content_ids = ids_list[idx + 1:]
        except ValueError:
            content_ids = ids_list

        text = self.tokenizer.decode(
            content_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return text

    def run_eval(self):
        # Reset per-eval-pass print counter on rank 0.
        self._eval_print_remaining = (
            int(self.cfg.runner.get("print_eval_samples", 0) or 0)
            if self._rank == 0
            else 0
        )

        # Setup eval logging — every rank writes its own tmp file + images.
        log_root = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
        )
        step = getattr(self, "global_step", 0)
        eval_dir = os.path.join(log_root, f"eval_step_{step}")
        self._eval_image_dir = os.path.join(eval_dir, "images")
        os.makedirs(self._eval_image_dir, exist_ok=True)
        self._eval_log_tmp = os.path.join(eval_dir, f"eval_rank_{self._rank}.jsonl")
        self._eval_log_file = open(self._eval_log_tmp, "w")
        self._eval_sample_idx = 0

        metrics = super().run_eval()

        self._eval_log_file.close()

        # Barrier so all ranks finish writing before rank 0 merges.
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._rank == 0:
            # Merge all rank files into one sorted JSONL.
            import glob
            merged_path = os.path.join(eval_dir, "eval_results.jsonl")
            all_records = []
            for rank_file in sorted(glob.glob(os.path.join(eval_dir, "eval_rank_*.jsonl"))):
                with open(rank_file) as f:
                    for line in f:
                        all_records.append(json.loads(line))
                os.remove(rank_file)
            all_records.sort(key=lambda r: r["sample_idx"])
            with open(merged_path, "w") as f:
                for r in all_records:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

            acc = metrics.get("eval_accuracy", 0.0)
            logging.info(
                f"[EVAL] accuracy = {acc:.4f} ({acc * 100:.1f}%), "
                f"{len(all_records)} samples saved to {merged_path}"
            )

        return metrics

    def get_eval_model_output(self, batch: dict[str, Any]):
        correct = 0
        input_ids = batch["prompt"].to(self.device)
        answers = batch["answer"]
        attention_mask = batch["attention_mask"].to(self.device)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = v.to(device=self.device)

        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        eval_max_new_tokens = self.cfg.data.get("eval_max_new_tokens", 128)

        with torch.no_grad():
            generate_ids = generate_with_kv_cache(
                model=self.model,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                amp_context=self.amp_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
                max_new_tokens=eval_max_new_tokens,
            )

        for i in range(len(answers)):
            new_token_ids = generate_ids[i, input_ids.shape[1]:]

            # Full decoded text (preserving <think> for logging).
            raw_decoded = self.tokenizer.decode(
                new_token_ids.tolist(), skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            # Strip thinking for subtask extraction (aligned with VLM inference).
            if self._is_qwen3:
                content_text = self._strip_thinking_tokens(new_token_ids)
            else:
                content_text = raw_decoded

            _, _, pred_text = parse_vlm_output(content_text)
            gold_text = answers[i]

            is_correct = self._normalize_text(pred_text) == self._normalize_text(
                gold_text
            )
            if is_correct:
                correct += 1

            # Log to console (first N samples)
            if self._rank == 0 and getattr(self, "_eval_print_remaining", 0) > 0:
                verdict = "CORRECT" if is_correct else "WRONG"
                print(
                    f"\n==== EVAL ({verdict}) ====\n"
                    f"Gold:  {gold_text!r}\n"
                    f"Pred:  {pred_text!r}\n"
                    f"Raw:   {raw_decoded!r}\n"
                    "================================",
                    flush=True,
                )
                self._eval_print_remaining -= 1

            # Log to JSONL + save image (all ranks)
            if hasattr(self, "_eval_log_file"):
                from PIL import Image as PILImage

                img_path = ""
                image_data = batch.get("image_data")
                if image_data and i < len(image_data) and image_data[i]:
                    pil_img = image_data[i][0]
                    if isinstance(pil_img, PILImage.Image):
                        img_path = os.path.join(
                            self._eval_image_dir,
                            f"r{self._rank}_s{self._eval_sample_idx:05d}.jpg",
                        )
                        pil_img.save(img_path, "JPEG")

                output_token_len = int((new_token_ids != pad_token_id).sum().item())
                record = {
                    "sample_idx": self._eval_sample_idx,
                    "rank": self._rank,
                    "prompt_text": batch["prompt_text"][i],
                    "gold_answer": gold_text,
                    "pred_raw": raw_decoded,
                    "pred_content": content_text,
                    "pred_subtask": pred_text,
                    "is_correct": is_correct,
                    "input_token_len": int(input_ids.shape[1]),
                    "output_token_len": output_token_len,
                    "image_path": img_path,
                }
                self._eval_log_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                self._eval_sample_idx += 1

        return correct

    def get_train_model_output(self, batch: dict[str, Any]):
        # hundle the input batch
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = v.to(device=self.device)
        label_mask = batch["label_mask"].to(device=self.device, dtype=torch.bool)

        labels = input_ids.detach().clone().masked_fill(~attention_mask, -100)
        # label_mask is encode by prompt without answer, so we need to mask the labels just save the answer tokens
        labels = labels.masked_fill(label_mask, -100)

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **multi_modal_inputs,
            )

        # train model return the loss
        return outputs.loss
