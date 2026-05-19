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

from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import generate_with_kv_cache
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlmSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        self._is_agentic = (
            cfg.data.get("enable_reasoning", False)
            or cfg.data.get("enable_memory", False)
            or not cfg.data.get("simple_skill", True)
        )
        super().__init__(cfg)

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

                from rlinf.models.embodiment.openpi.dataconfig.behavior_vlm_data_loader import (
                    create_behavior_vlm_data_loader_qwen,
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
                data_loader = create_behavior_vlm_data_loader_qwen(
                    data_root=data_dir,
                    tasks=["turning_on_radio"],
                    processor=processor,
                    tokenizer=self.tokenizer,
                    eval_mode=eval_dataset,
                    batch_size=batch_size,
                    num_workers=self.cfg.data.get("num_workers", 4),
                    seed=self.cfg.data.get("seed", 42),
                    system_prompt=self.cfg.data.get("system_prompt", None),
                    enable_reasoning=self.cfg.data.get("enable_reasoning", False),
                    enable_memory=self.cfg.data.get("enable_memory", False),
                    simple_skill=self.cfg.data.get("simple_skill", True),
                    sft_data_dir=self.cfg.data.get("sft_data_dir", None),
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

    def _extract_boxed(self, text: str) -> str | None:
        idx = text.rfind("boxed")
        if idx < 0:
            return None
        s = text[idx + len("boxed") :].strip()
        if not s:
            return None
        if s[0] != "{":
            return s.split("$")[0].strip() or None

        depth = 0
        out = []
        for ch in s:
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            if depth >= 1:
                out.append(ch)
        ans = "".join(out).strip()
        return ans or None

    def _extract_answer(self, text: str) -> str:
        if SupportedModel(self.cfg.actor.model.model_type) not in [
            SupportedModel.QWEN2_5_VL_SFT,
            SupportedModel.QWEN3_VL_SFT,
            SupportedModel.QWEN3_VL_MOE_SFT,
        ]:
            raise ValueError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

        if not text:
            return ""

        # 1) Get the last assistant span from common chat templates.
        patterns = [
            r"<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
            r"<\|assistant\|>\s*(.*?)(?:<\|end\|>|$)",
        ]
        body = None
        for p in patterns:
            matches = re.findall(p, text, flags=re.DOTALL | re.IGNORECASE)
            if matches:
                body = matches[-1].strip()
                break
        if body is None:
            body = text.strip()

        # 2) Remove reasoning blocks if present.
        body = re.sub(
            r"<think>.*?</think>", "", body, flags=re.DOTALL | re.IGNORECASE
        ).strip()

        # 3) Remove chat special tokens (e.g., <|im_end|>, <|endoftext|>)
        body = re.sub(r"<\|[^>]+?\|>", " ", body).strip()

        # 4) Try explicit "final answer" markers.
        marker_patterns = [
            r"(?:final answer is|the answer is)\s*[:：]?\s*(.+)$",
            r"(?:answer)\s*[:：]\s*(.+)$",
        ]
        for p in marker_patterns:
            m = re.search(p, body, flags=re.IGNORECASE | re.DOTALL)
            if m:
                cand = m.group(1).strip()
                cand = re.split(r"\n|<\|im_end\|>", cand)[0].strip()
                if cand:
                    body = cand
                    break

        # 5) Math-style boxed fallback.
        boxed = self._extract_boxed(body)
        if boxed:
            body = boxed

        # 6) Last non-empty line fallback.
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines:
            body = lines[-1]

        # final cleanup
        body = body.strip().strip("`").strip()
        body = body.rstrip(".").rstrip("/")
        return body

    def run_eval(self):
        # Reset per-eval-pass print counter on rank 0.
        self._eval_print_remaining = (
            int(self.cfg.runner.get("print_eval_samples", 0) or 0)
            if self._rank == 0
            else 0
        )

        if self._is_agentic:
            self._eval_loss_sum = 0.0
            self._eval_loss_count = 0
            result = super().run_eval()
            # Add eval_loss metric
            if self._eval_loss_count > 0:
                result["eval_loss"] = self._eval_loss_sum / self._eval_loss_count
            else:
                result["eval_loss"] = 0.0
            return result

        return super().run_eval()

    def _compute_eval_loss(self, batch: dict[str, Any]) -> float:
        """Compute eval loss with the same masking as training (no grad, model.eval)."""
        input_ids = batch["prompt"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device, dtype=torch.bool)
        multi_modal_inputs = {
            k: v.to(device=self.device) for k, v in batch["multi_modal_inputs"].items()
        }
        label_mask = batch["label_mask"].to(device=self.device, dtype=torch.bool)

        labels = input_ids.detach().clone().masked_fill(~attention_mask, -100)
        labels = labels.masked_fill(label_mask, -100)

        with torch.no_grad(), self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **multi_modal_inputs,
            )
        return outputs.loss.item()

    def get_eval_model_output(self, batch: dict[str, Any]):
        correct = 0
        input_ids = batch["prompt"].to(self.device)
        answers = batch["answer"]
        attention_mask = batch["attention_mask"].to(self.device)
        multi_modal_inputs = batch["multi_modal_inputs"]
        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = v.to(device=self.device)

        if self._is_agentic:
            # 1) Compute eval loss (same masking as training)
            # input_ids contains full sequence (prompt + answer) because
            # agentic transform always uses eval_mode=False
            eval_loss = self._compute_eval_loss(batch)
            self._eval_loss_sum += eval_loss
            self._eval_loss_count += 1

            # 2) Generate model output for logging
            # Extract prompt-only tokens using label_mask.
            # After left-padding + collation, layout is:
            #   [False(pad), True(prompt), False(answer)]
            # Find the position right after the last True = start of answer.
            label_mask = batch["label_mask"].to(self.device, dtype=torch.bool)
            # Flip and find first True from the right → gives answer start position
            prompt_end_positions = (
                label_mask.shape[1] - label_mask.flip(1).long().argmax(1)
            )  # [B], each is the position right after the last True
            max_prompt_end = prompt_end_positions.max().item()
            gen_input_ids = input_ids[:, :max_prompt_end]
            gen_attention_mask = attention_mask[:, :max_prompt_end]
            gen_multi_modal = multi_modal_inputs

            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else (eos_token_id if eos_token_id is not None else 0)
            )
            with torch.no_grad():
                generate_ids = generate_with_kv_cache(
                    model=self.model,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                    amp_context=self.amp_context,
                    input_ids=gen_input_ids,
                    attention_mask=gen_attention_mask,
                    multi_modal_inputs=gen_multi_modal,
                )

            # 3) Print samples
            for i in range(len(answers)):
                new_token_ids = generate_ids[i, gen_input_ids.shape[1]:]
                full_pred_text = self.tokenizer.decode(
                    new_token_ids.tolist(), skip_special_tokens=False
                )
                gold_text = answers[i]
                prompt_text = batch.get("prompt_text", [""] * len(answers))[i]

                if self._rank == 0 and getattr(self, "_eval_print_remaining", 0) > 0:
                    print(
                        f"\n==== EVAL AGENTIC (loss={eval_loss:.4f}) ====\n"
                        f"Input:\n{prompt_text}\n"
                        f"---\n"
                        f"Gold:\n{gold_text[:500]}\n"
                        f"---\n"
                        f"Pred:\n{full_pred_text[:500]}\n"
                        "======================",
                        flush=True,
                    )
                    self._eval_print_remaining -= 1

            return 0  # no accuracy for agentic

        # --- Original skill-based eval ---
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else (eos_token_id if eos_token_id is not None else 0)
        )

        with torch.no_grad():
            generate_ids = generate_with_kv_cache(
                model=self.model,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
                amp_context=self.amp_context,
                input_ids=input_ids,
                attention_mask=attention_mask,
                multi_modal_inputs=multi_modal_inputs,
            )

        for i in range(len(answers)):
            new_token_ids = generate_ids[i, input_ids.shape[1]:]
            full_pred_text = self.tokenizer.decode(
                new_token_ids.tolist(), skip_special_tokens=False
            )

            pred_text = self._extract_answer(full_pred_text)
            gold_text = answers[i]

            is_correct = self._normalize_text(pred_text) == self._normalize_text(gold_text)
            if is_correct:
                correct += 1

            if self._rank == 0 and getattr(self, "_eval_print_remaining", 0) > 0:
                verdict = "CORRECT" if is_correct else "WRONG"
                print(
                    f"\n==== EVAL SKILL ({verdict}) ====\n"
                    f"Gold:  {gold_text!r}\n"
                    f"Pred:  {pred_text!r}\n"
                    f"Raw:   {full_pred_text[:100]!r}\n"
                    "================================",
                    flush=True,
                )
                self._eval_print_remaining -= 1

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
