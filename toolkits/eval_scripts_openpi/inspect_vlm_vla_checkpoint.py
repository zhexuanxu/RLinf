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

"""Inspect a pi0.5 vlm_vla checkpoint: subtask generation + action denoising.

Loads a checkpoint (the base model or an SFT ``full_weights.pt``), builds a
batched eval observation (batch > 1, mixed prompt lengths) from real BEHAVIOR
frames, then walks the reasoning-then-acting path stage by stage, printing and
asserting at each one:

1. tokenizer masks of the generation prefix (the AC table: prefix
   bidirectional/no-loss/in-KV; padding inert);
2. right alignment (every row's last valid token in the last column → one
   shared static-cache write column per step);
3. greedy generation with per-row EOS: decoded text, EOS step, termination —
   rows exhausting the budget are reported as NON-terminated;
4. per-row cache-column validity: each row's EOS and post-EOS columns must be
   invisible to the denoise attention;
5. the denoised action tensor shapes.

With ``--expect-subtasks`` the script additionally checks that every
terminated row's text matches one of the configured subtask labels (use for a
TRAINED checkpoint); without it, a base checkpoint is expected to FAIL that
check — demonstrating the inspection is not vacuous.

Run with the project interpreter, e.g.::

    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/eval_scripts_openpi/inspect_vlm_vla_checkpoint.py \
        --checkpoint /path/to/checkpoints/global_step_60 \
        --expect-subtasks \
        --report /mnt/public/xzxuan/tmp/vlm_vla_inspection.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SUBTASKS = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help=(
            "Checkpoint to inspect: a global_step_N directory (containing "
            "actor/model_state_dict/full_weights.pt), a full_weights.pt file, "
            "or omitted to inspect the BASE model."
        ),
    )
    parser.add_argument(
        "--expect-subtasks",
        action="store_true",
        help="Require terminated rows to emit one of the known subtask labels.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument(
        "--data-root", default="/mnt/public/xzxuan/data/2025-challenge-demos"
    )
    parser.add_argument(
        "--base-model", default="/mnt/public/xzxuan/models/pi05_base_pytorch_new"
    )
    parser.add_argument(
        "--tokenizer",
        default="/mnt/public/xzxuan/models/paligemma_tokenizer/paligemma_tokenizer.model",
    )
    parser.add_argument(
        "--assets-dir",
        default=(
            "/mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/"
            "train/pi05_b1k-task0000_sft_pytorch_mixed"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--report", default="/mnt/public/xzxuan/tmp/vlm_vla_inspection.txt"
    )
    return parser.parse_args()


def build_model(args):
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch import get_model

    cfg = OmegaConf.create(
        {
            "model_path": args.base_model,
            "precision": "bf16",
            "num_action_chunks": 32,
            "num_steps": 10,
            "action_dim": 23,
            "openpi": {
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "assets_dir": args.assets_dir,
                "asset_id": "behavior-1k/2025-challenge-demos",
                "paligemma_tokenizer": args.tokenizer,
                "env": "behavior",
                "mode": "vlm_vla",
                "max_new_tokens": args.max_new_tokens,
            },
        }
    )
    wrapper = get_model(cfg)
    if args.checkpoint:
        path = Path(args.checkpoint)
        if path.is_dir():
            candidates = sorted(path.glob("**/full_weights.pt"))
            if not candidates:
                raise FileNotFoundError(f"no full_weights.pt under {path}")
            path = candidates[0]
        state = torch.load(path, map_location="cpu", weights_only=True)
        if any(key.startswith("model.") for key in state):
            state = {
                key[len("model.") :]: value
                for key, value in state.items()
                if key.startswith("model.")
            }
        missing, unexpected = wrapper.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"checkpoint mismatch: missing={missing[:5]} "
                f"unexpected={unexpected[:5]}"
            )
        print(f"loaded SFT weights from {path}")
    else:
        print("inspecting the BASE checkpoint (no SFT weights)")
    return wrapper.to(args.device).eval()


def build_eval_batch(args, log):
    """Real frames -> generation-prefix Observation with mixed prompt lengths."""
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
        BehaviorSftTransform,
    )
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        BehaviorSftDataset,
    )
    from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import load_norm_stats
    from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
        PaligemmaTokenizer,
    )

    dataset = BehaviorSftDataset(
        repo_id="behavior-1k/2025-challenge-demos",
        root=args.data_root,
        episodes=[2],
        tasks=["turning_on_radio"],
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(32)]},
        chunk_streaming_using_keyframe=True,
        shuffle=False,
        seed=42,
        fine_grained_level=1,
        subtask_labels=dict(enumerate(SUBTASKS)),
        enable_gap=True,
    )
    transform = BehaviorSftTransform(
        norm_stats=load_norm_stats(args.assets_dir, "behavior-1k/2025-challenge-demos"),
        tokenizer_path=args.tokenizer,
        action_dim=32,
        max_token_len=200,
        vlm_vla=True,
    )
    # Spread the probes across the episode so different ground-truth subtasks
    # appear in one batch (the stream starts at frame 0; skip between picks).
    items, frames = [], []
    stride = 400
    for index in range(args.batch_size):
        raw = dataset[0]
        frames.append(round(raw["timestamp"].item() * dataset.fps))
        items.append(transform(raw))
        for _ in range(stride - 1):
            dataset.current_streaming_frame_idx += 1
            for key in dataset.obs_loaders:
                next(dataset.obs_loaders[key])[0]

    # Mixed prompt lengths across rows.
    prompts = [
        "Turn on the radio.",
        "Turn on the radio receiver that is on the table in the living room.",
        "Turn on the radio please.",
        "Turn the radio on.",
    ]
    prompts = (prompts * ((args.batch_size + 3) // 4))[: args.batch_size]

    tokenizer = PaligemmaTokenizer(args.tokenizer, max_len=200)
    tokens, masks, ars, losses, kvs = [], [], [], [], []
    for item, prompt in zip(items, prompts):
        state23 = np.asarray(item["state"])[:23]
        t, m, a, lo, kv = tokenizer.tokenize_with_subtask(prompt, state23, None)
        tokens.append(t), masks.append(m), ars.append(a)
        losses.append(lo), kvs.append(kv)

    device = args.device
    data = {
        "image": {
            k: torch.from_numpy(
                np.stack([np.asarray(it["image"][k]) for it in items])
            ).to(device)
            for k in IMAGE_KEYS
        },
        "image_mask": {
            k: torch.from_numpy(
                np.stack([np.asarray(it["image_mask"][k]) for it in items])
            ).to(device)
            for k in IMAGE_KEYS
        },
        "state": torch.from_numpy(
            np.stack([np.asarray(it["state"], dtype=np.float32) for it in items])
        ).to(device),
        "tokenized_prompt": torch.from_numpy(np.stack(tokens)).long().to(device),
        "tokenized_prompt_mask": torch.from_numpy(np.stack(masks).astype(bool)).to(
            device
        ),
        "token_ar_mask": torch.from_numpy(np.stack(ars).astype(bool)).to(device),
        "token_loss_mask": torch.from_numpy(np.stack(losses).astype(bool)).to(device),
        "token_kv_cache_mask": torch.from_numpy(np.stack(kvs).astype(bool)).to(device),
    }
    observation = Observation.from_dict(data)

    log("[stage 1] generation-prefix tokenizer masks")
    valid = np.stack(masks)
    log(f"  prompt lengths (valid tokens): {valid.sum(axis=1).tolist()}")
    log(f"  probe frames: {frames}")
    assert not np.stack(losses).any(), "generation prefix must carry no CE loss"
    assert not np.stack(ars).any(), "generation prefix must be bidirectional"
    for row in range(args.batch_size):
        n = int(valid[row].sum())
        assert kvs[row][:n].all() and not kvs[row][n:].any()
    log("  prefix: ar=False, loss=False, kv=True for valid tokens; padding inert")
    return observation, tokenizer, prompts


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    wrapper = build_model(args)
    model = wrapper.model
    observation, tokenizer, prompts = build_eval_batch(args, log)
    batch_size = observation.state.shape[0]

    with torch.no_grad():
        generation = model.generate_language(
            observation, eos_token_id=tokenizer.eos_token_id
        )

    cache_valid = generation["cache_valid_mask"]
    tokens = generation["tokens"]
    eos_steps = generation["eos_steps"]
    terminated = generation["terminated"]
    max_new = model.max_new_tokens
    prefill_size = cache_valid.shape[1] - max_new

    log("[stage 2] right alignment + static cache geometry")
    log(
        f"  cache columns: {tuple(cache_valid.shape)} "
        f"(prefill {prefill_size} + generation budget {max_new})"
    )
    prefix_valid = cache_valid[:, :prefill_size]
    for row in range(batch_size):
        row_valid = prefix_valid[row]
        n = int(row_valid.sum())
        assert row_valid[-n:].all() and not row_valid[: prefill_size - n].any(), (
            f"row {row}: prefix is not right-aligned"
        )
    log("  every row's valid prefix ends at the shared last prefill column")

    log("[stage 3] generation results (greedy, per-row EOS)")
    for row in range(batch_size):
        step = int(eos_steps[row])
        text = tokenizer.decode(tokens[row, :step].tolist())
        status = (
            f"EOS at step {step}"
            if bool(terminated[row])
            else f"NON-TERMINATED (budget {max_new} exhausted)"
        )
        known = text.rstrip(".") in SUBTASKS
        log(f"  row {row}: {status}; known-subtask={known}; prompt={prompts[row]!r}")
        log(f"    text: {text!r}")
        log(f"    token ids: {tokens[row].tolist()}")

    log("[stage 4] per-row cache-column validity (EOS exclusion)")
    generated_valid = cache_valid[:, prefill_size:]
    for row in range(batch_size):
        step = int(eos_steps[row])
        if bool(terminated[row]):
            assert not generated_valid[row, step:].any(), (
                f"row {row}: EOS/post-EOS columns visible to the action expert"
            )
            assert generated_valid[row, :step].all(), (
                f"row {row}: pre-EOS generated tokens must stay visible"
            )
            log(
                f"  row {row}: columns [{prefill_size + step}, {cache_valid.shape[1]}) "
                "invalid (EOS onwards) — verified"
            )
        else:
            assert generated_valid[row].all(), (
                f"row {row}: non-terminated rows keep every generated column"
            )
            log(f"  row {row}: non-terminated; all {max_new} columns valid")

    log("[stage 5] action denoising against the generation cache")
    with torch.no_grad():
        actions = model._denoise_actions(
            generation["observation"],
            generation["kv_cache"],
            generation["cache_valid_mask"],
            num_steps=wrapper.num_steps,
            noise=None,
            rng=None,
        )
    log(f"  denoised actions: {tuple(actions.shape)} dtype={actions.dtype}")
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
    assert torch.isfinite(actions).all()

    if args.expect_subtasks:
        failures = []
        for row in range(batch_size):
            step = int(eos_steps[row])
            text = tokenizer.decode(tokens[row, :step].tolist()).rstrip(".")
            if not bool(terminated[row]) or text not in SUBTASKS:
                failures.append((row, bool(terminated[row]), text))
        assert not failures, (
            f"trained checkpoint emitted non-subtask outputs: {failures}"
        )
        log("[stage 6] all rows terminated with a known subtask label — PASSED")
    else:
        log("[stage 6] --expect-subtasks not set (base-model inspection mode)")

    log("INSPECTION COMPLETE")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
