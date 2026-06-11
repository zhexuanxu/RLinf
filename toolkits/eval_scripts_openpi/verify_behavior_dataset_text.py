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

"""Verify the BEHAVIOR streaming dataset's per-frame text contract on real data.

Streams real frames from the 2025-challenge-demos dataset and asserts, frame by
frame, that the text attached by ``BehaviorSftDataset`` matches the deterministic
skill-segment rules:

* ``fine_grained_level=0``: every item carries exactly one text — the main-task
  prompt — and no subtask response.
* ``fine_grained_level=1``: every item carries the main-task prompt plus the
  subtask response resolved from the episode's ``skill_annotation``;
  ``enable_gap=True`` assigns gap frames to the next skill, ``enable_gap=False``
  yields no gap frames at all; trailing/out-of-valid frames never appear.

Also checks the rejected-configuration matrix (removed kwargs, bad
``skill_list``, missing/short subtask labels) and the collated
``(Observation, actions)`` contract including the per-token mask table.

Run with the project interpreter, e.g.::

    /mnt/public/xzxuan/.venv_pi/bin/python \
        toolkits/eval_scripts_openpi/verify_behavior_dataset_text.py \
        --report /mnt/public/xzxuan/tmp/behavior_dataset_text_report.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

SUBTASKS = [
    "move to radio",
    "pick up radio from coffee table",
    "press radio",
    "place radio on coffee table",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default="/mnt/public/xzxuan/data/2025-challenge-demos"
    )
    parser.add_argument("--repo-id", default="behavior-1k/2025-challenge-demos")
    parser.add_argument("--task", default="turning_on_radio")
    parser.add_argument(
        "--episode-position",
        type=int,
        default=2,
        help="Positional index of the probe episode (2 -> episode_00000030).",
    )
    parser.add_argument(
        "--max-stream-frames",
        type=int,
        default=1550,
        help="How far into the probe episode the streamed check walks.",
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
    parser.add_argument(
        "--report", default="/mnt/public/xzxuan/tmp/behavior_dataset_text_report.txt"
    )
    return parser.parse_args()


def build_dataset(args, *, level: int, enable_gap: bool, labels=None):
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        BehaviorSftDataset,
    )

    return BehaviorSftDataset(
        repo_id=args.repo_id,
        root=args.data_root,
        episodes=[args.episode_position],
        tolerance_s=1e-4,
        tasks=[args.task],
        modalities=["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(32)]},
        chunk_streaming_using_keyframe=True,
        shuffle=False,
        seed=42,
        fine_grained_level=level,
        subtask_labels=labels,
        enable_gap=enable_gap,
    )


def expected_subtask(segments, frame: int, enable_gap: bool):
    from rlinf.data.datasets.openpi_pytorch.behavior.skill_segments import (
        resolve_frame_subtask,
    )

    index = resolve_frame_subtask(segments, frame, enable_gap)
    return None if index is None else SUBTASKS[index]


def stream_and_check(dataset, *, level, enable_gap, max_frame, segments, log):
    """Walk the stream and assert each yielded frame's text; return stats."""
    yielded: dict[int, str | None] = {}
    last_frame = -1
    main_task = None
    while last_frame < max_frame:
        item = dataset[0]
        frame = round(item["timestamp"].item() * dataset.fps)
        if frame < last_frame:
            break  # The stream wrapped around the episode.
        last_frame = frame
        main_task = item["task"]
        response = item.get("response")
        if level == 0:
            assert "response" not in item, f"level 0 yielded a response at {frame}"
        else:
            expected = expected_subtask(segments, frame, enable_gap)
            assert expected is not None, (
                f"frame {frame} was yielded but the resolver marks it unused"
            )
            assert response == expected, (
                f"frame {frame}: yielded {response!r}, expected {expected!r}"
            )
        yielded[frame] = response

    if level == 1:
        for frame in range(0, max_frame + 1):
            expected = expected_subtask(segments, frame, enable_gap)
            if expected is None:
                assert frame not in yielded, (
                    f"frame {frame} must be skipped (gap/invalid) but was yielded"
                )
            else:
                assert frame in yielded, (
                    f"frame {frame} (label {expected!r}) is missing from the stream"
                )

    log(
        f"  streamed {len(yielded)} frames up to {last_frame}; "
        f"main task = {main_task!r}"
    )
    if level == 1:
        boundaries = {}
        for frame in sorted(yielded):
            label = yielded[frame]
            if boundaries.get(label) is None:
                boundaries[label] = frame
        for label, first in boundaries.items():
            log(f"    first frame of {label!r}: {first}")
    return yielded


def check_rejections(args, log):
    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
        BehaviorSftDataset,
    )

    def expect(exc, message, **kwargs):
        defaults = {
            "repo_id": args.repo_id,
            "root": args.data_root,
            "episodes": [args.episode_position],
            "tasks": [args.task],
            "modalities": ["rgb"],
            "local_only": True,
            "chunk_streaming_using_keyframe": True,
            "shuffle": False,
        }
        defaults.update(kwargs)
        try:
            BehaviorSftDataset(**defaults)
        except exc as err:
            log(f"  rejected as expected ({message}): {err}")
            return
        raise AssertionError(f"configuration was NOT rejected: {message}")

    expect(TypeError, "use_skill kwarg removed", use_skill=True)
    expect(TypeError, "allow_left kwarg removed", allow_left=10)
    expect(TypeError, "allow_right kwarg removed", allow_right=10)
    expect(ValueError, "weighted skill_list rejected", skill_list=["press radio:2.0"])
    expect(ValueError, "level 1 without labels", fine_grained_level=1)
    expect(
        ValueError,
        "level 1 with too few labels (skill_idx out of bounds)",
        fine_grained_level=1,
        subtask_labels={0: SUBTASKS[0], 1: SUBTASKS[1]},
    )
    expect(
        ValueError,
        "bad fine_grained_level",
        fine_grained_level=3,
    )


def check_collation(args, dataset, log):
    """Transform + collate a few level-1 items and assert the mask table."""
    import numpy as np

    from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_data_loader import (
        BehaviorSftTransform,
        collate_behavior_sft_items,
    )
    from rlinf.models.embodiment.openpi_pytorch.utils.normalize import load_norm_stats

    norm_stats = load_norm_stats(args.assets_dir, args.repo_id)
    transform = BehaviorSftTransform(
        norm_stats=norm_stats,
        tokenizer_path=args.tokenizer,
        action_dim=32,
        max_token_len=200,
        vlm_vla=True,
    )
    items = [transform(dataset[0]) for _ in range(4)]
    observation, actions = collate_behavior_sft_items(items)

    assert observation.token_ar_mask.shape == (4, 200)
    assert observation.token_loss_mask.shape == (4, 200)
    assert observation.token_kv_cache_mask.shape == (4, 200)
    assert observation.token_ar_mask.dtype == observation.token_loss_mask.dtype
    assert actions.shape == (4, 32, 32)

    tokens = observation.tokenized_prompt[0].numpy()
    valid = observation.tokenized_prompt_mask[0].numpy()
    ar = observation.token_ar_mask[0].numpy()
    loss = observation.token_loss_mask[0].numpy()
    kv = observation.token_kv_cache_mask[0].numpy()
    n_valid = int(valid.sum())
    response_positions = np.flatnonzero(loss)
    prefix_len = int(response_positions[0])
    eos_position = int(response_positions[-1])

    assert eos_position == n_valid - 1, "EOS must be the last valid token"
    assert not ar[:prefix_len].any(), "prefix must be bidirectional (ar False)"
    assert not loss[:prefix_len].any(), "prefix must carry no CE loss"
    assert kv[:prefix_len].all(), "prefix must be visible to the action expert"
    assert ar[prefix_len : eos_position + 1].all(), "response+EOS must be causal"
    assert loss[prefix_len : eos_position + 1].all(), "response+EOS must have loss"
    assert kv[prefix_len:eos_position].all(), "response must be in the KV cache"
    assert not kv[eos_position], "EOS must NOT be visible to the action expert"
    assert not valid[n_valid:].any() and not ar[n_valid:].any()
    assert not loss[n_valid:].any() and not kv[n_valid:].any()

    from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import (
        PaligemmaTokenizer,
    )

    tok = PaligemmaTokenizer(args.tokenizer, max_len=200)
    decoded_prefix = tok.decode(tokens[:prefix_len])
    decoded_response = tok.decode(tokens[prefix_len:eos_position])
    log(f"  prefix tail: ...{decoded_prefix[-40:]!r}")
    log(f"  response: {decoded_response!r} + EOS(id={tok.eos_token_id})")
    assert decoded_prefix.rstrip().endswith("Subtask:"), (
        "prefix must end with the 'Subtask:' generation cue"
    )
    assert decoded_response.rstrip(".") in SUBTASKS
    log("  collation contract + mask table verified on real data")


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def log(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    from rlinf.data.datasets.openpi_pytorch.behavior.skill_segments import (
        build_skill_segments,
    )

    annotation_path = (
        Path(args.data_root) / "annotations" / "task-0000" / "episode_00000030.json"
    )
    segments = build_skill_segments(
        json.loads(annotation_path.read_text()), len(SUBTASKS), episode_id=30
    )
    log(f"probe episode windows: starts={segments.starts} ends={segments.ends}")
    log(f"valid_duration: [{segments.valid_start}, {segments.valid_end})")

    labels = dict(enumerate(SUBTASKS))

    log("[1/5] fine_grained_level=0 (main task only)")
    dataset = build_dataset(args, level=0, enable_gap=True)
    stream_and_check(
        dataset, level=0, enable_gap=True, max_frame=120, segments=segments, log=log
    )

    log("[2/5] fine_grained_level=1, enable_gap=True (gaps -> next skill)")
    dataset = build_dataset(args, level=1, enable_gap=True, labels=labels)
    stream_and_check(
        dataset,
        level=1,
        enable_gap=True,
        max_frame=args.max_stream_frames,
        segments=segments,
        log=log,
    )

    log("[3/5] fine_grained_level=1, enable_gap=False (gaps skipped)")
    dataset = build_dataset(args, level=1, enable_gap=False, labels=labels)
    stream_and_check(
        dataset,
        level=1,
        enable_gap=False,
        max_frame=args.max_stream_frames,
        segments=segments,
        log=log,
    )

    log("[4/5] rejected-configuration matrix")
    check_rejections(args, log)

    log("[5/5] transform + collation contract (vlm_vla tokenization)")
    dataset = build_dataset(args, level=1, enable_gap=True, labels=labels)
    check_collation(args, dataset, log)

    log("ALL CHECKS PASSED")
    report_path.write_text("\n".join(lines) + "\n")
    print(f"report written to {report_path}")


if __name__ == "__main__":
    main()
