#!/usr/bin/env python3
"""Re-index BEHAVIOR task-0000 episodes to contiguous indices for LeRobot compatibility.

LeRobot v2.1 has an episode indexing bug where non-contiguous episode indices
(e.g., 10, 20, 30, ...) cause IndexError in _get_query_indices. This script
creates a clean subset dataset with contiguous indices 0-199.

Usage:
    python toolkits/behavior/prepare_behavior_task0000.py

Source: /mnt/public/xzxuan/data/2025-challenge-demos-short/
Output: /mnt/public/xzxuan/data/behavior-task0000-reindexed/
"""

import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

SRC_DIR = Path("/mnt/public/xzxuan/data/2025-challenge-demos-short")
DST_DIR = Path("/mnt/public/xzxuan/data/behavior-task0000-reindexed")

TASK_TEXT = "Turn on the radio receiver that's on the table in the living room."

# Only keep RGB video features (skip depth and segmentation)
RGB_VIDEO_KEYS = [
    "observation.images.rgb.head",
    "observation.images.rgb.left_wrist",
    "observation.images.rgb.right_wrist",
]


def find_task0_episodes():
    """Find all episode indices belonging to task-0000."""
    episodes_path = SRC_DIR / "meta" / "episodes.jsonl"
    task0_eps = []
    with open(episodes_path) as f:
        for line in f:
            ep = json.loads(line)
            if TASK_TEXT in ep.get("tasks", []):
                task0_eps.append(ep)
    task0_eps.sort(key=lambda e: e["episode_index"])
    return task0_eps


def compute_chunk(episode_index: int, chunks_size: int = 1000) -> int:
    return episode_index // chunks_size


def main():
    print(f"Source: {SRC_DIR}")
    print(f"Output: {DST_DIR}")

    # Find task-0000 episodes
    task0_eps = find_task0_episodes()
    print(f"Found {len(task0_eps)} task-0000 episodes")

    # Build old_idx -> new_idx mapping
    old_to_new = {}
    for new_idx, ep in enumerate(task0_eps):
        old_to_new[ep["episode_index"]] = new_idx

    # Clean output directory
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    DST_DIR.mkdir(parents=True)

    # --- Write meta/tasks.jsonl ---
    meta_dir = DST_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({
            "task_index": 0,
            "task": TASK_TEXT,
        }) + "\n")

    # --- Write meta/episodes.jsonl ---
    total_frames = 0
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for new_idx, ep in enumerate(task0_eps):
            new_ep = {
                "episode_index": new_idx,
                "tasks": ep["tasks"],
                "length": ep["length"],
            }
            # Copy optional fields
            for k in ["distance_traveled", "left_eef_displacement", "right_eef_displacement"]:
                if k in ep:
                    new_ep[k] = ep[k]
            total_frames += ep["length"]
            f.write(json.dumps(new_ep) + "\n")

    # --- Rewrite parquet files ---
    print("Rewriting parquet files...")
    for new_idx, ep in enumerate(task0_eps):
        old_idx = ep["episode_index"]
        old_chunk = compute_chunk(old_idx)
        new_chunk = compute_chunk(new_idx)

        # Source parquet
        src_parquet = SRC_DIR / "data" / f"chunk-{old_chunk:03d}" / f"episode_{old_idx:06d}.parquet"
        if not src_parquet.exists():
            # Try flat layout (no chunk subdirectory)
            src_parquet = SRC_DIR / "data" / f"task-0000" / f"episode_{old_idx:08d}.parquet"
        if not src_parquet.exists():
            print(f"  WARNING: parquet not found for episode {old_idx}, trying other patterns...")
            # Search for the file
            import glob
            candidates = glob.glob(str(SRC_DIR / "data" / "**" / f"episode_{old_idx:08d}.parquet"), recursive=True)
            if not candidates:
                candidates = glob.glob(str(SRC_DIR / "data" / "**" / f"episode_{old_idx:06d}.parquet"), recursive=True)
            if candidates:
                src_parquet = Path(candidates[0])
            else:
                raise FileNotFoundError(f"Cannot find parquet for episode {old_idx}")

        # Read and rewrite
        table = pq.read_table(src_parquet)

        # Replace episode_index column
        col_idx = table.column_names.index("episode_index")
        n_rows = len(table)
        new_ep_col = pa.array([new_idx] * n_rows, type=pa.int64())
        table = table.set_column(col_idx, "episode_index", new_ep_col)

        # Replace task_index to 0 if present
        if "task_index" in table.column_names:
            ti_idx = table.column_names.index("task_index")
            new_ti_col = pa.array([0] * n_rows, type=pa.int64())
            table = table.set_column(ti_idx, "task_index", new_ti_col)

        # Recompute contiguous index column
        if "index" in table.column_names:
            # We need a global frame offset for this episode
            # For now, keep the original index values — they'll be recomputed by LeRobot
            pass

        # Write to destination
        dst_chunk_dir = DST_DIR / "data" / f"chunk-{new_chunk:03d}"
        dst_chunk_dir.mkdir(parents=True, exist_ok=True)
        dst_parquet = dst_chunk_dir / f"episode_{new_idx:06d}.parquet"
        pq.write_table(table, dst_parquet)

        if new_idx % 50 == 0:
            print(f"  Processed {new_idx + 1}/{len(task0_eps)} episodes")

    print(f"  Done: {len(task0_eps)} parquet files written")

    # --- Symlink video files ---
    print("Symlinking video files...")
    for new_idx, ep in enumerate(task0_eps):
        old_idx = ep["episode_index"]
        old_chunk = compute_chunk(old_idx)
        new_chunk = compute_chunk(new_idx)

        for vkey in RGB_VIDEO_KEYS:
            # Source video path
            src_video = SRC_DIR / "videos" / f"chunk-{old_chunk:03d}" / vkey / f"episode_{old_idx:06d}.mp4"
            if not src_video.exists():
                src_video = SRC_DIR / "videos" / f"task-0000" / vkey / f"episode_{old_idx:08d}.mp4"
            if not src_video.exists():
                # Search
                import glob
                candidates = glob.glob(str(SRC_DIR / "videos" / "**" / vkey / f"episode_{old_idx:08d}.mp4"), recursive=True)
                if not candidates:
                    candidates = glob.glob(str(SRC_DIR / "videos" / "**" / vkey / f"episode_{old_idx:06d}.mp4"), recursive=True)
                if candidates:
                    src_video = Path(candidates[0])
                else:
                    print(f"  WARNING: video not found for episode {old_idx}, key {vkey}")
                    continue

            dst_video_dir = DST_DIR / "videos" / f"chunk-{new_chunk:03d}" / vkey
            dst_video_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_video_dir / f"episode_{new_idx:06d}.mp4"
            if not dst_video.exists():
                os.symlink(src_video.resolve(), dst_video)

    print(f"  Done: video symlinks created")

    # --- Write meta/info.json ---
    # Read original info.json for reference
    with open(SRC_DIR / "meta" / "info.json") as f:
        orig_info = json.load(f)

    chunks_size = 1000
    num_episodes = len(task0_eps)

    # Build features dict — only keep what we need
    features = {}
    for vkey in RGB_VIDEO_KEYS:
        if vkey in orig_info.get("features", {}):
            features[vkey] = orig_info["features"][vkey]

    # Add non-video features
    for fkey in ["action", "timestamp", "episode_index", "index", "observation.state",
                 "observation.cam_rel_poses", "observation.task_info", "task_index"]:
        if fkey in orig_info.get("features", {}):
            features[fkey] = orig_info["features"][fkey]

    info = {
        "codebase_version": orig_info.get("codebase_version", "v2.1"),
        "robot_type": orig_info.get("robot_type", "R1Pro"),
        "total_episodes": num_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": num_episodes * len(RGB_VIDEO_KEYS),
        "total_chunks": (num_episodes + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": orig_info.get("fps", 30),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
        "splits": {"train": f"0:{num_episodes}"},
    }

    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDataset created at: {DST_DIR}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Features: {list(features.keys())}")


if __name__ == "__main__":
    main()
