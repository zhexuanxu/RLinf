#!/usr/bin/env python3
"""Extract video frames from BEHAVIOR-1K demos to JPEG photos on disk.

Usage:
    python tools/extract_behavior_photos.py \
        --data-root /mnt/public/xzxuan/data/2025-challenge-demos \
        --tasks turning_on_radio \
        --quality 75 \
        --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import av
from PIL import Image
from tqdm import tqdm

# Task name -> task index mapping (from behavior_dataset.py)
TASK_NAMES_TO_INDICES = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}


def extract_episode(args: tuple) -> tuple[int, int]:
    """Extract all frames from one episode video to JPEG files.

    Returns (episode_index, num_frames).
    """
    video_path, output_dir, quality, expected_frames = args
    ep_idx = int(Path(video_path).stem.replace("episode_", ""))

    # Idempotent: skip if already extracted with expected count
    if expected_frames and os.path.isdir(output_dir):
        existing = len([f for f in os.listdir(output_dir) if f.endswith(".jpg")])
        if existing == expected_frames:
            return ep_idx, existing

    os.makedirs(output_dir, exist_ok=True)

    container = av.open(video_path)
    stream = container.streams.video[0]
    frame_count = 0
    for frame in container.decode(stream):
        img = frame.to_ndarray(format="rgb24")
        pil_img = Image.fromarray(img)
        pil_img.save(
            os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"),
            "JPEG",
            quality=quality,
        )
        frame_count += 1
    container.close()

    return ep_idx, frame_count


def main():
    parser = argparse.ArgumentParser(description="Extract BEHAVIOR video frames to JPEG photos")
    parser.add_argument("--data-root", required=True, help="Path to 2025-challenge-demos")
    parser.add_argument("--tasks", nargs="+", default=["turning_on_radio"],
                        help="Task names to extract (default: turning_on_radio)")
    parser.add_argument("--camera", default="head", help="Camera name (default: head)")
    parser.add_argument("--quality", type=int, default=75, help="JPEG quality (default: 75)")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    photo_root = data_root / "photos"
    episodes_path = data_root / "meta" / "episodes.jsonl"

    # Load episode metadata
    episodes_by_task: dict[int, list[dict]] = {}
    with episodes_path.open() as f:
        for line in f:
            ep = json.loads(line)
            task_id = int(ep["episode_index"] // 1e4)
            episodes_by_task.setdefault(task_id, []).append(ep)

    # Build extraction work list
    work_items = []
    for task_name in args.tasks:
        if task_name not in TASK_NAMES_TO_INDICES:
            print(f"WARNING: Unknown task '{task_name}', skipping")
            continue
        task_id = TASK_NAMES_TO_INDICES[task_name]
        task_episodes = episodes_by_task.get(task_id, [])
        if not task_episodes:
            print(f"WARNING: No episodes found for task '{task_name}' (task-{task_id:04d})")
            continue

        video_dir = data_root / "videos" / f"task-{task_id:04d}" / f"observation.images.rgb.{args.camera}"
        if not video_dir.exists():
            print(f"WARNING: Video directory not found: {video_dir}")
            continue

        print(f"Task '{task_name}' (task-{task_id:04d}): {len(task_episodes)} episodes")

        for ep in sorted(task_episodes, key=lambda x: x["episode_index"]):
            ep_idx = ep["episode_index"]
            video_file = video_dir / f"episode_{ep_idx:08d}.mp4"
            if not video_file.exists():
                print(f"  WARNING: Missing video: {video_file}")
                continue
            output_dir = photo_root / f"task-{task_id:04d}" / f"episode_{ep_idx:08d}"
            work_items.append((
                str(video_file),
                str(output_dir),
                args.quality,
                ep.get("length", 0),
            ))

    if not work_items:
        print("No episodes to extract. Exiting.")
        sys.exit(1)

    print(f"\nTotal: {len(work_items)} episodes to extract")
    print(f"Output: {photo_root}")
    print(f"Quality: {args.quality}, Workers: {args.workers}\n")

    # Extract in parallel
    manifest = {}
    with Pool(args.workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(extract_episode, work_items),
            total=len(work_items),
            desc="Extracting frames",
        ))

    for ep_idx, num_frames in results:
        task_id = int(ep_idx // 1e4)
        manifest[str(ep_idx)] = {"num_frames": num_frames, "task_id": task_id}

    # Write manifest
    manifest_path = photo_root / "manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nManifest written to {manifest_path}")

    # Verify against episodes.jsonl
    mismatches = 0
    for ep_str, info in manifest.items():
        ep_idx = int(ep_str)
        task_id = info["task_id"]
        task_eps = episodes_by_task.get(task_id, [])
        expected = next((e["length"] for e in task_eps if e["episode_index"] == ep_idx), None)
        if expected and info["num_frames"] != expected:
            print(f"  MISMATCH: episode {ep_idx}: extracted {info['num_frames']} frames, expected {expected}")
            mismatches += 1

    if mismatches == 0:
        print(f"All {len(manifest)} episodes verified successfully!")
    else:
        print(f"WARNING: {mismatches} episodes have frame count mismatches")

    # Check sample image
    sample_ep = next(iter(manifest))
    task_id = manifest[sample_ep]["task_id"]
    sample_img = photo_root / f"task-{task_id:04d}" / f"episode_{int(sample_ep):08d}" / "frame_000000.jpg"
    if sample_img.exists():
        img = Image.open(sample_img)
        print(f"Sample image: {sample_img} -> {img.size}, mode={img.mode}")

    total_frames = sum(v["num_frames"] for v in manifest.values())
    print(f"\nTotal frames extracted: {total_frames:,}")


if __name__ == "__main__":
    main()
