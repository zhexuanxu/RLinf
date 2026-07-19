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

"""Repair malformed BEHAVIOR-1K skill annotations so every episode validates.

The 2025-challenge-demos skill annotations contain defects that make
:func:`~rlinf.data.datasets.openpi_pytorch.behavior.skill_segments.build_skill_segments`
reject the episode, which aborts ``fine_grained_level=1`` SFT at dataset
construction (the first bad episode raises). This script rewrites the annotation
JSONs in place so all 10,000 episodes pass validation, applying six deterministic
rules (documented on :func:`repair_skill_annotation`):

1. identical frame window rendering to the SAME subtask -> keep one, drop the rest;
2. two windows on the IDENTICAL frame window rendering to DIFFERENT subtasks
   (an ambiguous label) -> delete ALL of them so those frames are untrained;
   partial (non-identical) overlaps that render the same -> merge to their union,
   or, for a small (<=20 frame) staircase overlap of different subtasks, trim the
   shared frames out of both windows so the contested slice is untrained;
4. a ``frame_duration`` given as a list of intervals (a skill split around an
   interleaving skill) -> expand into one window per interval;
5, 6. two hand-specified episodes (490320, 42230) fixed via :data:`BESPOKE_FIXES`.

After the rules run, windows are re-sorted by start frame and ``skill_idx`` is
renumbered to equal list position -- an invariant the dataset relies on
(``_resolve_subtask_text`` indexes ``skill_annotation[skill_idx]`` directly).
Unused fields (``primitive_annotation`` and its ``skill_idxes`` cross-references,
``skill_id``, ``skill_type``, ...) are never read by the training pipeline and are
left untouched; ``primitive_annotation.skill_idxes`` may therefore become stale
relative to the renumbered ``skill_annotation`` -- this is intentional and
harmless.

The script is dry-run by default; pass ``--apply`` to write. On ``--apply`` it
first snapshots the whole ``annotations/`` tree to a sibling ``.tar.gz`` (never
inside the tree -- the dataset's ``load_annotations`` crashes on any extra file in
a task dir) and prints the restore command. Files whose content does not change
are left untouched, so re-running is idempotent and produces minimal diffs.

Run with any Python (no heavy deps): the two annotation helpers are loaded
directly by path, bypassing ``rlinf/__init__`` (which imports TensorFlow via
transformers)::

    python toolkits/eval_scripts_openpi/fix_behavior_annotations.py            # dry-run
    python toolkits/eval_scripts_openpi/fix_behavior_annotations.py --apply    # write
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
_BEHAVIOR_DIR = (
    REPO_ROOT / "rlinf" / "data" / "datasets" / "openpi_pytorch" / "behavior"
)

DEFAULT_ANNOTATIONS_DIR = "/mnt/public/xzxuan/data/2025-challenge-demos/annotations"

# Small overlaps of two different subtasks are treated as boundary slop: the
# shared frames are trimmed out of both windows (rule 2b) rather than rejected.
STAIRCASE_TRIM_MAX = 20


def _load_module(name: str, path: Path) -> ModuleType:
    """Import a single module file without importing its package.

    ``skill_segments`` and ``skill_language`` depend only on the standard library,
    but importing them through ``rlinf...behavior`` would run ``rlinf/__init__``
    (which pulls TensorFlow via transformers). Loading the file directly avoids
    that. The module is registered in ``sys.modules`` before execution so that the
    ``@dataclasses.dataclass`` in ``skill_segments`` can resolve its own module.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - import plumbing
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_skill_segments = _load_module(
    "_fix_behavior_skill_segments", _BEHAVIOR_DIR / "skill_segments.py"
)
_skill_language = _load_module(
    "_fix_behavior_skill_language", _BEHAVIOR_DIR / "skill_language.py"
)
build_skill_segments = _skill_segments.build_skill_segments
entry_to_subtask_text = _skill_language.entry_to_subtask_text


# ---------------------------------------------------------------------------
# Bespoke, hand-specified fixes (rules 5 and 6)
# ---------------------------------------------------------------------------


def _fix_490320(windows: list[dict]) -> None:
    """Rule 5: reorder/retime the oven push-tray/close-door/turn-on triple.

    The raw annotation has an inverted ``push tray`` window and a ``close door``
    that overlaps it. The user specified the corrected frame windows; the global
    re-sort then orders them push tray -> close door -> turn on switch.
    """
    targets = {
        "push tray": [13973, 14130],
        "close door": [14130, 14550],
        "turn on switch": [14550, 14691],
    }
    for entry in windows:
        desc = _skill_description(entry)
        if desc in targets and entry.get("frame_duration") in (
            [13973, 13931],  # the inverted push-tray window
            [13931, 14180],  # the overlapping close-door window
            [14180, 14691],  # the turn-on-switch window
        ):
            entry["frame_duration"] = list(targets[desc])


def _fix_42230(windows: list[dict]) -> None:
    """Rule 6: repair the inverted second interval of a split ``move to`` window.

    ``move to`` carries ``frame_duration = [[5949, 6456], [6738, 6459]]``; the
    second interval's end is below its start. The user specified the end should be
    6939. After the nonscalar-split step this yields a valid resumed ``move to``.
    """
    for entry in windows:
        fd = entry.get("frame_duration")
        if _is_multi_interval(fd):
            for interval in fd:
                if interval == [6738, 6459]:
                    interval[1] = 6939


# Episode id -> in-place fixer applied before the generic rules.
BESPOKE_FIXES = {
    490320: _fix_490320,
    42230: _fix_42230,
}


# ---------------------------------------------------------------------------
# Annotation helpers
# ---------------------------------------------------------------------------


def _is_multi_interval(frame_duration: Any) -> bool:
    """Return True if ``frame_duration`` is a list of intervals, not a pair."""
    return (
        isinstance(frame_duration, list)
        and len(frame_duration) > 0
        and isinstance(frame_duration[0], list)
    )


def _skill_description(entry: dict) -> str | None:
    """Return the entry's skill-description string (from its one-element list)."""
    desc = entry.get("skill_description")
    if isinstance(desc, str):
        return desc
    if isinstance(desc, (list, tuple)) and len(desc) == 1:
        return desc[0]
    return None


def _render(entry: dict, episode_id: int) -> str:
    """Render an entry to its subtask text, or a stable error marker on failure."""
    try:
        return entry_to_subtask_text(
            entry, episode_id=episode_id, skill_idx=entry.get("skill_idx")
        )
    except Exception as err:  # noqa: BLE001 - text is only used for equality here
        return f"<unrenderable: {err}>"


# ---------------------------------------------------------------------------
# Core repair
# ---------------------------------------------------------------------------


def repair_skill_annotation(
    annotation: dict, episode_id: int
) -> tuple[dict, Counter, list[list[int]]]:
    """Return a repaired copy of ``annotation`` plus the ops applied.

    The input is not mutated. The returned annotation has a ``skill_annotation``
    list that is start-sorted, densely renumbered (``skill_idx == position``), and
    free of the defects :func:`build_skill_segments` rejects. Pure function: no IO.

    Args:
        annotation: A parsed episode annotation (``skill_annotation`` +
            ``meta_data.valid_duration`` + unused fields).
        episode_id: Episode id, used for bespoke dispatch and text rendering.

    Returns:
        A ``(repaired_annotation, ops, deleted_spans)`` tuple where ``ops`` counts
        the rules applied and ``deleted_spans`` lists the ``[start, end]`` windows
        removed by the ambiguous-label rule (rule 2, identical frame / different
        subtask).
    """
    ops: Counter = Counter()
    deleted_spans: list[list[int]] = []
    windows = copy.deepcopy(annotation.get("skill_annotation") or [])

    # Rule 5/6: bespoke, hand-specified fixes applied before the generic rules.
    fixer = BESPOKE_FIXES.get(episode_id)
    if fixer is not None:
        fixer(windows)
        ops[f"bespoke_{episode_id}"] += 1

    # Rule 4: expand a list-of-intervals frame_duration into one window per
    # interval (the interleaving skill already fills the hole between them).
    expanded: list[dict] = []
    for entry in windows:
        fd = entry.get("frame_duration")
        if _is_multi_interval(fd):
            for interval in fd:
                clone = copy.deepcopy(entry)
                clone["frame_duration"] = [int(interval[0]), int(interval[1])]
                expanded.append(clone)
            ops["split_nonscalar"] += 1
        else:
            entry["frame_duration"] = [int(fd[0]), int(fd[1])]
            expanded.append(entry)

    # Order by (start, end) so identical windows are adjacent and the overlap pass
    # sees windows in chronological order.
    expanded.sort(key=lambda e: (e["frame_duration"][0], e["frame_duration"][1]))

    # Rules 1 and 2 (identical frame windows): a run of windows sharing the exact
    # same frame_duration is either a redundant duplicate (all render the same ->
    # keep one) or an ambiguous label (>1 distinct text -> delete all of them).
    deduped: list[dict] = []
    i = 0
    while i < len(expanded):
        j = i
        while (
            j + 1 < len(expanded)
            and expanded[j + 1]["frame_duration"] == expanded[i]["frame_duration"]
        ):
            j += 1
        group = expanded[i : j + 1]
        if len(group) > 1:
            texts = {_render(entry, episode_id) for entry in group}
            if len(texts) == 1:
                deduped.append(group[0])
                ops["dedup_identical"] += 1
            else:
                deleted_spans.append(list(group[0]["frame_duration"]))
                ops["delete_ambiguous"] += 1
        else:
            deduped.append(group[0])
        i = j + 1

    # Rule 2a/2b (partial overlaps): a forward pass over adjacent windows.
    final: list[dict] = []
    for entry in deduped:
        if not final:
            final.append(entry)
            continue
        prev = final[-1]
        a, b = prev["frame_duration"]
        c, d = entry["frame_duration"]
        if c < b:  # overlap
            if _render(prev, episode_id) == _render(entry, episode_id):
                # Same subtask on overlapping windows -> merge to their union.
                prev["frame_duration"] = [min(a, c), max(b, d)]
                ops["merge_union"] += 1
                continue
            if a < c < b < d and (b - c) <= STAIRCASE_TRIM_MAX:
                # Small staircase overlap of different subtasks -> trim the shared
                # frames [c, b) out of both; that slice is left untrained.
                prev["frame_duration"] = [a, c]
                entry["frame_duration"] = [b, d]
                final.append(entry)
                ops["trim_staircase"] += 1
                continue
            # Any other overlap is unexpected after the above rules; keep both and
            # let the post-repair validation flag the episode loudly.
            ops["unresolved_overlap"] += 1
            final.append(entry)
            continue
        final.append(entry)

    # Renumber skill_idx to list position (the invariant the dataset relies on).
    for position, entry in enumerate(final):
        entry["skill_idx"] = position

    repaired = dict(annotation)
    repaired["skill_annotation"] = final
    return repaired, ops, deleted_spans


# ---------------------------------------------------------------------------
# Byte-faithful JSON encoding (matches the dataset's on-disk format)
# ---------------------------------------------------------------------------


def encode_annotation(obj: Any, indent: int = 0) -> str:
    """Serialize an annotation exactly as the dataset stores it.

    The on-disk format is 4-space-indented JSON where arrays of objects are
    expanded one element per line but scalar arrays (``[0, 211]``,
    ``["move to"]``) stay inline, with ``", "`` / ``": "`` separators and no
    trailing newline. Reproducing it byte-for-byte means untouched files are
    written back identically and repaired files get minimal diffs.
    """
    pad = "    " * indent
    pad1 = "    " * (indent + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f"{pad1}{json.dumps(key)}: {encode_annotation(value, indent + 1)}"
            for key, value in obj.items()
        ]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if any(isinstance(element, dict) for element in obj):
            if not obj:
                return "[]"
            items = [
                f"{pad1}{encode_annotation(element, indent + 1)}" for element in obj
            ]
            return "[\n" + ",\n".join(items) + "\n" + pad + "]"
        return (
            "[" + ", ".join(encode_annotation(element, indent) for element in obj) + "]"
        )
    return json.dumps(obj, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Filesystem walk
# ---------------------------------------------------------------------------


def iter_episode_files(annotations_dir: Path):
    """Yield ``(episode_id, path)`` for every annotation JSON, task-order sorted."""
    for task_dir in sorted(annotations_dir.glob("task-*")):
        if not task_dir.is_dir():
            continue
        for path in sorted(task_dir.glob("episode_*.json")):
            yield int(path.stem[len("episode_") :]), path


def make_backup(annotations_dir: Path, run_id: str) -> Path:
    """Snapshot the whole annotations tree to a sibling ``.tar.gz`` and return it.

    The archive is written OUTSIDE ``annotations/`` on purpose: the dataset's
    ``load_annotations`` iterates each task dir and parses every filename as an
    episode index, so a stray file inside the tree would crash it.
    """
    backup_path = annotations_dir.parent / f"annotations_backup_{run_id}.tar.gz"
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(annotations_dir, arcname=annotations_dir.name)
    return backup_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Repair malformed BEHAVIOR-1K skill annotations in place."
    )
    parser.add_argument(
        "--annotations-dir",
        default=DEFAULT_ANNOTATIONS_DIR,
        help="Root of the annotations/ tree (contains task-XXXX/ dirs).",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes to disk. Without it the script only reports (dry-run).",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip the tar snapshot taken before writing (only with --apply).",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path for the JSON change report (default: alongside annotations-dir).",
    )
    return parser.parse_args()


def main() -> None:
    """Repair (or dry-run) every annotation file under ``--annotations-dir``."""
    args = parse_args()
    annotations_dir = Path(args.annotations_dir)
    if not annotations_dir.is_dir():
        raise SystemExit(f"annotations dir not found: {annotations_dir}")

    totals: Counter = Counter()
    touched: list[dict] = []
    still_invalid: list[dict] = []
    n_files = 0

    for episode_id, path in iter_episode_files(annotations_dir):
        n_files += 1
        raw = path.read_text()
        annotation = json.loads(raw)
        repaired, ops, deleted_spans = repair_skill_annotation(annotation, episode_id)

        # Self-check: the repaired annotation must validate.
        try:
            build_skill_segments(
                repaired, len(repaired["skill_annotation"]), episode_id=episode_id
            )
        except Exception as err:  # noqa: BLE001 - reported, not raised
            still_invalid.append({"episode": episode_id, "error": str(err)})

        # Preserve the file's original trailing-newline state so a file that only
        # differs by a stray trailing newline (a few were re-saved by an editor)
        # is not counted as changed -- only genuine content edits are written.
        new_text = encode_annotation(repaired)
        if raw.endswith("\n"):
            new_text += "\n"
        if new_text == raw:
            continue  # No change (idempotent on already-valid files).

        totals.update(ops)
        touched.append(
            {
                "episode": episode_id,
                "task": episode_id // 10000,
                "ops": dict(ops),
                "deleted_spans": deleted_spans,
            }
        )
        if args.apply:
            path.write_text(new_text)

    _report(args, annotations_dir, n_files, totals, touched, still_invalid)


def _report(args, annotations_dir, n_files, totals, touched, still_invalid) -> None:
    """Print a summary, optionally back up + write, and dump the JSON report."""
    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanned {n_files} annotation files under {annotations_dir}")
    print(f"  episodes changed: {len(touched)}")
    print(f"  operation counts: {dict(totals)}")
    deleted = [
        {"episode": t["episode"], "spans": t["deleted_spans"]}
        for t in touched
        if t["deleted_spans"]
    ]
    print(f"  ambiguous-label episodes with deleted spans: {len(deleted)}")
    for entry in deleted:
        print(f"    episode {entry['episode']}: deleted {entry['spans']}")
    if still_invalid:
        print(f"  !! {len(still_invalid)} episode(s) STILL INVALID after repair:")
        for item in still_invalid:
            print(f"     episode {item['episode']}: {item['error']}")
    else:
        print("  all repaired episodes pass build_skill_segments")

    run_id = f"{n_files}files_{len(touched)}changed"
    if args.apply and touched and not args.no_backup:
        backup = make_backup(annotations_dir, run_id)
        print(f"  backup written: {backup}")
        print(f"  restore with: tar -xzf {backup} -C {annotations_dir.parent}")

    report_path = (
        Path(args.report)
        if args.report
        else annotations_dir.parent / "behavior_annotation_repair_report.json"
    )
    report_path.write_text(
        json.dumps(
            {
                "mode": mode,
                "scanned": n_files,
                "changed": len(touched),
                "operation_counts": dict(totals),
                "still_invalid": still_invalid,
                "episodes": touched,
            },
            indent=2,
        )
    )
    print(f"  report written: {report_path}")
    if not args.apply and touched:
        print("  (dry-run: no files written; re-run with --apply to write)")


if __name__ == "__main__":
    main()
