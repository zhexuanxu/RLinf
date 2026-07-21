#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One command: convert ALL 50 BEHAVIOR R1Pro tasks from 23-dim absolute-joint
# actions to 21-dim delta-EEF actions, then build the 50-task delta-EEF
# quantile norm-stats asset.
#
# Run it foreground in a tmux terminal (it prints per-batch progress + ETA):
#
#     tmux new -s eef_convert
#     bash toolkits/eval_scripts_openpi/convert_all_50_to_eef_delta.sh
#
# Every path is overridable via environment variables (defaults below match the
# joint-side layout: 2025-challenge-demos + outputs/norm_stats/50tasks_reorder).
# Example overrides:
#
#     DST_ROOT=/mnt/public/xzxuan/data/2025-challenge-demos-eef-delta-50 \
#     OVERWRITE=0 VENV_PATH=/root/.venv_pi \
#     bash toolkits/eval_scripts_openpi/convert_all_50_to_eef_delta.sh
#
# What it produces:
#   * $DST_ROOT                          -- 21-dim delta-EEF LeRobot dataset
#                                           (videos/ + per-task meta symlinked,
#                                           original dataset untouched)
#   * $ASSETS_DIR/$ASSET_ID/norm_stats.json
#                                        -- eef_delta_pose norm stats (21
#                                           meaningful dims, padded to ACTION_DIM,
#                                           with a control-mode manifest)
# ---------------------------------------------------------------------------
set -euo pipefail

# ---- Configuration (override any of these via the environment) -------------
VENV_PATH="${VENV_PATH:-/mnt/public/xzxuan/.venv_pi}"
REPO="${REPO:-/mnt/public/xzxuan/repos/RLinf}"

# Source (original 23-dim joint dataset) and destination (converted 21-dim delta).
SRC_ROOT="${SRC_ROOT:-/mnt/public/xzxuan/data/2025-challenge-demos}"
DST_ROOT="${DST_ROOT:-/mnt/public/xzxuan/data/2025-challenge-demos-eef-delta}"
URDF_PATH="${URDF_PATH:-/mnt/public/xzxuan/data/BEHAVIOR-1K-datasets_372/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf}"
OMNIGIBSON_VERSION="${OMNIGIBSON_VERSION:-3.7.2}"

# Norm-stats asset: {ASSETS_DIR}/{ASSET_ID}/norm_stats.json.
ASSETS_DIR="${ASSETS_DIR:-$REPO/outputs/norm_stats}"
ASSET_ID="${ASSET_ID:-50tasks_eef_delta}"
# MUST match actor.model.openpi.state_order in the training/eval config, and
# ACTION_DIM MUST match the model action dim (pi05 = 32).
STATE_ORDER="${STATE_ORDER:-align}"
ACTION_DIM="${ACTION_DIM:-32}"

# 1 = delete and rebuild $DST_ROOT if it already exists; 0 = fail if it exists.
OVERWRITE="${OVERWRITE:-1}"
PROGRESS_EVERY="${PROGRESS_EVERY:-200}"
# Space-separated task names to convert; empty (default) = ALL tasks in src.
TASKS="${TASKS:-}"

CVT="$REPO/rlinf/data/datasets/openpi_pytorch/behavior/convert_to_eef_delta.py"
STATS="$REPO/rlinf/data/datasets/openpi_pytorch/behavior/compute_norm_stats.py"

# ---- Preflight -------------------------------------------------------------
[ -f "$VENV_PATH/bin/activate" ] || { echo "ERROR: no venv at $VENV_PATH" >&2; exit 1; }
[ -f "$SRC_ROOT/meta/tasks.jsonl" ] || { echo "ERROR: no $SRC_ROOT/meta/tasks.jsonl" >&2; exit 1; }
[ -f "$URDF_PATH" ] || { echo "ERROR: URDF not found: $URDF_PATH" >&2; exit 1; }
[ -f "$CVT" ] || { echo "ERROR: converter not found: $CVT" >&2; exit 1; }
[ -f "$STATS" ] || { echo "ERROR: norm-stats script not found: $STATS" >&2; exit 1; }

# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
cd "$REPO"

N_TASKS="$(grep -c . "$SRC_ROOT/meta/tasks.jsonl")"

cat <<BANNER
============================================================================
 BEHAVIOR joint -> delta-EEF conversion (all tasks)
----------------------------------------------------------------------------
 venv        : $VENV_PATH
 src (joint) : $SRC_ROOT   ($N_TASKS tasks)
 dst (delta) : $DST_ROOT
 urdf        : $URDF_PATH
 norm stats  : $ASSETS_DIR/$ASSET_ID/norm_stats.json
 state order : $STATE_ORDER   action dim (pad): $ACTION_DIM
 overwrite   : $OVERWRITE
============================================================================
BANNER

CVT_ARGS=(
  --src-root "$SRC_ROOT"
  --dst-root "$DST_ROOT"
  --urdf-path "$URDF_PATH"
  --omnigibson-version "$OMNIGIBSON_VERSION"
  --progress-every "$PROGRESS_EVERY"
)
if [ -n "$TASKS" ]; then
  # shellcheck disable=SC2206  # intentional word-split into separate --tasks args
  CVT_ARGS+=(--tasks $TASKS)
  echo "NOTE: TASKS set -> converting only: $TASKS"
fi
if [ "$OVERWRITE" = "1" ]; then
  CVT_ARGS+=(--overwrite)
  if [ -e "$DST_ROOT" ]; then
    echo "WARNING: OVERWRITE=1 -> $DST_ROOT will be DELETED and rebuilt."
    echo "         (Ctrl-C within 8s to abort; set OVERWRITE=0 or DST_ROOT to keep it.)"
    sleep 8
  fi
fi

# ---- Step 1/2: convert every task (all tasks = omit --tasks) ---------------
# Run by file path: the converter imports only numpy/pyarrow (no torch/Isaac).
echo ">>> [1/2] converting actions ($(date '+%F %T')) ..."
python "$CVT" "${CVT_ARGS[@]}"

# ---- Step 2/2: build the delta-EEF norm-stats asset ------------------------
# --from-episodes-stats aggregates the converted dataset's (already task-filtered)
# meta/episodes_stats.jsonl with the faithful numpy re-impl of OmniGibson's
# aggregate_stats. PYTHONPATH=$REPO so `import rlinf` resolves when run by path.
echo ">>> [2/2] computing $ASSET_ID norm stats ($(date '+%F %T')) ..."
PYTHONPATH="$REPO" python "$STATS" \
  --dataset-root "$DST_ROOT" \
  --control-mode eef_delta_pose \
  --from-episodes-stats \
  --state-order "$STATE_ORDER" \
  --action-dim "$ACTION_DIM" \
  --output-dir "$ASSETS_DIR/$ASSET_ID"

# ---- Verify (JSON reads + one real parquet action row) ---------------------
# Assert (not just print) that the dataset, provenance, tasks.jsonl, and the
# stats manifest all agree on the expected number of converted tasks, so a
# partial/failed conversion fails loudly here instead of training on a subset.
echo ">>> verifying outputs ..."
if [ -n "$TASKS" ]; then
  EXPECTED_TASKS=$(printf '%s\n' $TASKS | grep -c .)
else
  EXPECTED_TASKS="$N_TASKS"
fi
python - "$DST_ROOT" "$ASSETS_DIR/$ASSET_ID/norm_stats.json" "$EXPECTED_TASKS" <<'PY'
import glob
import json
import sys

import numpy as np
import pyarrow.parquet as pq

dst, stats_path, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
info = json.load(open(f"{dst}/meta/info.json"))
prov = json.load(open(f"{dst}/meta/eef_delta_provenance.json"))
stats = json.load(open(stats_path))
meta = stats.get("metadata", {})
act_shape = info["features"]["action"]["shape"]
assert act_shape == [21], f"dataset action shape {act_shape} != [21]"
assert prov["control_mode"] == "eef_delta_pose"
assert meta.get("control_mode") == "eef_delta_pose", meta
assert meta.get("action_env_dim") == 21, meta
# Coverage: every task-count view must agree on the expected number of tasks.
n_taskrows = sum(1 for ln in open(f"{dst}/meta/tasks.jsonl") if ln.strip())
assert info.get("total_tasks") == expected, (
    f"info.total_tasks={info.get('total_tasks')} != expected {expected}"
)
assert len(prov.get("tasks") or []) == expected, (
    f"provenance tasks={len(prov.get('tasks') or [])} != expected {expected}"
)
assert n_taskrows == expected, f"tasks.jsonl rows={n_taskrows} != expected {expected}"
assert len(meta.get("tasks") or []) == expected, (
    f"stats manifest tasks={len(meta.get('tasks') or [])} != expected {expected}"
)
# Inspect a real parquet action row, not just metadata.
sample = sorted(glob.glob(f"{dst}/data/*/*.parquet"))
assert sample, "no parquet files found under data/"
row = pq.read_table(sample[0], columns=["action"]).to_pandas()["action"].iloc[0]
assert len(np.asarray(row)) == 21, f"parquet action width {len(np.asarray(row))} != 21"
n_meaningful = sum(1 for v in stats["norm_stats"]["actions"]["q99"] if v != 0.0)
print(f"  dataset action shape : {act_shape}  (sampled parquet row width 21 OK)")
print(
    f"  tasks covered        : {expected} "
    f"(info.total_tasks / provenance / tasks.jsonl / manifest all agree)"
)
print(f"  dataset episodes     : {info.get('total_episodes')}  frames: {info.get('total_frames')}")
print(f"  stats meaningful dims: >= {n_meaningful} (of 21)  padded to {len(stats['norm_stats']['actions']['q99'])}")
print("  OK: dataset + norm stats are consistent for eef_delta_pose.")
PY

echo "============================================================================"
echo " DONE. Wire a delta SFT/eval config with:"
echo "   behavior_dataset_root_eef_delta: $DST_ROOT"
echo "   assets_dir_eef_delta: $ASSETS_DIR   asset_id_eef_delta: $ASSET_ID"
echo "   control_mode: eef_delta_pose   state_order: $STATE_ORDER   action_dim: 21"
echo "============================================================================"
