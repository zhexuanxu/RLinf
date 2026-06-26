#!/bin/bash
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
#
# Restart-from-scratch supervisor for BEHAVIOR pi0.5 vlm_vla SFT runs.
#
# Both A800 boxes have shown a rare transient Ray "owner's node has crashed"
# driver death after hours of HEALTHY training (no OOM / reboot / disk / val).
# IMPORTANT: the SFT runner's checkpoint-resume path is BROKEN for the streaming
# behavior dataloader, so this wrapper NEVER resumes. On any crash it RESTARTS
# THE RUN FROM SCRATCH (step 0) and loops until one attempt completes max_steps
# cleanly (detected by a global_step_<max_steps> checkpoint appearing). A
# fast-fail guard aborts on a genuine (non-transient) error so it cannot
# hot-loop. The runner loops `for _step in range(0, max_steps)` with no epoch
# cap (the streaming loader cycles), so max_steps alone bounds a run.
#
# Usage:
#   supervise_vla_sft.sh <config> <experiment_name> <max_steps> [extra hydra overrides...]
set -u

REPO="/mnt/public/xzxuan/repos/RLinf"
if [[ $# -lt 3 ]]; then
    echo "usage: $0 <config> <experiment_name> <max_steps> [extra hydra overrides...]" >&2
    exit 2
fi
CONFIG="$1"; EXP="$2"; MAX="$3"; shift 3
EXTRA=("$@")
cd "$REPO"

# Newest checkpoint step for this experiment across ALL timestamped log dirs
# (each fresh attempt creates its own logs/<ts>-<config>/ dir). A clean finish
# writes global_step_<MAX>; that is the only completion signal we trust, since
# run_vla_sft.sh pipes through `tee` and so always returns exit code 0.
latest_step() {
    find "$REPO/logs" -maxdepth 4 -type d \
        -path "*-${CONFIG}/${EXP}/checkpoints/global_step_*" -printf '%f\n' 2>/dev/null \
        | sed 's/^global_step_//' | grep -E '^[0-9]+$' | sort -n | tail -1
}

echo "[supervisor] === ${EXP} (config=${CONFIG}, target=${MAX} steps); RESTART-FROM-SCRATCH on crash, resume DISABLED ==="
attempt=0
fastfail=0
while true; do
    attempt=$((attempt + 1))
    S="$(latest_step)"; S="${S:-0}"
    if [[ "$S" -ge "$MAX" ]]; then
        echo "[supervisor] ${EXP}: found global_step_${S} >= ${MAX} -- COMPLETE."
        break
    fi

    echo "[supervisor] attempt ${attempt}: FRESH start from step 0 (resume intentionally disabled)"
    T0=$SECONDS
    # Keep a few safety checkpoints without flooding the shared FS; resume is
    # disabled so intermediate checkpoints are only a completion marker + the
    # final step-50000 checkpoint used for evaluation.
    bash "$REPO/examples/sft/run_vla_sft.sh" "$CONFIG" \
        runner.max_steps="$MAX" runner.save_interval=10000 \
        runner.logger.experiment_name="$EXP" \
        "${EXTRA[@]}"
    DT=$((SECONDS - T0))

    A="$(latest_step)"; A="${A:-0}"
    echo "[supervisor] attempt ${attempt} returned after ${DT}s; newest checkpoint step ${A}"
    if [[ "$A" -ge "$MAX" ]]; then
        echo "[supervisor] ${EXP} COMPLETE (step ${A})."
        break
    fi

    # A transient node crash still trains for hours; a real config/code error
    # fails within seconds. Abort after 4 consecutive sub-2-minute attempts.
    if [[ "$DT" -lt 120 ]]; then
        fastfail=$((fastfail + 1))
        echo "[supervisor] WARNING: fast failure ${fastfail}/4 (ran ${DT}s) -- likely a real error, not a transient node crash"
        if [[ "$fastfail" -ge 4 ]]; then
            echo "[supervisor] ABORT: 4 consecutive fast failures; stopping to avoid a hot-loop." >&2
            exit 1
        fi
    else
        fastfail=0
    fi
    sleep 30
done
echo "[supervisor] === done: ${EXP} ==="
