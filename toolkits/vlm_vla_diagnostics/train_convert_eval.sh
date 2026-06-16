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
# Unattended train -> convert(sft2new) -> eval pipeline for the BEHAVIOR pi0.5
# vlm_vla recipe. Runs SFT, resolves the final checkpoint, converts it to the
# new bare-Pi0 layout the eval path loads, places norm stats where eval resolves
# them, then runs the env eval. Each path is explicit; nothing is guessed.
#
# Usage:
#   train_convert_eval.sh <train_config> <eval_config> <input_norm_stats> [--dry-run] [--final-256] [extra eval hydra overrides...]
#
# Iteration gate (64 trajectories for the current 8-env eval config):
#   toolkits/vlm_vla_diagnostics/train_convert_eval.sh \
#     behavior_pi05_vlm_vla behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval \
#     /mnt/public/xzxuan/repos/openpi-comet-pytorch-mixed/outputs/assets/train/pi05_b1k-task0000_sft_pytorch_mixed/behavior-1k/2025-challenge-demos/norm_stats.json
#
# Final confirmation gate (256 trajectories):
#   toolkits/vlm_vla_diagnostics/train_convert_eval.sh \
#     behavior_pi05_vlm_vla behavior_ppo_openpi_pi05_pytorch_vlm_vla_eval \
#     /path/to/norm_stats.json --final-256
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYBIN="${PYBIN:-/mnt/public/xzxuan/.venv_pi/bin/python}"
ASSET_ID="${ASSET_ID:-physical-intelligence/behavior}"

if [[ $# -lt 3 ]]; then
    echo "usage: $0 <train_config> <eval_config> <input_norm_stats> [--dry-run] [--final-256] [eval hydra overrides...]" >&2
    exit 2
fi
TRAIN_CONFIG="$1"; EVAL_CONFIG="$2"; INPUT_NORM_STATS="$3"; shift 3
DRY_RUN=0
FINAL_256=0
EXTRA_OVERRIDES=()
for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=1
            ;;
        --final-256)
            FINAL_256=1
            ;;
        *)
            EXTRA_OVERRIDES+=("$arg")
            ;;
    esac
done

run() {
    printf "+ "
    printf "%q " "$@"
    printf "\n"
    if [[ "$DRY_RUN" -eq 0 ]]; then "$@"; fi
}

# 1) Train. run_vla_sft.sh creates logs/<timestamp>-<config>/ and trains there.
echo "=== [1/3] SFT training: ${TRAIN_CONFIG} ==="
BEFORE_MARK="$(date +%s)"
run bash "${REPO_PATH}/examples/sft/run_vla_sft.sh" "${TRAIN_CONFIG}"

# Resolve the log dir this run created (newest logs/*-<train_config> after the mark).
LOG_DIR="$(find "${REPO_PATH}/logs" -maxdepth 1 -type d -name "*-${TRAIN_CONFIG}" -newermt "@${BEFORE_MARK}" 2>/dev/null | sort | tail -1 || true)"
if [[ "$DRY_RUN" -eq 1 && -z "${LOG_DIR}" ]]; then LOG_DIR="${REPO_PATH}/logs/<timestamp>-${TRAIN_CONFIG}"; fi
[[ -n "${LOG_DIR}" ]] || { echo "ERROR: could not resolve training log dir" >&2; exit 1; }
echo "log dir: ${LOG_DIR}"

# 2) Resolve the final checkpoint and convert it (sft2new).
EXP_DIR="$(find "${LOG_DIR}" -maxdepth 1 -mindepth 1 -type d ! -name tensorboard 2>/dev/null | sort | tail -1 || true)"
[[ -n "${EXP_DIR}" || "$DRY_RUN" -eq 1 ]] || { echo "ERROR: no experiment dir under ${LOG_DIR}" >&2; exit 1; }
EXP_DIR="${EXP_DIR:-${LOG_DIR}/<experiment>}"
FINAL_CKPT="$(find "${EXP_DIR}/checkpoints" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)"
FINAL_CKPT="${FINAL_CKPT:-${EXP_DIR}/checkpoints/global_step_<N>}"
CONVERTED_DIR="${EXP_DIR}/pi05_sft_pytorch_new"
OUTPUT_NORM_STATS="${CONVERTED_DIR}/${ASSET_ID}/norm_stats.json"

echo "=== [2/3] Convert (sft2new): ${FINAL_CKPT} -> ${CONVERTED_DIR} ==="
run "${PYBIN}" -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
    --ckpt "${FINAL_CKPT}/actor" \
    --input-norm-stats "${INPUT_NORM_STATS}" \
    --output-model "${CONVERTED_DIR}" \
    --output-norm-stats "${OUTPUT_NORM_STATS}"

# 3) Eval the converted checkpoint. Point model_path + assets at the converted
#    dir so eval resolves {assets_dir}/{asset_id}/norm_stats.json under it.
echo "=== [3/3] Eval: ${EVAL_CONFIG} ==="
EVAL_OVERRIDES=(
    "rollout.model.model_path=${CONVERTED_DIR}"
    "actor.model.openpi.assets_dir=${CONVERTED_DIR}"
    "actor.model.openpi.asset_id=${ASSET_ID}"
    "${EXTRA_OVERRIDES[@]}"
)
if [[ "$FINAL_256" -eq 1 ]]; then
    # Current config uses 8 eval envs; 32 rollout epochs gives 256 trajectories.
    EVAL_OVERRIDES+=("algorithm.eval_rollout_epoch=32")
fi
run bash "${REPO_PATH}/examples/embodiment/eval_embodiment.sh" "${EVAL_CONFIG}" \
    "${EVAL_OVERRIDES[@]}"

echo "=== done: train -> convert -> eval complete (converted: ${CONVERTED_DIR}) ==="
