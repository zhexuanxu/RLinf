#!/usr/bin/env bash

# Convert BEHAVIOR SFT data with one generic command. Example:
#
# bash toolkits/behavior/convert_openpi_control_mode.sh \
#   --source-dataset-root /mnt/public/xzxuan/data/2025-challenge-demos \
#   --output-dataset-root /mnt/public/xzxuan/tmp/behavior-delta-joint \
#   --control-mode delta_joint \
#   --tasks turning_on_radio

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python}" \
  "${REPO_ROOT}/rlinf/data/datasets/openpi_pytorch/behavior/convert_control_mode.py" \
  "$@"
