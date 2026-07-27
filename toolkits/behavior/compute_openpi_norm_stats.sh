#!/usr/bin/env bash

# Compute a norm-stats asset using the same generic fields as SFT YAML. Example:
#
# bash toolkits/behavior/compute_openpi_norm_stats.sh \
#   --behavior-dataset-root /mnt/public/xzxuan/tmp/behavior-delta-joint \
#   --assets-dir /mnt/public/xzxuan/tmp/behavior-norm-stats \
#   --asset-id turning_on_radio_delta_joint \
#   --control-mode delta_joint \
#   --state-token abs_joint

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

exec "${PYTHON:-python}" \
  "${REPO_ROOT}/rlinf/data/datasets/openpi_pytorch/behavior/compute_norm_stats.py" \
  "$@"
