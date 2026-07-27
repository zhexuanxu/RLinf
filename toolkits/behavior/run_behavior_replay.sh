#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <control_mode> <state_token> <dataset_root> <activity_instance_id> <episode_index> [hydra_overrides...]" >&2
    exit 2
fi

CONTROL_MODE="$1"
STATE_TOKEN="$2"
DATASET_ROOT="$3"
ACTIVITY_INSTANCE_ID="$4"
EPISODE_INDEX="$5"
shift 5

case "${CONTROL_MODE}" in
    abs_joint|delta_joint|abs_eef|delta_eef) ;;
    *)
        echo "control_mode must be one of: abs_joint, delta_joint, abs_eef, delta_eef" >&2
        exit 2
        ;;
esac

case "${STATE_TOKEN}" in
    none|abs_joint_old|abs_joint|abs_eef) ;;
    *)
        echo "state_token must be one of: none, abs_joint_old, abs_joint, abs_eef" >&2
        exit 2
        ;;
esac

if ! [[ "${ACTIVITY_INSTANCE_ID}" =~ ^[0-9]+$ ]]; then
    echo "activity_instance_id must be a non-negative integer" >&2
    exit 2
fi
if ! [[ "${EPISODE_INDEX}" =~ ^[0-9]+$ ]]; then
    echo "episode_index must be a non-negative integer" >&2
    exit 2
fi
if [ ! -f "${DATASET_ROOT}/meta/info.json" ]; then
    echo "dataset_root has no meta/info.json: ${DATASET_ROOT}" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export RLINF_EVAL_ENTRYPOINT="${REPO_ROOT}/evaluations/behavior/replay_embodied_agent.py"

exec bash "${REPO_ROOT}/evaluations/run_eval.sh" \
    behavior \
    behavior_openpi_pi05_pytorch_replay \
    "rollout.model.openpi.control_mode=${CONTROL_MODE}" \
    "rollout.model.openpi.state_token=${STATE_TOKEN}" \
    "env.eval.replay.dataset_root=${DATASET_ROOT}" \
    "env.eval.replay.activity_instance_id=${ACTIVITY_INSTANCE_ID}" \
    "env.eval.omni_config.task.activity_instance_id=${ACTIVITY_INSTANCE_ID}" \
    "env.eval.replay.episode_index=${EPISODE_INDEX}" \
    "runner.logger.experiment_name=behavior_replay_${CONTROL_MODE}_instance_${ACTIVITY_INSTANCE_ID}_episode_${EPISODE_INDEX}" \
    "$@"
