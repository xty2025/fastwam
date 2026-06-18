#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TASK="${TASK:-navsim_uncond_front_192x352_1e-4}"
CKPT="${CKPT:-${PROJECT_ROOT}/runs/navsim_uncond_front_192x352_1e-4/2026-06-07_20-41-46/checkpoints/weights/step_049000.pt}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-step_049000_navtest}"
SPLIT="${SPLIT:-navtest}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-27890}"

NAVSIM_ROOT="${NAVSIM_ROOT:-${PROJECT_ROOT}/data/navsim}"
FALLBACK_FASTWAM_ROOT="${FALLBACK_FASTWAM_ROOT:-/mnt/zhaozc_workspace/project/FastWAM}"

if [[ -d "${PROJECT_ROOT}/data/metric_cache_${SPLIT}" ]]; then
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${PROJECT_ROOT}/data/metric_cache_${SPLIT}}"
else
  METRIC_CACHE_PATH="${METRIC_CACHE_PATH:-${FALLBACK_FASTWAM_ROOT}/data/metric_cache_${SPLIT}}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/evaluate_results/navsim/${TASK}/${EXPERIMENT_NAME}/${TIMESTAMP}}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/navsim:${PROJECT_ROOT}:${PYTHONPATH:-}"
export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${NAVSIM_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/evaluate_results}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}/navsim}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${NAVSIM_ROOT}}"

case "${SPLIT}" in
  navtest)
    export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_ROOT}/navsim_logs/test}"
    export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${NAVSIM_ROOT}/sensor_blobs/test}"
    ;;
  navtrain|train)
    export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_ROOT}/navsim_logs/trainval}"
    export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${NAVSIM_ROOT}/sensor_blobs/trainval_all}"
    ;;
  navsim|navval|val)
    export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${NAVSIM_ROOT}/navsim_logs/trainval}"
    export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${NAVSIM_ROOT}/sensor_blobs/trainval_all}"
    ;;
  *)
    echo "Unsupported SPLIT=${SPLIT}. Expected navtest, navtrain, or navsim." >&2
    exit 2
    ;;
esac

echo "[eval] project=${PROJECT_ROOT}"
echo "[eval] task=${TASK}"
echo "[eval] ckpt=${CKPT}"
echo "[eval] split=${SPLIT}"
echo "[eval] output=${OUTPUT_DIR}"
echo "[eval] metric_cache=${METRIC_CACHE_PATH}"

cd "${PROJECT_ROOT}"

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes 1 \
  --master-port "${MASTER_PORT}" \
  experiments/navsim/eval_navsim.py \
  "task=${TASK}" \
  "ckpt=${CKPT}" \
  "experiment_name=${EXPERIMENT_NAME}" \
  "EVALUATION.split=${SPLIT}" \
  "EVALUATION.output_dir=${OUTPUT_DIR}" \
  "EVALUATION.save_videos=${SAVE_VIDEOS:-false}" \
  ${MAX_SAMPLES:+EVALUATION.max_samples=${MAX_SAMPLES}}

python "${NAVSIM_DEVKIT_ROOT}/navsim/planning/script/run_pdm_score.py" \
  "train_test_split=${SPLIT}" \
  "metric_cache_path=${METRIC_CACHE_PATH}" \
  "agent=npy_trajectory_agent" \
  "agent.pred_actions_path=${OUTPUT_DIR}/pred_actions" \
  "experiment_name=${EXPERIMENT_NAME}_pdm" \
  "output_dir=${OUTPUT_DIR}/pdm_score"

echo "[done] eval output: ${OUTPUT_DIR}"
echo "[done] pdm csv dir: ${OUTPUT_DIR}/pdm_score"
