#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/absolute/path/to/navsim_depth_training_root}"

if [[ -d "${DATA_ROOT}/_meta" || -d "${DATA_ROOT}/metadata" ]]; then
  export NAVSIM_DEPTH_ROOT="${DATA_ROOT}"
elif [[ -d "${DATA_ROOT}/depthnav" ]]; then
  export NAVSIM_DEPTH_ROOT="${DATA_ROOT}/depthnav"
elif [[ -d "${DATA_ROOT}/depth" ]]; then
  export NAVSIM_DEPTH_ROOT="${DATA_ROOT}/depth"
else
  echo "DepthNav root not found. Expected ${DATA_ROOT}/depthnav, ${DATA_ROOT}/depth, or metadata/_meta directly under DATA_ROOT." >&2
  exit 2
fi

export NAVSIM_LOG_PATH="${DATA_ROOT}/navsim_logs"
export NAVSIM_SENSOR_BLOBS_PATH="${DATA_ROOT}/navsim_sensor_blobs"
export NAVSIM_TEXT_EMBED_CACHE="${DATA_ROOT}/text_embeds_cache"
export NAVSIM_STATS_PATH="${DATA_ROOT}/navsim_dataset_stats.json"

for required_path in "${NAVSIM_LOG_PATH}" "${NAVSIM_SENSOR_BLOBS_PATH}" "${NAVSIM_DEPTH_ROOT}"; do
  [[ -e "${required_path}" ]] || {
    echo "Required dataset path is missing: ${required_path}" >&2
    exit 2
  }
done

mkdir -p "${NAVSIM_TEXT_EMBED_CACHE}"
