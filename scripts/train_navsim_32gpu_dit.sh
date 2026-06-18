#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${FASTWAM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_ROOT="${FASTWAM_ENV_ROOT:-/mnt/zhaozc_workspace/envs/fastwam_xty}"

if [[ ! -f "${ENV_ROOT}/bin/activate" ]]; then
  echo "Error: cannot find env: ${ENV_ROOT}" >&2
  exit 1
fi

set +u
source "${ENV_ROOT}/bin/activate"
set -u
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/navsim:${PROJECT_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_ROOT}/checkpoints"

export NUPLAN_MAP_VERSION="${NUPLAN_MAP_VERSION:-nuplan-maps-v1.0}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${OPENSCENE_DATA_ROOT}/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/runs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}}"
export NAVSIM_LOG_PATH="${NAVSIM_LOG_PATH:-${OPENSCENE_DATA_ROOT}/navsim_logs/trainval}"
export NAVSIM_SENSOR_BLOBS_PATH="${NAVSIM_SENSOR_BLOBS_PATH:-${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval_all}"
export NAVSIM_TEXT_EMBED_CACHE="${NAVSIM_TEXT_EMBED_CACHE:-${PROJECT_ROOT}/data/text_embeds_cache/navsim}"
export NAVSIM_STATS_PATH="${NAVSIM_STATS_PATH:-${PROJECT_ROOT}/data/navsim_dataset_stats.json}"
export NAVSIM_FLOW_CACHE_DIR="${NAVSIM_FLOW_CACHE_DIR:-/mnt/zhaozc_workspace/data/fastwam_xty_dit_flow_dataset_motus_rigid}"
export NAVSIM_FLOW_RGB_CACHE_DIR="${NAVSIM_FLOW_RGB_CACHE_DIR:-/mnt/zhaozc_workspace/data/fastwam_xty_dpflow_rgb_dataset_official_192x352}"

export ACCELERATE_USE_DEEPSPEED="${ACCELERATE_USE_DEEPSPEED:-true}"
export ACCELERATE_DEEPSPEED_CONFIG_FILE="${ACCELERATE_DEEPSPEED_CONFIG_FILE:-${PROJECT_ROOT}/scripts/ds_configs/ds_zero1_config.json}"

NNODES="${NNODES:-${FASTWAM_NUM_MACHINES:-2}}"
NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
MASTER_PORT="${MASTER_PORT:-29500}"
TASK_BASENAME="${TASK_BASENAME:-navsim_uncond_front_192x352_xty_dit_32gpu}"
RUN_ID="${RUN_ID:-navsim_32gpu_001}"
HOST_NAME="$(hostname)"

if [[ -z "${NODE_RANK:-}" ]]; then
  if [[ "${HOST_NAME}" =~ -master-([0-9]+)$ ]]; then
    NODE_RANK="${BASH_REMATCH[1]}"
  elif [[ "${HOST_NAME}" =~ -worker-([0-9]+)$ ]]; then
    NODE_RANK="$(( BASH_REMATCH[1] + 1 ))"
  else
    NODE_RANK="0"
  fi
fi

if [[ -z "${MASTER_ADDR:-}" ]]; then
  if [[ "${NNODES}" == "1" ]]; then
    MASTER_ADDR="127.0.0.1"
  elif [[ "${HOST_NAME}" =~ ^(.+)-worker-[0-9]+$ ]]; then
    MASTER_ADDR="${BASH_REMATCH[1]}-master-0"
  else
    MASTER_ADDR="${HOST_NAME}"
  fi
fi

python -m torch.distributed.run \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  scripts/train.py \
  "task=${TASK_BASENAME}" \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "$@"
