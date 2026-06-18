#!/usr/bin/env bash
set -euo pipefail

if [[ "${FASTWAM_IN_LOGIN_SHELL:-0}" != "1" ]]; then
  export FASTWAM_IN_LOGIN_SHELL=1
  printf -v _fastwam_cmd "%q " "$0" "$@"
  exec bash -lc "cd $(printf "%q" "$(pwd)") && ${_fastwam_cmd}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${FASTWAM_PROJECT_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
ENV_ROOT="${FASTWAM_ENV_ROOT:-}"
if [[ -z "${ENV_ROOT}" ]]; then
  for candidate in \
    "/mnt/zhaozc_workspace/envs/fastwam_xty" \
    "/mnt/workspace/envs/fastwam_xty" \
    "/mnt/zczhao_workspace/envs/fastwam_xty"; do
    if [[ -d "${candidate}" ]]; then
      ENV_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ENV_ROOT}" || ! -f "${ENV_ROOT}/bin/activate" ]]; then
  echo "Error: cannot find fastwam_xty env. Set FASTWAM_ENV_ROOT=/path/to/envs/fastwam_xty." >&2
  exit 1
fi

set +u
source "${ENV_ROOT}/bin/activate"
set -u
export PATH="${ENV_ROOT}/bin:${PATH}"
cd "${PROJECT_ROOT}"

if [[ "$(command -v python)" != "${ENV_ROOT}/bin/python" ]]; then
  echo "Error: expected python from ${ENV_ROOT}, got $(command -v python)" >&2
  exit 1
fi
if ! command -v deepspeed >/dev/null 2>&1; then
  echo "Error: deepspeed not found in PATH=${PATH}" >&2
  exit 1
fi
echo "Using python: $(command -v python)"
echo "Using deepspeed: $(command -v deepspeed)"

mkdir -p "${TRITON_CACHE_DIR:-/root/.triton}/autotune"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/navsim:${PROJECT_ROOT}:${PYTHONPATH:-}"
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_ROOT}/checkpoints"

if [[ "${FASTWAM_ENV_CHECK_ONLY:-0}" == "1" ]]; then
  python - <<'PY'
import shutil
import sys
import fastwam  # noqa: F401
import deepspeed  # noqa: F401

print(f"python={sys.executable}")
print(f"deepspeed={shutil.which('deepspeed')}")
print("fastwam_env_check_ok")
PY
  exit 0
fi

PPU_CUDA_SDK_ROOT="${PPU_CUDA_SDK_ROOT:-/usr/local/PPU_SDK/CUDA_SDK}"
CUDA_LIBRARY_PATHS="${PPU_CUDA_SDK_ROOT}/lib64:${PPU_CUDA_SDK_ROOT}/targets/x86_64-linux/lib"
if [[ -n "${CUDA_HOME:-}" ]]; then
  CUDA_LIBRARY_PATHS="${CUDA_HOME}/lib64:${CUDA_HOME}/targets/x86_64-linux/lib:${CUDA_LIBRARY_PATHS}"
fi
export LD_LIBRARY_PATH="${CUDA_LIBRARY_PATHS}:${LD_LIBRARY_PATH:-}"

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

export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29500}"
TASK_BASENAME="${TASK_BASENAME:-navsim_uncond_front_192x352_xty_dit_16gpu}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}"

python -m accelerate.commands.launch \
  --config_file "${ACCELERATE_CONFIG}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "$(( NUM_MACHINES * NPROC_PER_NODE ))" \
  scripts/train.py \
  "task=${TASK_BASENAME}" \
  "output_dir=./runs/${TASK_BASENAME}/${RUN_ID}" \
  "wandb.name=${TASK_BASENAME}" \
  "$@"
