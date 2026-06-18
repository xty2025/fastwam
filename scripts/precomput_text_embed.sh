# python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# torchrun --node=1 --nproc_per_node=8 scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# generate navsim text embeds
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/PPU_SDK/CUDA_SDK}/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-${PROJECT_ROOT}/data/navsim/maps}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-${PROJECT_ROOT}/runs}"
export NAVSIM_DEVKIT_ROOT="${NAVSIM_DEVKIT_ROOT:-${PROJECT_ROOT}}"
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-${PROJECT_ROOT}/data/navsim}"

export NAVSIM_LOG_PATH="${OPENSCENE_DATA_ROOT}/navsim_logs/trainval"
export NAVSIM_SENSOR_BLOBS_PATH="${OPENSCENE_DATA_ROOT}/sensor_blobs/trainval_all"

python scripts/precompute_navsim_text_embeds.py task=navsim_uncond_front_768x1024_1e-4
