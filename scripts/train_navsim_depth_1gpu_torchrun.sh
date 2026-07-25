#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/activate_fastwam_depth_env.sh"
source "${ROOT}/scripts/navsim_depth_paths.sh"
"${ROOT}/scripts/prepare_navsim_depth_assets.sh"

TORCHRUN="${TORCHRUN:-$(command -v torchrun)}"
export DIFFSYNTH_MODEL_BASE_PATH="${FASTWAM_ASSET_ROOT:-/data1/zczhao/fastwam-depth-assets}/checkpoints"
ACTION_DIT_CHECKPOINT="${DIFFSYNTH_MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
DEPTH_MODE="${DEPTH_MODE:-future_denoise}"
echo "Effective global batch: 1 per GPU x 1 GPU x 3 accumulation = 3"
echo "Depth experiment mode: ${DEPTH_MODE}"

cd "${ROOT}"
"${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=1 \
  scripts/train.py task=navsim_uncond_front_192x352_depth \
  batch_size=1 gradient_accumulation_steps=3 \
  model.action_dit_pretrained_path="${ACTION_DIT_CHECKPOINT}" \
  model.skip_dit_load_from_pretrain=false \
  model.depth_experiment.mode="${DEPTH_MODE}" \
  "$@"
