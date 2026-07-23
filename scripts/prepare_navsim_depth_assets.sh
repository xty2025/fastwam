#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT}/scripts/activate_fastwam_depth_env.sh"

ASSET_ROOT="${FASTWAM_ASSET_ROOT:-/data1/zczhao/fastwam-depth-assets}"
MODEL_ROOT="${ASSET_ROOT}/checkpoints"
ACTION_DIT_CHECKPOINT="${MODEL_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim_3outdim.pt"

mkdir -p "${MODEL_ROOT}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_ROOT}"
export DIFFSYNTH_DOWNLOAD_SOURCE="${DIFFSYNTH_DOWNLOAD_SOURCE:-modelscope}"
export DIFFSYNTH_SKIP_DOWNLOAD=false

if [[ ! -f "${ACTION_DIT_CHECKPOINT}" ]]; then
  python "${ROOT}/scripts/preprocess_action_dit_backbone.py" \
    --model-config "${ROOT}/configs/model/fastwam_navsim_depth.yaml" \
    --output "${ACTION_DIT_CHECKPOINT}" \
    --device cuda \
    --dtype bfloat16
fi

[[ -s "${ACTION_DIT_CHECKPOINT}" ]] || {
  echo "ActionDiT preparation failed: ${ACTION_DIT_CHECKPOINT}" >&2
  exit 1
}
compgen -G "${MODEL_ROOT}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors" >/dev/null || {
  echo "Missing fully pretrained Wan2.2 DiT under ${MODEL_ROOT}" >&2
  exit 1
}
compgen -G "${MODEL_ROOT}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors" >/dev/null || {
  echo "Missing Wan2.2 VAE under ${MODEL_ROOT}" >&2
  exit 1
}

printf 'FASTWAM_ASSET_ROOT=%s\n' "${ASSET_ROOT}"
printf 'DIFFSYNTH_MODEL_BASE_PATH=%s\n' "${MODEL_ROOT}"
printf 'ACTION_DIT_CHECKPOINT=%s\n' "${ACTION_DIT_CHECKPOINT}"
