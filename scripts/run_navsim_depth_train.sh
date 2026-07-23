#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zczhao/micromamba/envs/fastwam-depth-py310-cu118/bin/python}"

: "${NAVSIM_LOG_PATH:?Set NAVSIM_LOG_PATH to the NavSim logs directory.}"
: "${NAVSIM_SENSOR_BLOBS_PATH:?Set NAVSIM_SENSOR_BLOBS_PATH to the NavSim sensor blobs directory.}"
: "${NAVSIM_DEPTH_ROOT:?Set NAVSIM_DEPTH_ROOT to the separate relative-depth root.}"
: "${NAVSIM_TEXT_EMBED_CACHE:?Set NAVSIM_TEXT_EMBED_CACHE to the existing T5 embedding cache.}"

cd "${ROOT}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${ROOT}/checkpoints}"

"${PYTHON}" scripts/train.py \
  task=navsim_uncond_front_192x352_depth \
  "$@"
