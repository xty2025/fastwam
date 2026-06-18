#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
export NUPLAN_MAPS_ROOT="${PROJECT_ROOT}/data/navsim/maps"
export NAVSIM_EXP_ROOT="${PROJECT_ROOT}/evaluate_results"
export NAVSIM_DEVKIT_ROOT="${PROJECT_ROOT}"
export OPENSCENE_DATA_ROOT="${PROJECT_ROOT}/data/navsim"
export NAVSIM_LOG_PATH="${PROJECT_ROOT}/data/navsim/navsim_logs/trainval"
export NAVSIM_SENSOR_BLOBS_PATH="${PROJECT_ROOT}/data/navsim/sensor_blobs/trainval_all"

export CUDA_VISIBLE_DEVICES=0,1
# cd "${PROJECT_ROOT}"

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --master-port 27890 \
  experiments/navsim/eval_navsim.py \
  task=navsim_uncond_front_384x672_1e-4 \
  ckpt=./runs/navsim_uncond_front_384x672_1e-4/2026-05-24_15-20-27/checkpoints/weights/step_033000.pt \
  experiment_name=step_033000

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --master-port 27890 \
  experiments/navsim/eval_navsim.py \
  task=navsim_uncond_front_384x672_1e-4 \
  ckpt=./runs/navsim_uncond_front_384x672_1e-4/2026-05-24_15-20-27/checkpoints/weights/step_044000.pt \
  experiment_name=step_044000

torchrun \
  --nproc_per_node 2 \
  --nnodes 1 \
  --master-port 27890 \
  experiments/navsim/eval_navsim.py \
  task=navsim_uncond_front_384x672_1e-4 \
  ckpt=./runs/navsim_uncond_front_384x672_1e-4/2026-05-24_15-20-27/checkpoints/weights/step_044400.pt \
  experiment_name=step_044400