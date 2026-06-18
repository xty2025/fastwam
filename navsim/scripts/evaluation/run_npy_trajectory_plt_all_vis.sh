#!/usr/bin/env bash

set -euo pipefail

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim/maps"
export NAVSIM_EXP_ROOT="/mnt/zhaozc_workspace/project/FastWAM/evaluate_results"
export NAVSIM_DEVKIT_ROOT="/mnt/zhaozc_workspace/project/FastWAM/navsim"
export OPENSCENE_DATA_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim"

PRED_ACTIONS_PATH="$NAVSIM_EXP_ROOT/navsim/navsim_uncond_front_768x1024_1e-4/step_001000/20260522_110520/pred_actions"
SPLIT="test"
FILTER="navtest"
PLOT_MODE="bev_camera"
OUTPUT_DIR="$NAVSIM_EXP_ROOT/navsim/navsim_uncond_front_768x1024_1e-4/step_001000/20260522_110520/plt_all_vis"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/plt_all_vis.py" \
  --pred-actions-path "$PRED_ACTIONS_PATH" \
  --split "$SPLIT" \
  --filter "$FILTER" \
  --plot-mode "$PLOT_MODE" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
