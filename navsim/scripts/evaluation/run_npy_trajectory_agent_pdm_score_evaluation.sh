#!/usr/bin/env bash

set -euo pipefail

TRAIN_TEST_SPLIT=navtest

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim/maps"
export NAVSIM_EXP_ROOT="/mnt/zhaozc_workspace/project/FastWAM/evaluate_results"
export NAVSIM_DEVKIT_ROOT="/mnt/zhaozc_workspace/project/FastWAM/navsim"
export OPENSCENE_DATA_ROOT="/mnt/zhaozc_workspace/project/FastWAM/data/navsim"

python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py" \
train_test_split="$TRAIN_TEST_SPLIT" \
metric_cache_path="/mnt/zhaozc_workspace/project/FastWAM/data/metric_cache_navtest" \
agent=npy_trajectory_agent \
agent.pred_actions_path="$NAVSIM_EXP_ROOT/navsim/navsim_uncond_front_384x672_1e-4/step_044400/20260526_121243/pred_actions" \
experiment_name=navsim_uncond_front_384x672_1e-4_step_044400