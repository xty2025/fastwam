#!/usr/bin/env bash

# 多个要清理的目标目录
TARGET_DIRS=(
  "/mnt/zhaozc_workspace/project/FastWAM/runs/navsim_uncond_front_384x672_1e-4/2026-05-24_15-20-27/checkpoints/state"
  "/mnt/zhaozc_workspace/project/FastWAM/runs/navsim_uncond_front_384x512_1e-4/2026-05-22_12-58-14/checkpoints/state"
)

# 每隔多少秒执行一次：3600 秒 = 1 小时
INTERVAL=3600

while true; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Start cleaning..."

  for TARGET_DIR in "${TARGET_DIRS[@]}"; do
    echo "Cleaning step_* directories under: $TARGET_DIR"

    if [ -d "$TARGET_DIR" ]; then
      find "$TARGET_DIR" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name "step_*" \
        -exec rm -rf {} +
        # -print
        

      echo "Done: $TARGET_DIR"
    else
      echo "WARNING: directory not found: $TARGET_DIR"
    fi
  done

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] All done. Sleep ${INTERVAL}s..."
  sleep "$INTERVAL"
done