#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/zczhao/micromamba/envs/fastwam-depth-py310-cu118/bin/python}"
DEPTH_ROOT="${DEPTH_ROOT:-${ROOT}/data/mock_relative_depth}"

"${PYTHON}" "${ROOT}/scripts/make_mock_depth_data.py" --output "${DEPTH_ROOT}"
"${PYTHON}" "${ROOT}/scripts/smoke_depth_epoch.py" --depth-root "${DEPTH_ROOT}" --epochs 1
