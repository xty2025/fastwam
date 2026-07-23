#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MAMBA_ROOT="${MAMBA_ROOT:-/home/zczhao/micromamba}"
MICROMAMBA="${MICROMAMBA:-/home/zczhao/bin/micromamba}"
ENV_NAME="${ENV_NAME:-fastwam-depth-py310-cu118}"

if [[ ! -x "${MICROMAMBA}" ]]; then
  mkdir -p "$(dirname "${MICROMAMBA}")"
  curl -L https://micro.mamba.pm/api/micromamba/linux-64/latest |
    tar -xj -C "$(dirname "${MICROMAMBA}")" --strip-components=1 bin/micromamba
fi

"${MICROMAMBA}" create -y -r "${MAMBA_ROOT}" -n "${ENV_NAME}" -c conda-forge python=3.10 pip
PYTHON="${MAMBA_ROOT}/envs/${ENV_NAME}/bin/python"

"${PYTHON}" -m pip install --upgrade pip
"${PYTHON}" -m pip install \
  torch==2.5.1+cu118 torchvision==0.20.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
"${PYTHON}" -m pip install -e "${ROOT}"
"${PYTHON}" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
