#!/usr/bin/env bash
# Source this file; do not execute it: source scripts/activate_fastwam_depth_env.sh

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Use: source ${BASH_SOURCE[0]}" >&2
  return 1 2>/dev/null || exit 1
fi

MAMBA_ROOT="${MAMBA_ROOT:-/home/zczhao/micromamba}"
MICROMAMBA="${MICROMAMBA:-/home/zczhao/bin/micromamba}"
ENV_NAME="${ENV_NAME:-fastwam-depth-py310-cu118}"

[[ -x "${MICROMAMBA}" ]] || { echo "Micromamba not found: ${MICROMAMBA}" >&2; return 1; }
_fastwam_restore_nounset=0
if [[ $- == *u* ]]; then
  _fastwam_restore_nounset=1
  set +u
fi
eval "$("${MICROMAMBA}" shell hook --shell bash --root-prefix "${MAMBA_ROOT}")"
micromamba activate "${ENV_NAME}"
if [[ "${_fastwam_restore_nounset}" -eq 1 ]]; then
  set -u
fi
unset _fastwam_restore_nounset
