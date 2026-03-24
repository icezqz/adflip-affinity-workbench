#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_DIR}/configs/base.yaml}"
EVAL_CSV="${2:-}"
GPU_ID="${3:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/score_zero_shot_nll.py"
  --config "${CONFIG_PATH}"
  --cuda_visible_devices "${GPU_ID}"
)

if [[ -n "${EVAL_CSV}" ]]; then
  CMD+=(--eval_csv "${EVAL_CSV}")
fi

"${CMD[@]}"
