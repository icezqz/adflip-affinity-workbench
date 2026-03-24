#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_DIR}/configs/base.yaml}"
CKPT_PATH="${2:-}"
EVAL_CSV="${3:-}"
GPU_ID="${4:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -z "${CKPT_PATH}" ]]; then
  echo "Usage: bash eval.sh <config_path> <ckpt_path> [eval_csv] [gpu_id]"
  exit 1
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/eval_affinity_adflip_fewshot.py"
  --config "${CONFIG_PATH}"
  --ckpt "${CKPT_PATH}"
  --cuda_visible_devices "${GPU_ID}"
)

if [[ -n "${EVAL_CSV}" ]]; then
  CMD+=(--eval_csv "${EVAL_CSV}")
fi

"${CMD[@]}"
