#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-}"
CKPT_PATH="${2:-}"
EVAL_CSV="${3:-}"
OUTPUT_DIR="${4:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ -z "${CONFIG_PATH}" || -z "${CKPT_PATH}" || -z "${EVAL_CSV}" || -z "${OUTPUT_DIR}" ]]; then
  echo "Usage: bash eval.sh <config_path> <ckpt_path> <eval_csv> <output_dir>"
  exit 1
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/eval_single_csv.py" \
  --config "${CONFIG_PATH}" \
  --ckpt "${CKPT_PATH}" \
  --eval_csv "${EVAL_CSV}" \
  --output_dir "${OUTPUT_DIR}"