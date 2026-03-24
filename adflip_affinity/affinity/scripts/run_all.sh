#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${1:-${PROJECT_DIR}/configs/base.yaml}"
GPU_ID="${2:-1}"
TRAIN_SCRIPT="${PROJECT_DIR}/train_affinity_adflip_fewshot.py"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" --config "${CONFIG_PATH}"
