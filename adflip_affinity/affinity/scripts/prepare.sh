#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_ROOT="$(cd "${PROJECT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/prepare_affinity_datasets.py" \
  --data_root "${WORK_ROOT}/data" \
  --structure_root "${WORK_ROOT}/structure" \
  --out_root "${PROJECT_DIR}/outputs/prepared" \
  --seed 42
