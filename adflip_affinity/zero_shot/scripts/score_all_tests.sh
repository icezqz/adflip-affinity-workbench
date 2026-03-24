#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AFFINITY_ALL_DIR="$(cd "${PROJECT_DIR}/../affinity_all" && pwd)"
CONFIG_PATH="${1:-${PROJECT_DIR}/configs/base.yaml}"
GPU_ID="${2:-0}"
MANIFEST_PATH="${3:-${AFFINITY_ALL_DIR}/data/prepared/manifest.csv}"
EVAL_SPLIT="${4:-test}"
PDB_COL="${5:-pdb_abs}"
Y_COL="${6:-label_neglog_m}"
LABEL_MODE="${7:-raw}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

CMD=(
  "${PYTHON_BIN}" "${PROJECT_DIR}/score_all_tests.py"
  --config "${CONFIG_PATH}"
  --eval_split "${EVAL_SPLIT}"
  --cuda_visible_devices "${GPU_ID}"
)

CMD+=(--manifest "${MANIFEST_PATH}")
CMD+=(--pdb_col "${PDB_COL}")
CMD+=(--y_col "${Y_COL}")
CMD+=(--label_mode "${LABEL_MODE}")

"${CMD[@]}"
