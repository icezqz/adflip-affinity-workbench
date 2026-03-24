#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SUMMARY_CSV="${1:-${PROJECT_DIR}/outputs/summary/all_tests_summary.csv}"
OUT_DIR="${2:-${PROJECT_DIR}/outputs/summary}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/summarize_all_tests.py" \
  --summary_csv "${SUMMARY_CSV}" \
  --out_dir "${OUT_DIR}"
