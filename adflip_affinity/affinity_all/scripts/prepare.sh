#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
AFFINITY_DIR="$(cd "${PROJECT_DIR}/../affinity" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

DATASETS=("$@")
if [ "${#DATASETS[@]}" -eq 0 ]; then
  DATASETS=(
    Shanehsazzadeh2023_trastuzumab_zero_kd
    Warszawski2019_d44_Kd
    Koenig2017_g6_Kd
  )
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_joint_training_data.py" \
  --source_manifest "${AFFINITY_DIR}/outputs/prepared/manifest.csv" \
  --out_data_root "${PROJECT_DIR}/data" \
  --seed 42 \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --datasets "${DATASETS[@]}"
