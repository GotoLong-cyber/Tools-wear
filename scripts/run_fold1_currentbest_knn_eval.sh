#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_fold1_knn_retrieval.py" \
  --project_root "${PROJECT_ROOT}" \
  --results_subdir "${RESULTS_SUBDIR:-20260324_KNNFold1}" \
  --ks "${KNN_KS:-5,10}" \
  --betas "${KNN_BETAS:-0.3,0.5}"
