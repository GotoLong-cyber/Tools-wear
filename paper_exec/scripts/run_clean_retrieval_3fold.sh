#!/usr/bin/env bash
set -euo pipefail

# Sequential 3-fold clean formal retrieval inference.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-3}"
PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_RetrievalV21_cleanformal}"

ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
mkdir -p "${ROUND_DIR}"
STATUS_FILE="${ROUND_DIR}/seq_status_clean_retrieval_gpu${GPU_ID}.txt"
: > "${STATUS_FILE}"

for FOLD in fold1 fold2 fold3; do
  echo "[START] clean retrieval ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
  if PYTHON_BIN="${PYTHON_BIN}" GPU_ID="${GPU_ID}" FOLD="${FOLD}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
      bash "${SCRIPT_DIR}/run_clean_retrieval_fold.sh" \
      > "${ROUND_DIR}/launcher_clean_retrieval_${FOLD}_gpu${GPU_ID}.log" 2>&1; then
    echo "[DONE] clean retrieval ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
  else
    echo "[FAIL] clean retrieval ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
    exit 1
  fi
done

echo "[ALL_DONE] clean retrieval gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
