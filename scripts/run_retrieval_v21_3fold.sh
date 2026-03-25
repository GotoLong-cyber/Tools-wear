#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260325_RetrievalV21_formal}"
mkdir -p "${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"

CUDA_VISIBLE_DEVICES=0 RESULTS_SUBDIR="${RESULTS_SUBDIR}" bash "${PROJECT_ROOT}/scripts/run_retrieval_v21_fold1.sh" \
  > "${PROJECT_ROOT}/results/${RESULTS_SUBDIR}/master_retrieval_v21_fold1_gpu0.log" 2>&1 < /dev/null &
PID1=$!
CUDA_VISIBLE_DEVICES=1 RESULTS_SUBDIR="${RESULTS_SUBDIR}" bash "${PROJECT_ROOT}/scripts/run_retrieval_v21_fold2.sh" \
  > "${PROJECT_ROOT}/results/${RESULTS_SUBDIR}/master_retrieval_v21_fold2_gpu1.log" 2>&1 < /dev/null &
PID2=$!
CUDA_VISIBLE_DEVICES=2 RESULTS_SUBDIR="${RESULTS_SUBDIR}" bash "${PROJECT_ROOT}/scripts/run_retrieval_v21_fold3.sh" \
  > "${PROJECT_ROOT}/results/${RESULTS_SUBDIR}/master_retrieval_v21_fold3_gpu2.log" 2>&1 < /dev/null &
PID3=$!

echo "fold1=${PID1}"
echo "fold2=${PID2}"
echo "fold3=${PID3}"
