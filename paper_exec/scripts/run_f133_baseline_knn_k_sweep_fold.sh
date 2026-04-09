#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
FOLD="${FOLD:?FOLD must be fold1/fold2/fold3}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
WEAR_AGG="${WEAR_AGG:-mean}"
SPLIT_RATIO="${SPLIT_RATIO:-1.0}"
TIME_GAP="${TIME_GAP:-0}"
K_VALUES="${K_VALUES:-1,3,5,7,9,11,15}"
RESULTS_PREFIX="${RESULTS_PREFIX:-20260410_f133ksweep}"
TAG_PREFIX="${TAG_PREFIX:-f133ksweep}"

CONFIG_DIR="${PROJECT_ROOT}/paper_exec/train_only_param_selection/results_f133_baseline_e300/k_sweep_configs"
mkdir -p "${CONFIG_DIR}"

IFS=',' read -r -a KS <<< "${K_VALUES}"

for K in "${KS[@]}"; do
  K="$(echo "${K}" | xargs)"
  CONFIG_PATH="${CONFIG_DIR}/knn_k${K}_b10_q00.json"
  cat > "${CONFIG_PATH}" <<JSON
{
  "k": ${K},
  "beta": 1.0,
  "late_q": 0.0
}
JSON

  RESULTS_SUBDIR="${RESULTS_PREFIX}_${FOLD}_k${K}"
  TAG="${TAG_PREFIX}_${FOLD}_k${K}"

  FOLD="${FOLD}" \
  GPU_ID="${GPU_ID}" \
  RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  TAG="${TAG}" \
  KNN_CONFIG="${CONFIG_PATH}" \
  WEAR_AGG="${WEAR_AGG}" \
  SPLIT_RATIO="${SPLIT_RATIO}" \
  TIME_GAP="${TIME_GAP}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  bash "${PROJECT_ROOT}/paper_exec/scripts/run_f133_baseline_knn_eval_fold.sh"
done

echo "[F133-K-SWEEP][OK] ${FOLD}"
