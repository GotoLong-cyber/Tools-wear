#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
FOLD="${FOLD:?FOLD must be fold1/fold2/fold3}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-300}"
WEAR_AGG="${WEAR_AGG:-mean}"
SPLIT_RATIO="${SPLIT_RATIO:-1.0}"
TIME_GAP="${TIME_GAP:-0}"
NO_VAL_FIXED_EPOCHS="${NO_VAL_FIXED_EPOCHS:-1}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260409_r7r300_${FOLD}}"
MODEL_TAG="${MODEL_TAG:-r7r300_${FOLD}}"

case "${FOLD}" in
  fold1) TRAIN_RUNS="c1,c4"; TEST_RUNS="c6" ;;
  fold2) TRAIN_RUNS="c4,c6"; TEST_RUNS="c1" ;;
  fold3) TRAIN_RUNS="c1,c6"; TEST_RUNS="c4" ;;
  *) echo "Unknown FOLD=${FOLD}" >&2; exit 1 ;;
esac

VARIANT=rms7_retrieval \
FOLD="${FOLD}" \
GPU_ID="${GPU_ID}" \
RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
PATIENCE=100 \
WEAR_AGG="${WEAR_AGG}" \
SPLIT_RATIO="${SPLIT_RATIO}" \
TIME_GAP="${TIME_GAP}" \
NO_VAL_FIXED_EPOCHS="${NO_VAL_FIXED_EPOCHS}" \
MODEL_ID_EXTRA_SUFFIX="${MODEL_TAG}" \
bash "${PROJECT_ROOT}/paper_exec/scripts/run_clean_timer_fold.sh"

echo "[RMS7-RETRIEVAL][OK] ${FOLD} e${TRAIN_EPOCHS} backbone complete"
