#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
FOLD="${FOLD:?FOLD must be fold1/fold2/fold3}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-200}"
WEAR_AGG="${WEAR_AGG:-mean}"
SPLIT_RATIO="${SPLIT_RATIO:-1.0}"
TIME_GAP="${TIME_GAP:-0}"
NO_VAL_FIXED_EPOCHS="${NO_VAL_FIXED_EPOCHS:-1}"

case "${FOLD}" in
  fold1) TRAIN_RUNS="c1,c4"; TEST_RUNS="c6" ;;
  fold2) TRAIN_RUNS="c4,c6"; TEST_RUNS="c1" ;;
  fold3) TRAIN_RUNS="c1,c6"; TEST_RUNS="c4" ;;
  *) echo "Unknown FOLD=${FOLD}" >&2; exit 1 ;;
esac

VARIANT=rms7ptpwav \
FOLD="${FOLD}" \
GPU_ID="${GPU_ID}" \
RESULTS_SUBDIR="20260408_r7p7w_${FOLD}" \
PYTHON_BIN="${PYTHON_BIN}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
PATIENCE=100 \
WEAR_AGG="${WEAR_AGG}" \
SPLIT_RATIO="${SPLIT_RATIO}" \
TIME_GAP="${TIME_GAP}" \
NO_VAL_FIXED_EPOCHS="${NO_VAL_FIXED_EPOCHS}" \
MODEL_ID_EXTRA_SUFFIX="r7p7w_${FOLD}" \
bash "${PROJECT_ROOT}/paper_exec/scripts/run_clean_timer_fold.sh"

CKPT="$(find "${PROJECT_ROOT}/checkpoints" -path "*PHM_${TRAIN_RUNS//,/}_to_${TEST_RUNS}_rms7ptpwav_BaselineClean_${WEAR_AGG}agg_dual_seed2026_e${TRAIN_EPOCHS}_bt96_gpu${GPU_ID}_novalfix_r7p7w_${FOLD}*checkpoint.pth" | head -n 1)"
test -n "${CKPT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_headonly_fold.py" \
  --project_root "${PROJECT_ROOT}" \
  --runtime_root "${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_ptp7_wav7_baseline_clean_${FOLD}_gpu${GPU_ID}" \
  --checkpoint_path "${CKPT}" \
  --results_subdir "20260408_r7p7wh_${FOLD}" \
  --train_runs "${TRAIN_RUNS}" \
  --test_runs "${TEST_RUNS}" \
  --tag "r7p7w_${FOLD}" \
  --wear_agg "${WEAR_AGG}" \
  --split_ratio "${SPLIT_RATIO}" \
  --time_gap "${TIME_GAP}" \
  --n_vars 21

echo "[RMS7PTP7WAV7-BASELINE][OK] ${FOLD} complete"
