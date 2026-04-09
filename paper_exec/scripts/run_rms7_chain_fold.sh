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

run_variant() {
  local variant="$1"
  local results_subdir="$2"
  local extra_suffix="$3"
  VARIANT="${variant}" \
  FOLD="${FOLD}" \
  GPU_ID="${GPU_ID}" \
  RESULTS_SUBDIR="${results_subdir}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
  PATIENCE=100 \
  WEAR_AGG="${WEAR_AGG}" \
  SPLIT_RATIO="${SPLIT_RATIO}" \
  TIME_GAP="${TIME_GAP}" \
  NO_VAL_FIXED_EPOCHS="${NO_VAL_FIXED_EPOCHS}" \
  MODEL_ID_EXTRA_SUFFIX="${extra_suffix}" \
  bash "${PROJECT_ROOT}/paper_exec/scripts/run_clean_timer_fold.sh"
}

find_ckpt() {
  local stem="$1"
  find "${PROJECT_ROOT}/checkpoints" -path "*${stem}*checkpoint.pth" | head -n 1
}

eval_headonly() {
  local runtime_root="$1"
  local ckpt="$2"
  local results_subdir="$3"
  local tag="$4"
  local n_vars="$5"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_headonly_fold.py" \
    --project_root "${PROJECT_ROOT}" \
    --runtime_root "${runtime_root}" \
    --checkpoint_path "${ckpt}" \
    --results_subdir "${results_subdir}" \
    --train_runs "${TRAIN_RUNS}" \
    --test_runs "${TEST_RUNS}" \
    --tag "${tag}" \
    --wear_agg "${WEAR_AGG}" \
    --split_ratio "${SPLIT_RATIO}" \
    --time_gap "${TIME_GAP}" \
    --n_vars "${n_vars}"
}

eval_knn() {
  local runtime_root="$1"
  local ckpt="$2"
  local results_subdir="$3"
  local tag="$4"
  local n_vars="$5"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_fold1_knn_delta_retrieval.py" \
    --project_root "${PROJECT_ROOT}" \
    --runtime_root "${runtime_root}" \
    --checkpoint_path "${ckpt}" \
    --results_subdir "${results_subdir}" \
    --train_runs "${TRAIN_RUNS}" \
    --test_runs "${TEST_RUNS}" \
    --tag "${tag}" \
    --wear_agg "${WEAR_AGG}" \
    --split_ratio "${SPLIT_RATIO}" \
    --time_gap "${TIME_GAP}" \
    --n_vars "${n_vars}"
}

run_variant rms7 "20260408_r7b_${FOLD}" "r7b_${FOLD}"
CKPT="$(find_ckpt "PHM_${TRAIN_RUNS//,/}_to_${TEST_RUNS}_rms7_BaselineClean_${WEAR_AGG}agg_dual_seed2026_e${TRAIN_EPOCHS}_bt96_gpu${GPU_ID}_novalfix_r7b_${FOLD}")"
test -n "${CKPT}"
eval_headonly \
  "${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_baseline_clean_${FOLD}_gpu${GPU_ID}" \
  "${CKPT}" \
  "20260408_r7bh_${FOLD}" \
  "r7b_${FOLD}" \
  7

run_variant rms7_tma "20260408_r7t_${FOLD}" "r7t_${FOLD}"
CKPT="$(find_ckpt "PHM_${TRAIN_RUNS//,/}_to_${TEST_RUNS}_rms7_TMAClean_${WEAR_AGG}agg_dual_seed2026_e${TRAIN_EPOCHS}_bt96_gpu${GPU_ID}_novalfix_r7t_${FOLD}")"
test -n "${CKPT}"
eval_headonly \
  "${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_tma_clean_${FOLD}_gpu${GPU_ID}" \
  "${CKPT}" \
  "20260408_r7th_${FOLD}" \
  "r7t_${FOLD}" \
  7

run_variant rms7_retrieval "20260408_r7r_${FOLD}" "r7r_${FOLD}"
CKPT="$(find_ckpt "PHM_${TRAIN_RUNS//,/}_to_${TEST_RUNS}_rms7_TMAClean_${WEAR_AGG}agg_dual_seed2026_e${TRAIN_EPOCHS}_bt96_gpu${GPU_ID}_novalfix_r7r_${FOLD}")"
test -n "${CKPT}"
eval_knn \
  "${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_retrieval_clean_${FOLD}_gpu${GPU_ID}" \
  "${CKPT}" \
  "20260408_r7k_${FOLD}" \
  "r7k_${FOLD}" \
  7

echo "[RMS7-CHAIN][OK] ${FOLD} complete"
