#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
FOLD="${FOLD:?FOLD must be fold1/fold2/fold3}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260409_r7k300_${FOLD}}"
TAG="${TAG:-r7k300_${FOLD}}"
KNN_CONFIG="${KNN_CONFIG:-paper_exec/train_only_param_selection/results_rms7_e300/selected_knn_config_inner_loso.json}"
WEAR_AGG="${WEAR_AGG:-mean}"
SPLIT_RATIO="${SPLIT_RATIO:-1.0}"
TIME_GAP="${TIME_GAP:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

case "${FOLD}" in
  fold1)
    RUNTIME_ROOT="dataset/passlevel_tree_select/runtime_rms7_retrieval_clean_fold1_gpu0"
    CHECKPOINT_PATH="checkpoints/forecast_PHM_c1c4_to_c6_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu0_novalfix_r7r300_fold1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"
    TRAIN_RUNS="c1,c4"
    TEST_RUNS="c6"
    ;;
  fold2)
    RUNTIME_ROOT="dataset/passlevel_tree_select/runtime_rms7_retrieval_clean_fold2_gpu1"
    CHECKPOINT_PATH="checkpoints/forecast_PHM_c4c6_to_c1_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu1_novalfix_r7r300_fold2_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"
    TRAIN_RUNS="c4,c6"
    TEST_RUNS="c1"
    ;;
  fold3)
    RUNTIME_ROOT="dataset/passlevel_tree_select/runtime_rms7_retrieval_clean_fold3_gpu2"
    CHECKPOINT_PATH="checkpoints/forecast_PHM_c1c6_to_c4_rms7_TMAClean_meanagg_dual_seed2026_e300_bt96_gpu2_novalfix_r7r300_fold3_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"
    TRAIN_RUNS="c1,c6"
    TEST_RUNS="c4"
    ;;
  *)
    echo "Unknown FOLD=${FOLD}" >&2
    exit 1
    ;;
esac

ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
mkdir -p "${ROUND_DIR}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_fold1_knn_delta_retrieval.py"
  --project_root "${PROJECT_ROOT}"
  --runtime_root "${RUNTIME_ROOT}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --results_subdir "${RESULTS_SUBDIR}"
  --train_runs "${TRAIN_RUNS}"
  --test_runs "${TEST_RUNS}"
  --tag "${TAG}"
  --wear_agg "${WEAR_AGG}"
  --split_ratio "${SPLIT_RATIO}"
  --time_gap "${TIME_GAP}"
  --n_vars 7
  --knn_config "${KNN_CONFIG}"
  --blend_mode delta_residual
  --library_wear_threshold_um 0
  --gate_mode none
)

LOG_FILE="${ROUND_DIR}/master_rms7_retrieval_eval_${FOLD}_gpu${GPU_ID}.log"
CMD_FILE="${ROUND_DIR}/cmd_rms7_retrieval_eval_${FOLD}_gpu${GPU_ID}.txt"
: > "${CMD_FILE}"
for arg in "${CMD[@]}"; do
  printf '%q ' "${arg}" >> "${CMD_FILE}"
done
printf '\n' >> "${CMD_FILE}"
cat "${CMD_FILE}"

"${CMD[@]}" > "${LOG_FILE}" 2>&1

echo "[RMS7-RETRIEVAL-EVAL][OK] ${FOLD}"
