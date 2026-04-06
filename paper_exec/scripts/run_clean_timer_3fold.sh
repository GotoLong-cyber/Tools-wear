#!/usr/bin/env bash
set -euo pipefail

# Launch clean-protocol 3-fold training in parallel on GPU 0/1/2.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANT="${VARIANT:?VARIANT must be baseline,tma,retrieval}"
PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"

case "${VARIANT}" in
  baseline) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_BaselineClean_3fold}" ;;
  tma) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_TMAClean_3fold}" ;;
  retrieval) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_RetrievalBackboneClean_3fold}" ;;
  *) echo "Unknown VARIANT=${VARIANT}" >&2; exit 1 ;;
esac

ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
mkdir -p "${ROUND_DIR}"

if [[ "${VARIANT}" == "retrieval" ]]; then
  "${PYTHON_BIN}" "${PROJECT_ROOT}/feature_extraction/build_rms7_feat4_plus_se1.py" \
    --project_root "${PROJECT_ROOT}" \
    --runs c1 c4 c6 \
    --out_dir "dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_se1" \
    --channel_idx 0
  export SKIP_FEATURE_BUILD=1
fi

PYTHON_BIN="${PYTHON_BIN}" VARIANT="${VARIANT}" FOLD=fold1 GPU_ID=0 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${SCRIPT_DIR}/run_clean_timer_fold.sh" > "${ROUND_DIR}/master_${VARIANT}_fold1_gpu0.log" 2>&1 &
PID1=$!
PYTHON_BIN="${PYTHON_BIN}" VARIANT="${VARIANT}" FOLD=fold2 GPU_ID=1 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${SCRIPT_DIR}/run_clean_timer_fold.sh" > "${ROUND_DIR}/master_${VARIANT}_fold2_gpu1.log" 2>&1 &
PID2=$!
PYTHON_BIN="${PYTHON_BIN}" VARIANT="${VARIANT}" FOLD=fold3 GPU_ID=2 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${SCRIPT_DIR}/run_clean_timer_fold.sh" > "${ROUND_DIR}/master_${VARIANT}_fold3_gpu2.log" 2>&1 &
PID3=$!

echo "VARIANT=${VARIANT}"
echo "RESULTS_SUBDIR=${RESULTS_SUBDIR}"
echo "PIDS: fold1=${PID1} fold2=${PID2} fold3=${PID3}"

wait "${PID1}"; S1=$?
wait "${PID2}"; S2=$?
wait "${PID3}"; S3=$?

echo "EXIT: fold1=${S1} fold2=${S2} fold3=${S3}"
