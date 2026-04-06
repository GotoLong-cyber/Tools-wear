#!/usr/bin/env bash
set -euo pipefail

# Run clean-protocol 3-fold training sequentially on one GPU.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

VARIANT="${VARIANT:?VARIANT must be baseline,tma,retrieval}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"

case "${VARIANT}" in
  baseline) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_BaselineClean_seqgpu${GPU_ID}}" ;;
  tma) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_TMAClean_seqgpu${GPU_ID}}" ;;
  retrieval) RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_RetrievalBackboneClean_seqgpu${GPU_ID}}" ;;
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

STATUS_FILE="${ROUND_DIR}/seq_status_${VARIANT}_gpu${GPU_ID}.txt"
: > "${STATUS_FILE}"

for FOLD in fold1 fold2 fold3; do
  echo "[START] ${VARIANT} ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
  if PYTHON_BIN="${PYTHON_BIN}" VARIANT="${VARIANT}" FOLD="${FOLD}" GPU_ID="${GPU_ID}" TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
      bash "${SCRIPT_DIR}/run_clean_timer_fold.sh" \
      > "${ROUND_DIR}/master_${VARIANT}_${FOLD}_gpu${GPU_ID}.log" 2>&1; then
    echo "[DONE] ${VARIANT} ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
  else
    echo "[FAIL] ${VARIANT} ${FOLD} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
    exit 1
  fi
done

echo "[ALL_DONE] ${VARIANT} gpu=${GPU_ID}" | tee -a "${STATUS_FILE}"
