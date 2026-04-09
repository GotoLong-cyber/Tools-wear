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
RESULTS_PREFIX="${RESULTS_PREFIX:-20260409_r7ksweep}"
TAG_PREFIX="${TAG_PREFIX:-r7ksweep}"
TMP_CONFIG_DIR="${PROJECT_ROOT}/paper_exec/train_only_param_selection/results_rms7_baseline_e300/k_sweep_configs"

mkdir -p "${TMP_CONFIG_DIR}"

IFS=',' read -r -a K_LIST <<< "${K_VALUES}"

for k in "${K_LIST[@]}"; do
  CONFIG_PATH="${TMP_CONFIG_DIR}/knn_k${k}_b10_q00.json"
  cat > "${CONFIG_PATH}" <<EOF
{
  "k": ${k},
  "beta": 1.0,
  "late_library_quantile": 0.0,
  "selection_metric": "manual outer-test sensitivity only",
  "selection_protocol": "fixed beta=1.0 and late_q=0.0 for k-sensitivity plotting",
  "note": "Outer-test k sweep for Pure RMS7 baseline checkpoint."
}
EOF

  env \
    FOLD="${FOLD}" \
    GPU_ID="${GPU_ID}" \
    PYTHON_BIN="${PYTHON_BIN}" \
    RESULTS_SUBDIR="${RESULTS_PREFIX}_${FOLD}_k${k}" \
    TAG="${TAG_PREFIX}_${FOLD}_k${k}" \
    KNN_CONFIG="${CONFIG_PATH}" \
    WEAR_AGG="${WEAR_AGG}" \
    SPLIT_RATIO="${SPLIT_RATIO}" \
    TIME_GAP="${TIME_GAP}" \
    bash "${PROJECT_ROOT}/paper_exec/scripts/run_rms7_baseline_knn_eval_fold.sh"
done

echo "[RMS7-K-SWEEP][OK] ${FOLD}"
