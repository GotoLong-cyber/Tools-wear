#!/usr/bin/env bash
set -euo pipefail

# Wait for clean baseline fold2 to finish with a final fullcurve_raw metric,
# then launch clean baseline fold3 on the same GPU. This is used only for
# round-2 evidence closure so the missing baseline folds can complete
# unattended after the wrapper fix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID:-3}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260401_BaselineClean_resume_gpu3}"
PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"

FOLD2_LOG="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}/longrun_PHM_c4c6_to_c1_rms7_plus_feat4_BaselineClean_dual_seed2026_e1000_bt96_gpu${GPU_ID}.log"

echo "[WAIT] baseline fold2 log=${FOLD2_LOG}"
until [[ -f "${FOLD2_LOG}" ]] && grep -q '\[Metric\]\[fullcurve_raw\]' "${FOLD2_LOG}"; do
  sleep 60
done

echo "[START] baseline fold3 after fold2 completion"
VARIANT=baseline \
FOLD=fold3 \
GPU_ID="${GPU_ID}" \
TRAIN_EPOCHS="${TRAIN_EPOCHS}" \
PATIENCE="${PATIENCE}" \
RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
PYTHON_BIN="${PYTHON_BIN}" \
bash "${SCRIPT_DIR}/run_clean_timer_fold.sh"
