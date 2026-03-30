#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260330_A2PlusSE1_3fold_train}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
mkdir -p "${ROUND_DIR}"

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: training requires GPU."
print("[GPU][OK] CUDA is available.")
PY

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_extraction/build_rms7_feat4_plus_se1.py" \
  --project_root "${PROJECT_ROOT}" \
  --runs c1 c4 c6 \
  --out_dir "dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_se1" \
  --channel_idx 0

PYTHON_BIN="${PYTHON_BIN}" SKIP_FEATURE_BUILD=1 GPU_ID=0 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${PROJECT_ROOT}/scripts/run_fold1_rms7_plus1_feat4_plus_se1_a2_seed2026.sh" \
  > "${ROUND_DIR}/master_train_fold1_gpu0.log" 2>&1 &
PID1=$!

PYTHON_BIN="${PYTHON_BIN}" SKIP_FEATURE_BUILD=1 GPU_ID=1 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${PROJECT_ROOT}/scripts/run_fold2_rms7_plus1_feat4_plus_se1_a2_seed2026.sh" \
  > "${ROUND_DIR}/master_train_fold2_gpu1.log" 2>&1 &
PID2=$!

PYTHON_BIN="${PYTHON_BIN}" SKIP_FEATURE_BUILD=1 GPU_ID=2 TRAIN_EPOCHS="${TRAIN_EPOCHS}" PATIENCE="${PATIENCE}" RESULTS_SUBDIR="${RESULTS_SUBDIR}" \
  bash "${PROJECT_ROOT}/scripts/run_fold3_rms7_plus1_feat4_plus_se1_a2_seed2026.sh" \
  > "${ROUND_DIR}/master_train_fold3_gpu2.log" 2>&1 &
PID3=$!

echo "RESULTS_SUBDIR=${RESULTS_SUBDIR}"
echo "PIDS: fold1=${PID1} fold2=${PID2} fold3=${PID3}"

wait "${PID1}"; S1=$?
wait "${PID2}"; S2=$?
wait "${PID3}"; S3=$?

echo "EXIT: fold1=${S1} fold2=${S2} fold3=${S3}"
