#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
GPU_ID="${GPU_ID:-3}"
RESULTS_DIR="${RESULTS_DIR:-paper_exec/train_only_param_selection/results}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_BIN}" \
  "${PROJECT_ROOT}/paper_exec/train_only_param_selection/scripts/select_knn_hparams_train_only.py" \
  --project_root "${PROJECT_ROOT}" \
  --results_dir "${PROJECT_ROOT}/${RESULTS_DIR}" \
  --k_grid "${K_GRID:-3,5,10}" \
  --beta_grid "${BETA_GRID:-0.3,0.5,0.7}" \
  --late_q_grid "${LATE_Q_GRID:-0.0,0.8}"
