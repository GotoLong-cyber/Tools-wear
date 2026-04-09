#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM}"
PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_ROOT}/paper_exec/train_only_param_selection/results_rms7_baseline_e300}"
K_GRID="${K_GRID:-3,5,7,10}"
BETA_GRID="${BETA_GRID:-0.3,0.5,0.7,1.0}"
LATE_Q_GRID="${LATE_Q_GRID:-0.0,0.5,0.8}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/paper_exec/train_only_param_selection/scripts/select_knn_hparams_inner_loso_rms7_baseline_e300.py" \
  --project_root "${PROJECT_ROOT}" \
  --results_dir "${RESULTS_DIR}" \
  --k_grid "${K_GRID}" \
  --beta_grid "${BETA_GRID}" \
  --late_q_grid "${LATE_Q_GRID}" \
  --selection_metric "mae_full_raw"
