#!/usr/bin/env bash
set -euo pipefail

# Template only. Do NOT auto-run in this turn.
# Example: Fold-1 train c1,c4 and later test on c6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROUND_ROOT}/../.." && pwd)"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"

DATASET_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/base_td28"
OUT_ROOT="${ROUND_ROOT}"

TRAIN_RUNS="c1,c4"
EXPORT_RUNS="c1,c4,c6"

"${PYTHON_BIN}" "${SCRIPT_DIR}/stageA_stability_filter.py" \
  --dataset_dir "${DATASET_DIR}" \
  --train_runs "${TRAIN_RUNS}" \
  --export_runs "${EXPORT_RUNS}" \
  --input_suffix "passlevel_td28" \
  --output_suffix "passlevel_td28_stageA" \
  --shift_sigma_thr 2.0 \
  --corr_std_thr 0.35 \
  --corr_sign_eps 0.05 \
  --corr_redundancy_thr 0.98 \
  --score_shift_weight 0.10 \
  --out_root "${OUT_ROOT}"
