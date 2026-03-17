#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PY="/home/lc24/miniconda3/envs/TimerXL/bin/python"

${PY} -u feature_extraction/pipeline_tree_selection.py \
  --dataset_root "${PROJECT_ROOT}/dataset" \
  --runs c1,c4,c6 \
  --train_runs c1,c4 \
  --feature_set td28 \
  --wear_agg max \
  --seq_len 96 \
  --pred_len 16 \
  --split_ratio 0.8 \
  --rf_seeds 2026,2027,2028 \
  --n_estimators 600 \
  --max_depth 10 \
  --top_k 16 \
  --out_dir "${PROJECT_ROOT}/dataset/passlevel_tree_select"
