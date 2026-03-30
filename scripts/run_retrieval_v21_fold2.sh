#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/run_retrieval_v21_infer.py" \
  --project_root "${PROJECT_ROOT}" \
  --runtime_root "dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_fold2_gpu1" \
  --checkpoint_path "checkpoints/forecast_PHM_c4c6_to_c1_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth" \
  --train_runs "c4,c6" \
  --test_runs "c1" \
  --tag "fold2" \
  --results_subdir "${RESULTS_SUBDIR:-20260325_RetrievalV21_formal}" \
  --k "${KNN_K:-5}" \
  --betas "${KNN_BETAS:-0.5}" \
  --blend_mode "${KNN_BLEND_MODE:-delta_residual}" \
  --library_wear_threshold_um "${LIB_WEAR_THR:-0}" \
  --library_wear_quantile "${LIB_WEAR_Q:-0.8}"
