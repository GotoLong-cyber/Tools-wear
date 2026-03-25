#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/evaluate_fold1_knn_delta_retrieval.py" \
  --project_root "${PROJECT_ROOT}" \
  --runtime_root "dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_fold2_gpu1" \
  --checkpoint_path "checkpoints/forecast_PHM_c4c6_to_c1_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth" \
  --train_runs "c4,c6" \
  --test_runs "c1" \
  --tag "fold2" \
  --results_subdir "${RESULTS_SUBDIR:-20260325_KNNGate3Fold}" \
  --k "${KNN_K:-5}" \
  --betas "${KNN_BETAS:-0.5}" \
  --blend_mode "${KNN_BLEND_MODE:-delta_residual}" \
  --library_wear_threshold_um "${LIB_WEAR_THR:-0}" \
  --library_wear_quantile "${LIB_WEAR_Q:-0.8}" \
  --gate_mode "${KNN_GATE_MODE:-distance_linear}" \
  --gate_stat "${KNN_GATE_STAT:-min}" \
  --gate_qlo "${KNN_GATE_QLO:-0.1}" \
  --gate_qhi "${KNN_GATE_QHI:-0.9}" \
  --gate_beta_min "${KNN_GATE_BMIN:-0.0}" \
  --gate_beta_max "${KNN_GATE_BMAX:-1.0}"
