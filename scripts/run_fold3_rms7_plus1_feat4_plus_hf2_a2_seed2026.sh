#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
unset CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: A2+HF2 must run on GPU."
print("[GPU][OK] cuda available")
PY

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_extraction/build_rms7_feat4_plus_hf2.py" \
  --project_root "${PROJECT_ROOT}" \
  --runs c1 c4 c6 \
  --out_dir "dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_hf2" \
  --channel_idx 1

GPU_ID="${GPU_ID:-2}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260324_A2PlusHF2}"
ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_hf2"
RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_hf2_a2_fold3_gpu${GPU_ID}"
mkdir -p "${RUNTIME_DIR}" "${ROUND_DIR}"

ln -sfn "${SELECTED_DIR}/c1_passlevel_rms7_plus_feat4_plus_hf2.npz" "${RUNTIME_DIR}/c1_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c4_passlevel_rms7_plus_feat4_plus_hf2.npz" "${RUNTIME_DIR}/c4_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c6_passlevel_rms7_plus_feat4_plus_hf2.npz" "${RUNTIME_DIR}/c6_passlevel_full133.npz"
ln -sfn "${PROJECT_ROOT}/dataset/c1/c1_wear.csv" "${RUNTIME_DIR}/c1_wear.csv"
ln -sfn "${PROJECT_ROOT}/dataset/c4/c4_wear.csv" "${RUNTIME_DIR}/c4_wear.csv"
ln -sfn "${PROJECT_ROOT}/dataset/c6/c6_wear.csv" "${RUNTIME_DIR}/c6_wear.csv"

model_id="PHM_c1c6_to_c4_rms7_plus_feat4_plus_hf2_A2_dual_seed2026_e${TRAIN_EPOCHS:-1000}_bt96_gpu${GPU_ID}"
run_log="${ROUND_DIR}/longrun_${model_id}.log"

"${PYTHON_BIN}" -u run.py \
  --task_name forecast \
  --is_training 1 \
  --seed 2026 \
  --root_path "${RUNTIME_DIR}" \
  --data_path "." \
  --model_id "${model_id}" \
  --model timer_xl \
  --data "PHM_MergedMultivariateNpy" \
  --seq_len 96 \
  --input_token_len 96 \
  --output_token_len 16 \
  --test_seq_len 96 \
  --test_pred_len 16 \
  --e_layers 8 \
  --d_model 1024 \
  --d_ff 2048 \
  --n_heads 8 \
  --batch_size 96 \
  --learning_rate 1e-4 \
  --train_epochs "${TRAIN_EPOCHS:-1000}" \
  --patience "${PATIENCE:-100}" \
  --gpu "${GPU_ID}" \
  --adaptation \
  --target_only \
  --target_idx -1 \
  --freeze_backbone \
  --cosine \
  --covariate \
  --last_token \
  --nonautoregressive \
  --visualize \
  --results_subdir "${RESULTS_SUBDIR}" \
  --lam_mono 0.01 \
  --lam_smooth 0.00001 \
  --n_vars 10 \
  --train_runs "c1,c6" \
  --test_runs "c4" \
  --split_ratio 0.8 \
  --time_gap 0 \
  --wear_agg "max" \
  --enable_dual_loader 1 \
  --num_workers 0 \
  --unfreeze_last_n 1 \
  --train_stride_candidates "1,2" \
  --train_stride_quantiles "0.5" \
  --train_stride_use_monotonic_wear 1 \
  --train_stride_policy random \
  --train_stride_random_seed 2026 \
  --pretrain_model_path "${PROJECT_ROOT}/checkpoint/checkpoint.pth" \
  > "${run_log}" 2>&1
