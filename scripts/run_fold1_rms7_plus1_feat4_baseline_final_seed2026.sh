#!/usr/bin/env bash
set -euo pipefail

# Official confirmation: Baseline-final (RMS7 + Feat_4), fold1 c1,c4 -> c6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
unset CUDA_VISIBLE_DEVICES

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: Baseline-final must run on GPU."
print("[GPU][OK] cuda available")
PY

GPU_ID="${GPU_ID:-0}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-20260323_BaselineFinal_feat4}"
ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4"
RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_baseline_final_fold1_gpu${GPU_ID}"
mkdir -p "${RUNTIME_DIR}" "${ROUND_DIR}"

ln -sfn "${SELECTED_DIR}/c1_passlevel_rms7_plus_feat4.npz" "${RUNTIME_DIR}/c1_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c4_passlevel_rms7_plus_feat4.npz" "${RUNTIME_DIR}/c4_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c6_passlevel_rms7_plus_feat4.npz" "${RUNTIME_DIR}/c6_passlevel_full133.npz"
ln -sfn "${PROJECT_ROOT}/dataset/c1/c1_wear.csv" "${RUNTIME_DIR}/c1_wear.csv"
ln -sfn "${PROJECT_ROOT}/dataset/c4/c4_wear.csv" "${RUNTIME_DIR}/c4_wear.csv"
ln -sfn "${PROJECT_ROOT}/dataset/c6/c6_wear.csv" "${RUNTIME_DIR}/c6_wear.csv"

model_name="timer_xl"
seq_len=96
token_len=96
pred_len=16
test_seq_len=96
test_pred_len=16
d_model=1024
e_layers=8
d_ff=2048
n_heads=8
batch_size=96
learning_rate=1e-4
train_epochs="${TRAIN_EPOCHS:-1000}"
patience="${PATIENCE:-100}"

train_runs="c1,c4"
test_runs="c6"
split_ratio=0.8
time_gap=0
wear_agg="max"
num_workers=0

lam_mono=0.01
lam_smooth=0.00001
unfreeze_last_n=1
seed=2026
enable_dual_loader=1
n_vars=9

model_id="PHM_c1c4_to_c6_rms7_plus_feat4_BaselineFinal_dual_seed${seed}_e${train_epochs}_bt${batch_size}_gpu${GPU_ID}"
run_log="${ROUND_DIR}/longrun_${model_id}.log"

echo "[RUN] ${model_id}"
echo "[RUN] log=${run_log}"
echo "[RUN] results_subdir=${RESULTS_SUBDIR}"

"${PYTHON_BIN}" -u run.py \
  --task_name forecast \
  --is_training 1 \
  --seed "${seed}" \
  --root_path "${RUNTIME_DIR}" \
  --data_path "." \
  --model_id "${model_id}" \
  --model "${model_name}" \
  --data "PHM_MergedMultivariateNpy" \
  --seq_len ${seq_len} \
  --input_token_len ${token_len} \
  --output_token_len ${pred_len} \
  --test_seq_len ${test_seq_len} \
  --test_pred_len ${test_pred_len} \
  --e_layers ${e_layers} \
  --d_model ${d_model} \
  --d_ff ${d_ff} \
  --n_heads ${n_heads} \
  --batch_size ${batch_size} \
  --learning_rate ${learning_rate} \
  --train_epochs ${train_epochs} \
  --patience ${patience} \
  --gpu ${GPU_ID} \
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
  --lam_mono ${lam_mono} \
  --lam_smooth ${lam_smooth} \
  --n_vars ${n_vars} \
  --train_runs "${train_runs}" \
  --test_runs "${test_runs}" \
  --split_ratio ${split_ratio} \
  --time_gap ${time_gap} \
  --wear_agg "${wear_agg}" \
  --enable_dual_loader ${enable_dual_loader} \
  --num_workers ${num_workers} \
  --unfreeze_last_n ${unfreeze_last_n} \
  --pretrain_model_path "${PROJECT_ROOT}/checkpoint/checkpoint.pth" \
  > "${run_log}" 2>&1

echo "[DONE] ${model_id}"
