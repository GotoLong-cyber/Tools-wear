#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROUND_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROUND_ROOT}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
export CUDA_VISIBLE_DEVICES=0

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
train_epochs=200
patience=1000

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
mode="dual"
enable_dual_loader=1

WEAR_DIR="${PROJECT_ROOT}/dataset"
DATA_DIR="${ROUND_ROOT}/data"
mkdir -p "${PROJECT_ROOT}/results"

for k in 6 8 10; do
  runtime_dir="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_stageB_fold1_k${k}_gpu1"
  mkdir -p "${runtime_dir}"

  ln -sfn "${DATA_DIR}/c1_passlevel_td28_stageB_fold1_k${k}.npz" "${runtime_dir}/c1_passlevel_full133.npz"
  ln -sfn "${DATA_DIR}/c4_passlevel_td28_stageB_fold1_k${k}.npz" "${runtime_dir}/c4_passlevel_full133.npz"
  ln -sfn "${DATA_DIR}/c6_passlevel_td28_stageB_fold1_k${k}.npz" "${runtime_dir}/c6_passlevel_full133.npz"
  ln -sfn "${WEAR_DIR}/c1/c1_wear.csv" "${runtime_dir}/c1_wear.csv"
  ln -sfn "${WEAR_DIR}/c4/c4_wear.csv" "${runtime_dir}/c4_wear.csv"
  ln -sfn "${WEAR_DIR}/c6/c6_wear.csv" "${runtime_dir}/c6_wear.csv"

  n_vars=$((k + 1))
  model_id="PHM_c1c4_to_c6_td28stageB_k${k}_${mode}_seed${seed}_e${train_epochs}_bt${batch_size}_gpu1"
  run_log="${PROJECT_ROOT}/results/longrun_${model_id}.log"

  echo "[RUN] k=${k} n_vars=${n_vars} model_id=${model_id}"

  "${PYTHON_BIN}" -u run.py \
    --task_name forecast \
    --is_training 1 \
    --seed "${seed}" \
    --root_path "${runtime_dir}" \
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
    --gpu 0 \
    --adaptation \
    --target_only \
    --target_idx -1 \
    --freeze_backbone \
    --cosine \
    --covariate \
    --last_token \
    --nonautoregressive \
    --visualize \
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

done
