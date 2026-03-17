#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

model_name=timer_xl
model_id=PHM_c1c4_to_c6_walklevel_ot16

DATA_ROOT="/home/lc24/Timer/TimerxlMu/OpenLTM/dataset/PHM_processed_walklevel"
PRETRAIN_PTH="/home/lc24/Timer/TimerxlMu/OpenLTM/checkpoint/checkpoint.pth"

seq_len=192
test_seq_len=96
test_pred_len=16
token_len=96

d_model=1024
e_layers=8
d_ff=2048
n_heads=8

batch_size=64
lr=1e-4
epochs=500

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path "${DATA_ROOT}" \
  --data_path "c1.npy" \
  --model_id "${model_id}" \
  --model "${model_name}" \
  --data "PHM_MergedMultivariateNpy" \
  --seq_len ${seq_len} \
  --input_token_len ${token_len} \
  --output_token_len ${test_pred_len} \
  --test_seq_len ${test_seq_len} \
  --test_pred_len ${test_pred_len} \
  --e_layers ${e_layers} \
  --d_model ${d_model} \
  --d_ff ${d_ff} \
  --n_heads ${n_heads} \
  --batch_size ${batch_size} \
  --learning_rate ${lr} \
  --train_epochs ${epochs} \
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
  --lam_mono 0.01 \
  --lam_smooth 0.0001 \
  --unfreeze_last_n 1 \
  --pretrain_model_path "${PRETRAIN_PTH}"
