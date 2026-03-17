#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=0

# 自动切到 OpenLTM 根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

model_name=timer_xl
model_id=PHM_c1c4_to_c6_walklevel

DATA_ROOT="/home/lc24/Timer/FeaturePHM/OpenLTM/dataset/passlevel_full133_npz/"
PRETRAIN_PTH="/home/lc24/Timer/FeaturePHM/OpenLTM/checkpoint/checkpoint.pth"

# 你的任务设定：历史96，预测24
seq_len=96
test_seq_len=96
test_pred_len=16

# ====== 关键：匹配 checkpoint 骨干 ======
# checkpoint 的 embedding 显示 input_token_len=96
token_len=96

# checkpoint 的 d_model=1024，且有至少8层
d_model=1024
e_layers=8
d_ff=2048

# n_heads 不影响线性层形状，但必须整除 d_model
n_heads=8

# 训练：建议先小学习率 + adaptation（一般会冻结或轻量调参）
batch_size=64
lr=1e-4
epochs=500

echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_PTH=${PRETRAIN_PTH}"

python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path "${DATA_ROOT}" \
  --data_path "c1_passlevel_full133.npz"\
  --model_id "${model_id}" \
  --model "${model_name}" \
  --data "PHM_MergedMultivariateNpy" \
  --seq_len ${seq_len} \
  --input_token_len ${token_len} \
  --output_token_len 16 \
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
  --adaptation \
  --visualize \
  --lam_mono 0.01 \
  --lam_smooth 0.00001 \
  --unfreeze_last_n 1 \
  --n_vars 100 \
  --patience 20 \
  --nonautoregressive \
  --keep_features_path /home/lc24/Timer/FeaturePHM/OpenLTM/dataset/passlevel_full133_npz/corr_stable_out/keep_features.txt \
  --pretrain_model_path "${PRETRAIN_PTH}"
