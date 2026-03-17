#!/usr/bin/env bash
set -e
export CUDA_VISIBLE_DEVICES=""

# 自动切到 OpenLTM 根目录   --unfreeze_last_n 1 \
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"
echo "Project root: ${PROJECT_ROOT}"

model_name=timer_xl

# ====== B2: 走刀级 + 全量133特征（npz） ======
# 交叉验证 fold1：C1+C4 -> C6
model_id=PHM_c1c4_to_c6_passlevel_full133_B2_quickDual

# 你生成的 npz 输出目录（注意：这里是目录，不是某个文件）
DATA_ROOT="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/dataset/passlevel_full133_npz/"
PRETRAIN_PTH="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/checkpoint/checkpoint.pth"

# ====== 任务设定 ======
# B2 输入包含 Feat_* + OT历史：
# - seq_len: 历史长度（走刀级），你要 96
# - output_token_len: 你计划每段预测 16（再 roll-out）
seq_len=96
token_len=96
pred_len=16

# 下面这俩是 exp_forecast.test 里用的（如果你只做 16，也设 16）
test_seq_len=${seq_len}
test_pred_len=${pred_len}

# ====== 关键：匹配 checkpoint 骨干 ======
d_model=256
e_layers=2
d_ff=512
n_heads=8

batch_size=8
lr=1e-4
epochs=1

echo "DATA_ROOT=${DATA_ROOT}"
echo "PRETRAIN_PTH=${PRETRAIN_PTH}"
echo "model_id=${model_id}"

/home/lc24/miniconda3/envs/TimerXL/bin/python -u run.py \
  --task_name forecast \
  --is_training 1 \
  --root_path "${DATA_ROOT}" \
  --data_path "c1_passlevel_full133.npz" \
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
  --visualize \
  --lam_mono 0.01 \
  --lam_smooth 0.00001 \
  --n_vars 134 \
  --train_runs c1,c4 \
  --test_runs c6 \
  --split_ratio 0.8 \
  --time_gap 0 \
  --wear_agg max \
  --enable_dual_loader 1 \
  --num_workers 0 \
  --unfreeze_last_n 1 \
  --pretrain_model_path "${PRETRAIN_PTH}"
