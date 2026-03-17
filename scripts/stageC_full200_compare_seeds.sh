#!/usr/bin/env bash
set -euo pipefail

# ===== project =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"

echo "[Info] start time: $(date '+%F %T')"

echo "[Info] python: /home/lc24/miniconda3/envs/TimerXL/bin/python"

# ===== runtime =====
export CUDA_VISIBLE_DEVICES="0,1,2"
PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"

# ===== data/checkpoint =====
DATA_ROOT="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/dataset/passlevel_full133_npz/"
PRETRAIN_PTH="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/checkpoint/checkpoint.pth"

# ===== fixed protocol (aligned) =====
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
lr=1e-4
epochs=100
patience=1000

train_runs="c1,c4"
test_runs="c6"
split_ratio=0.8
time_gap=0
wear_agg="max"
num_workers=8

lam_mono=0.01
lam_smooth=0.00001
n_vars=134
unfreeze_last_n=1

SEEDS=(2026 2027 2028)
MODES=(dual single)

for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    if [[ "${mode}" == "dual" ]]; then
      enable_dual_loader=1
    else
      enable_dual_loader=0
    fi

    model_id="PHM_c1c4_to_c6_full133_${mode}_seed${seed}_e200_gpu3"
    run_log="${PROJECT_ROOT}/results/longrun_${model_id}.log"

    echo "[Run] seed=${seed} mode=${mode} model_id=${model_id}"
    echo "[Run] log=${run_log}"

    "${PYTHON_BIN}" -u run.py \
      --task_name forecast \
      --is_training 1 \
      --seed "${seed}" \
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
      --patience ${patience} \
      --gpu 0 \
      --dp \
      --devices 0,1,2 \
      --adaptation \
      --target_only \
      --target_idx -1 \
      --freeze_backbone \
      --cosine \
      --covariate \
      --last_token \
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
      --pretrain_model_path "${PRETRAIN_PTH}" \
      > "${run_log}" 2>&1

    echo "[Done] seed=${seed} mode=${mode} at $(date '+%F %T')"
  done
done

echo "[Info] all runs done at $(date '+%F %T')"
