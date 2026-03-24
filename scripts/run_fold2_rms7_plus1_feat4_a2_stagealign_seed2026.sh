#!/usr/bin/env bash
set -euo pipefail

# A2 Fold-2 对照实验（磨损阶段对齐切片）
# train/val: c4,c6 ; test: c1
# 模型主体不改，仅改变训练域数据组织口径（按磨损进程重参数化）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
unset CUDA_VISIBLE_DEVICES

GPU_ID="${GPU_ID:-0}"
train_epochs="${TRAIN_EPOCHS:-1000}"
patience="${PATIENCE:-100}"

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: A2 must run on GPU."
print("[GPU][OK] cuda available")
PY

SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold2"
RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_a2_stagealign_fold2_gpu${GPU_ID}"
mkdir -p "${SELECTED_DIR}" "${RUNTIME_DIR}" "${PROJECT_ROOT}/results"

"${PYTHON_BIN}" "${PROJECT_ROOT}/feature_alignment_diagnosis/scripts/build_a2_stage_aligned_fold1_data.py" \
  --project_root "${PROJECT_ROOT}" \
  --base_npz_dir "dataset/passlevel_tree_select/base_td28" \
  --output_dir "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold2" \
  --runs c1 c4 c6 \
  --train_runs c4 c6 \
  --feature_idx 2 6 10 14 18 22 26 3

ln -sfn "${SELECTED_DIR}/c1_passlevel_rms7_plus_feat4_a2stage.npz" "${RUNTIME_DIR}/c1_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c4_passlevel_rms7_plus_feat4_a2stage.npz" "${RUNTIME_DIR}/c4_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c6_passlevel_rms7_plus_feat4_a2stage.npz" "${RUNTIME_DIR}/c6_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/wear_csv/c1_wear.csv" "${RUNTIME_DIR}/c1_wear.csv"
ln -sfn "${SELECTED_DIR}/wear_csv/c4_wear.csv" "${RUNTIME_DIR}/c4_wear.csv"
ln -sfn "${SELECTED_DIR}/wear_csv/c6_wear.csv" "${RUNTIME_DIR}/c6_wear.csv"

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

train_runs="c4,c6"
test_runs="c1"
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

model_id="PHM_c4c6_to_c1_rms7_plus_feat4_A2stagealign_dual_seed${seed}_e${train_epochs}_bt${batch_size}_gpu${GPU_ID}"
run_log="${PROJECT_ROOT}/results/longrun_${model_id}.log"

echo "[RUN] ${model_id}"
echo "[RUN] log=${run_log}"

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

