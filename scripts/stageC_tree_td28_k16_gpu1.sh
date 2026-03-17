#!/usr/bin/env bash
set -euo pipefail

# 阶段C（新）：树选择特征（td28 -> top16），单卡GPU严格实验
# 说明：
# 1）本脚本会准备一个运行时目录，文件名对齐 PHM 加载器约定。
# 2）不改 TimerXL 算法与损失，仅切换数据来源。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

echo "[信息] 项目根目录=${PROJECT_ROOT}"
echo "[信息] 开始时间=$(date '+%F %T')"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
export CUDA_VISIBLE_DEVICES=0
# 重要：这里使用单卡直连（--gpu 0），不启用 --dp。
# 原因：当前预训练权重加载逻辑按参数名精确匹配，DP 会引入 module. 前缀，导致 matched=0。

echo "[信息] GPU自检开始"
"${PYTHON_BIN}" -c "import sys, torch, torch.nn as nn; \
print('[GPU检查] cuda_available=', torch.cuda.is_available(), 'device_count=', torch.cuda.device_count()); \
sys.exit(0 if torch.cuda.is_available() else 1)"
"${PYTHON_BIN}" -c "import torch, torch.nn as nn; m=nn.Linear(4,4).to('cuda:0'); print('[GPU检查] test_model_device=', next(m.parameters()).device)"
echo "[信息] GPU自检通过"

# ===== 数据源（树选择 top16） =====
SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_td28_k16"
DATASET_ROOT="${PROJECT_ROOT}/dataset"

# 运行时目录遵循 PHM 加载器命名约定：
#   c1_passlevel_full133.npz / c4_passlevel_full133.npz / c6_passlevel_full133.npz
#   c1_wear.csv / c4_wear.csv / c6_wear.csv
RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_td28_k16_gpu1"
mkdir -p "${RUNTIME_DIR}"

ln -sfn "${SELECTED_DIR}/c1_passlevel_td28_tree_k16.npz" "${RUNTIME_DIR}/c1_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c4_passlevel_td28_tree_k16.npz" "${RUNTIME_DIR}/c4_passlevel_full133.npz"
ln -sfn "${SELECTED_DIR}/c6_passlevel_td28_tree_k16.npz" "${RUNTIME_DIR}/c6_passlevel_full133.npz"
ln -sfn "${DATASET_ROOT}/c1/c1_wear.csv" "${RUNTIME_DIR}/c1_wear.csv"
ln -sfn "${DATASET_ROOT}/c4/c4_wear.csv" "${RUNTIME_DIR}/c4_wear.csv"
ln -sfn "${DATASET_ROOT}/c6/c6_wear.csv" "${RUNTIME_DIR}/c6_wear.csv"

echo "[信息] 运行时目录=${RUNTIME_DIR}"
ls -l "${RUNTIME_DIR}" | sed 's/^/[信息] /'

mkdir -p "${PROJECT_ROOT}/results"

# ===== 严格实验协议（与历史长训口径对齐） =====
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

batch_size=32
learning_rate=1e-4
train_epochs=200
patience=1000

train_runs="c1,c4"
test_runs="c6"
split_ratio=0.8
time_gap=0
wear_agg="max"
num_workers=0
n_vars=17

lam_mono=0.01
lam_smooth=0.00001
unfreeze_last_n=1

SEEDS=(2026 2027 2028)
MODE="dual"
if [[ "${MODE}" == "dual" ]]; then
  enable_dual_loader=1
else
  enable_dual_loader=0
fi

for seed in "${SEEDS[@]}"; do
  model_id="PHM_c1c4_to_c6_td28tree_k16_${MODE}_seed${seed}_e${train_epochs}_bt${batch_size}_gpu1"
  run_log="${PROJECT_ROOT}/results/longrun_${model_id}.log"

  echo "[运行] seed=${seed} mode=${MODE} model_id=${model_id}"
  echo "[运行] 日志=${run_log}"

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
    --gpu 0 \
    --adaptation \
    --target_only \
    --target_idx -1 \
    --freeze_backbone \
    --cosine \
    --covariate \
    --last_token \
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

  setting_line="$(grep -m1 'start training :' "${run_log}" || true)"
  setting_name="$(echo "${setting_line}" | sed -E 's/.*start training : ([^>]+)>.*/\1/')"
  if [[ -n "${setting_name}" ]]; then
    echo "[可视化] ${PROJECT_ROOT}/results/${setting_name}/wear_window_0.png"
    echo "[可视化] ${PROJECT_ROOT}/results/${setting_name}/wear_full_curve_trueRaw_predWindows.png"
    echo "[可视化] ${PROJECT_ROOT}/test_results/${setting_name}/16/"
  fi
  echo "[完成] seed=${seed} 时间=$(date '+%F %T')"
done

echo "[信息] 全部运行结束，时间=$(date '+%F %T')"
