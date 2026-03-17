#!/usr/bin/env bash
set -euo pipefail

# Fold-1 正式实验（严格同口径）
# 目标：RMS7 + 来自 StageB-k6 的 1 个特征（Feat_4）= 8 维特征
# 说明：
# 1) 不修改模型结构与损失函数，仅修改输入特征集。
# 2) 训练/验证域：c1,c4；测试域：c6。
# 3) 参数与阶段B主实验保持一致（seed/epoch/batch/model规模等）。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
# 强约束：本实验必须使用 GPU。
# 该服务器环境下显式设置 CUDA_VISIBLE_DEVICES=0 会导致 torch 误判无 CUDA，
# 因此这里显式取消该变量，改由 --gpu 0 指定设备。
unset CUDA_VISIBLE_DEVICES

BASE_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/base_td28"
SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4"
RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_gpu1"
mkdir -p "${SELECTED_DIR}" "${RUNTIME_DIR}" "${PROJECT_ROOT}/results"

# 从 td28 中抽取：每通道 RMS + Feat_4（range of channel-1）
# td28 顺序为 [mean, std, rms, range] * 7
# 0-based 索引：
# RMS   -> [2, 6, 10, 14, 18, 22, 26]
# Feat_4 -> index=3
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import numpy as np

project = Path("/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
base_dir = project / "dataset/passlevel_tree_select/base_td28"
selected_dir = project / "dataset/passlevel_tree_select/selected_rms7_plus_feat4"
selected_dir.mkdir(parents=True, exist_ok=True)

idx = np.array([2, 6, 10, 14, 18, 22, 26, 3], dtype=np.int64)
names = [f"Feat_{int(i+1)}" for i in idx]

for run in ["c1", "c4", "c6"]:
    z = np.load(base_dir / f"{run}_passlevel_td28.npz", allow_pickle=True)
    X = z["X"][:, idx].astype(np.float32)
    y = z["y"].astype(np.float32)
    p = z["pass_idx"].astype(np.int32) if "pass_idx" in z.files else np.arange(X.shape[0], dtype=np.int32)
    np.savez(
        selected_dir / f"{run}_passlevel_rms7_plus_feat4.npz",
        X=X,
        y=y,
        pass_idx=p,
        feature_names=np.array(names, dtype=object),
    )

(selected_dir / "keep_features_rms7_plus_feat4.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
print("[OK] selected feature count =", len(names))
print("[OK] features =", names)
PY

# 适配 PHM_MergedMultivariateNpy 的命名约定
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
# 默认拉长训练轮次；可通过环境变量覆盖，如 TRAIN_EPOCHS=1200
train_epochs="${TRAIN_EPOCHS:-1000}"
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
enable_dual_loader=1
n_vars=9  # 8维特征 + 1维wear

model_id="PHM_c1c4_to_c6_rms7_plus_feat4_dual_seed${seed}_e${train_epochs}_bt${batch_size}_gpu1"
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

echo "[DONE] ${model_id}"
