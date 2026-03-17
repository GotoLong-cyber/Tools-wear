#!/usr/bin/env bash
set -euo pipefail

# Step 2: Fold-1 单通道 +1 队列实验（hardest fold: c1,c4 -> c6）
# 说明：
# 1) 不改模型算法，仅替换输入特征集合；
# 2) 使用 GPU（--gpu ${GPU_ID}），并沿用既有 TimerXL 训练协议；
# 3) 默认仅启动队列中的第 1 个候选，便于快速验收；
#    如需全量跑队列，可设置 MAX_CANDIDATES=9。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="/home/lc24/miniconda3/envs/TimerXL/bin/python"
unset CUDA_VISIBLE_DEVICES
GPU_ID="${GPU_ID:-0}"

# 强约束：实验必须使用 GPU，不允许 CPU fallback
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[GPU-ERROR] nvidia-smi 不可用，终止实验。"
  exit 10
fi

echo "[GPU-CHECK] host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv,noheader
"${PYTHON_BIN}" -c "import torch,sys; ok=torch.cuda.is_available(); n=torch.cuda.device_count(); print(f'[GPU-CHECK] cuda_available={ok} cuda_count={n}'); sys.exit(0 if ok and n>0 else 11)"

BASE_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/base_td28"
RESULT_DIR="${PROJECT_ROOT}/results"
mkdir -p "${RESULT_DIR}"

# 基线：RMS7 + Feat_4
# 0-based 索引：[2,6,10,14,18,22,26,3]
BASE_IDX=(2 6 10 14 18 22 26 3)

# Step 2 第一批单通道 +1 队列
CANDIDATES=(Feat_8 Feat_10 Feat_12 Feat_16 Feat_18 Feat_20 Feat_22 Feat_24 Feat_28)

MAX_CANDIDATES="${MAX_CANDIDATES:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"

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
split_ratio=0.8
time_gap=0
wear_agg="max"
num_workers=0

lam_mono=0.01
lam_smooth=0.00001
unfreeze_last_n=1

seed=2026
enable_dual_loader=1

train_runs="c1,c4"
test_runs="c6"

count=0
for feat in "${CANDIDATES[@]}"; do
  if [[ "${count}" -ge "${MAX_CANDIDATES}" ]]; then
    break
  fi
  count=$((count + 1))

  feat_num="${feat#Feat_}"
  feat_idx=$((feat_num - 1))
  feat_lc="$(echo "${feat}" | tr '[:upper:]' '[:lower:]')"

  SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_${feat_lc}"
  RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_${feat_lc}_gpu1"
  mkdir -p "${SELECTED_DIR}" "${RUNTIME_DIR}"

  echo "[STEP2] prepare ${feat} (idx=${feat_idx})"
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
import numpy as np

project = Path("${PROJECT_ROOT}")
base_dir = project / "dataset/passlevel_tree_select/base_td28"
selected_dir = Path("${SELECTED_DIR}")
selected_dir.mkdir(parents=True, exist_ok=True)

idx = np.array([2, 6, 10, 14, 18, 22, 26, 3, ${feat_idx}], dtype=np.int64)
names = [f"Feat_{int(i+1)}" for i in idx]

for run in ["c1", "c4", "c6"]:
    z = np.load(base_dir / f"{run}_passlevel_td28.npz", allow_pickle=True)
    X = z["X"][:, idx].astype(np.float32)
    y = z["y"].astype(np.float32)
    p = z["pass_idx"].astype(np.int32) if "pass_idx" in z.files else np.arange(X.shape[0], dtype=np.int32)
    np.savez(
        selected_dir / f"{run}_passlevel_rms7_plus_feat4_plus_${feat_lc}.npz",
        X=X,
        y=y,
        pass_idx=p,
        feature_names=np.array(names, dtype=object),
    )

(selected_dir / "keep_features_rms7_plus_feat4_plus_${feat_lc}.txt").write_text("\\n".join(names) + "\\n", encoding="utf-8")
print("[OK] selected feature count =", len(names))
print("[OK] features =", names)
PY

  ln -sfn "${SELECTED_DIR}/c1_passlevel_rms7_plus_feat4_plus_${feat_lc}.npz" "${RUNTIME_DIR}/c1_passlevel_full133.npz"
  ln -sfn "${SELECTED_DIR}/c4_passlevel_rms7_plus_feat4_plus_${feat_lc}.npz" "${RUNTIME_DIR}/c4_passlevel_full133.npz"
  ln -sfn "${SELECTED_DIR}/c6_passlevel_rms7_plus_feat4_plus_${feat_lc}.npz" "${RUNTIME_DIR}/c6_passlevel_full133.npz"
  ln -sfn "${PROJECT_ROOT}/dataset/c1/c1_wear.csv" "${RUNTIME_DIR}/c1_wear.csv"
  ln -sfn "${PROJECT_ROOT}/dataset/c4/c4_wear.csv" "${RUNTIME_DIR}/c4_wear.csv"
  ln -sfn "${PROJECT_ROOT}/dataset/c6/c6_wear.csv" "${RUNTIME_DIR}/c6_wear.csv"

  n_vars=10  # 9维特征 + 1维wear
  model_id="PHM_c1c4_to_c6_rms7_plus_feat4_plus_${feat_lc}_dual_seed${seed}_e${TRAIN_EPOCHS}_bt${batch_size}_gpu1_step2"
  run_log="${RESULT_DIR}/longrun_${model_id}.log"

  echo "[STEP2] run ${model_id}"
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
    --train_epochs ${TRAIN_EPOCHS} \
    --patience ${PATIENCE} \
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

  echo "[STEP2] done ${model_id}"
done

echo "[STEP2] completed count=${count}"
