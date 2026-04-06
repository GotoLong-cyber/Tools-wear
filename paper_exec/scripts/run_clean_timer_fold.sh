#!/usr/bin/env bash
set -euo pipefail

# Generic clean-protocol training entrypoint for PHM2010 fold runs.
# This wrapper disables epoch-wise test evaluation during training and keeps
# train/val-only model selection. Test is executed only after training freezes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/home/jyc23/miniconda3/envs/TimerXL/bin/python}"
VARIANT="${VARIANT:?VARIANT must be one of baseline,tma,retrieval}"
FOLD="${FOLD:?FOLD must be one of fold1,fold2,fold3}"
GPU_ID="${GPU_ID:?GPU_ID is required}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:?RESULTS_SUBDIR is required}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
SEED="${SEED:-2026}"
DISABLE_TRAIN_TEST_EVAL="${DISABLE_TRAIN_TEST_EVAL:-1}"
WEAR_AGG="${WEAR_AGG:-max}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON_BIN}" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable: clean formal training requires GPU."
print("[GPU][OK] cuda available")
PY

case "${FOLD}" in
  fold1) TRAIN_RUNS="c1,c4"; TEST_RUNS="c6" ;;
  fold2) TRAIN_RUNS="c4,c6"; TEST_RUNS="c1" ;;
  fold3) TRAIN_RUNS="c1,c6"; TEST_RUNS="c4" ;;
  *) echo "Unknown FOLD=${FOLD}" >&2; exit 1 ;;
esac

case "${VARIANT}" in
  baseline)
    N_VARS=9
    MODEL_SUFFIX="rms7_plus_feat4_BaselineClean"
    SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4"
    FILE_SUFFIX="rms7_plus_feat4"
    RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_baseline_clean_${FOLD}_gpu${GPU_ID}"
    EXTRA_ARGS=()
    ;;
  tma)
    N_VARS=9
    MODEL_SUFFIX="rms7_plus_feat4_TMAClean"
    SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4"
    FILE_SUFFIX="rms7_plus_feat4"
    RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_tma_clean_${FOLD}_gpu${GPU_ID}"
    EXTRA_ARGS=(
      --train_stride_candidates "1,2"
      --train_stride_quantiles "0.5"
      --train_stride_use_monotonic_wear 1
      --train_stride_policy random
      --train_stride_random_seed 2026
    )
    ;;
  retrieval)
    N_VARS=10
    MODEL_SUFFIX="rms7_plus_feat4_plus_se1_TMAClean"
    SELECTED_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_se1"
    FILE_SUFFIX="rms7_plus_feat4_plus_se1"
    RUNTIME_DIR="${PROJECT_ROOT}/dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_tma_clean_${FOLD}_gpu${GPU_ID}"
    if [[ "${SKIP_FEATURE_BUILD:-0}" != "1" ]]; then
      "${PYTHON_BIN}" "${PROJECT_ROOT}/feature_extraction/build_rms7_feat4_plus_se1.py" \
        --project_root "${PROJECT_ROOT}" \
        --runs c1 c4 c6 \
        --out_dir "dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_se1" \
        --channel_idx 0
    fi
    EXTRA_ARGS=(
      --train_stride_candidates "1,2"
      --train_stride_quantiles "0.5"
      --train_stride_use_monotonic_wear 1
      --train_stride_policy random
      --train_stride_random_seed 2026
    )
    ;;
  *)
    echo "Unknown VARIANT=${VARIANT}" >&2
    exit 1
    ;;
esac

ROUND_DIR="${PROJECT_ROOT}/results/${RESULTS_SUBDIR}"
mkdir -p "${RUNTIME_DIR}" "${ROUND_DIR}"

for run in c1 c4 c6; do
  ln -sfn "${SELECTED_DIR}/${run}_passlevel_${FILE_SUFFIX}.npz" "${RUNTIME_DIR}/${run}_passlevel_full133.npz"
  ln -sfn "${PROJECT_ROOT}/dataset/${run}/${run}_wear.csv" "${RUNTIME_DIR}/${run}_wear.csv"
done

MODEL_ID="PHM_${TRAIN_RUNS//,/}_to_${TEST_RUNS}_${MODEL_SUFFIX}_${WEAR_AGG}agg_dual_seed${SEED}_e${TRAIN_EPOCHS}_bt96_gpu${GPU_ID}"
RUN_LOG="${ROUND_DIR}/longrun_${MODEL_ID}.log"

CMD=(
  "${PYTHON_BIN}" -u run.py
  --task_name forecast
  --is_training 1
  --seed "${SEED}"
  --root_path "${RUNTIME_DIR}"
  --data_path "."
  --model_id "${MODEL_ID}"
  --model timer_xl
  --data "PHM_MergedMultivariateNpy"
  --seq_len 96
  --input_token_len 96
  --output_token_len 16
  --test_seq_len 96
  --test_pred_len 16
  --e_layers 8
  --d_model 1024
  --d_ff 2048
  --n_heads 8
  --batch_size 96
  --learning_rate 1e-4
  --train_epochs "${TRAIN_EPOCHS}"
  --patience "${PATIENCE}"
  --gpu 0
  --adaptation
  --target_only
  --target_idx -1
  --freeze_backbone
  --cosine
  --covariate
  --last_token
  --nonautoregressive
  --visualize
  --results_subdir "${RESULTS_SUBDIR}"
  --lam_mono 0.01
  --lam_smooth 0.00001
  --n_vars "${N_VARS}"
  --train_runs "${TRAIN_RUNS}"
  --test_runs "${TEST_RUNS}"
  --split_ratio 0.8
  --time_gap 0
  --wear_agg "${WEAR_AGG}"
  --enable_dual_loader 1
  --num_workers 0
  --unfreeze_last_n 1
  --pretrain_model_path "${PROJECT_ROOT}/checkpoint/checkpoint.pth"
)

if [[ "${DISABLE_TRAIN_TEST_EVAL}" == "1" ]]; then
  CMD+=(--disable_train_test_eval)
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[RUN] ${MODEL_ID}"
echo "[RUN] variant=${VARIANT} fold=${FOLD} gpu=${GPU_ID} wear_agg=${WEAR_AGG}"
echo "[RUN] log=${RUN_LOG}"
CMD_FILE="${ROUND_DIR}/cmd_${MODEL_ID}.txt"
: > "${CMD_FILE}"
for arg in "${CMD[@]}"; do
  printf '%q ' "${arg}" >> "${CMD_FILE}"
done
printf '\n' >> "${CMD_FILE}"
cat "${CMD_FILE}"
echo

set +e
"${CMD[@]}" > "${RUN_LOG}" 2>&1
cmd_status=$?
set -e

if [[ ${cmd_status} -ne 0 ]]; then
  echo "[FAIL] ${MODEL_ID} (exit=${cmd_status})" >&2
  exit "${cmd_status}"
fi

echo "[DONE] ${MODEL_ID}"
