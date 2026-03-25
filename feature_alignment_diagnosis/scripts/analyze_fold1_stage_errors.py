#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from exp.exp_forecast import Exp_Forecast


def build_args(project_root: Path, results_subdir: str) -> SimpleNamespace:
    runtime_dir = project_root / "dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_fold1_gpu0"
    checkpoint_setting = (
        "forecast_PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu0_"
        "timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
    )
    return SimpleNamespace(
        task_name="forecast",
        is_training=0,
        model_id="PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu0",
        model="timer_xl",
        seed=2026,
        data="PHM_MergedMultivariateNpy",
        root_path=str(runtime_dir),
        data_path=".",
        checkpoints=str(project_root / "checkpoints"),
        results_subdir=results_subdir,
        test_flag="T",
        seq_len=96,
        input_token_len=96,
        output_token_len=16,
        test_seq_len=96,
        test_pred_len=16,
        dropout=0.1,
        e_layers=8,
        d_model=1024,
        n_heads=8,
        d_ff=2048,
        activation="relu",
        covariate=True,
        node_num=100,
        node_list=[23, 37, 40],
        use_norm=False,
        nonautoregressive=True,
        test_dir=checkpoint_setting,
        test_file_name="checkpoint.pth",
        output_attention=False,
        visualize=False,
        flash_attention=False,
        target_only=True,
        target_idx=-1,
        adaptation=True,
        pretrain_model_path=str(project_root / "checkpoint/checkpoint.pth"),
        subset_rand_ratio=1,
        freeze_backbone=True,
        lam_mono=0.01,
        lam_smooth=0.00001,
        unfreeze_last_n=1,
        keep_features_path="",
        train_runs="c1,c4",
        test_runs="c6",
        split_ratio=0.8,
        time_gap=0,
        wear_agg="max",
        mask_future_features_in_y=False,
        enable_dual_loader=1,
        train_stride_candidates="1,2",
        train_stride_quantiles="0.5",
        train_stride_use_monotonic_wear=1,
        train_stride_policy="random",
        train_stride_random_seed=2026,
        train_window_weight_policy="none",
        train_window_weight_quantile=0.5,
        train_window_weight_seed=2026,
        num_workers=0,
        itr=1,
        train_epochs=1000,
        batch_size=96,
        patience=100,
        learning_rate=1e-4,
        des="test",
        loss="MSE",
        lradj="type1",
        cosine=True,
        tmax=10,
        weight_decay=0.0,
        valid_last=False,
        last_token=True,
        gpu=0,
        ddp=False,
        dp=False,
        devices="0,1,2,3",
        gpt_layers=6,
        patch_size=16,
        kernel_size=25,
        stride=8,
        n_vars=10,
        factor=2,
        mode="mix_channel",
        AP_levels=0,
        use_decoder=True,
        d_mode="common_channel",
        layers=8,
        hidden_dim=16,
        ts_vocab_size=1000,
        domain_des="PHM tool wear forecasting.",
        llm_model="LLAMA",
        llm_layers=6,
        local_rank=0,
    )


def compute_stage_metrics(pred_full: np.ndarray, true_full: np.ndarray, seq_len: int) -> tuple[pd.DataFrame, np.ndarray]:
    valid = np.isfinite(pred_full) & np.isfinite(true_full) & (np.arange(len(true_full)) >= seq_len)
    idx = np.where(valid)[0]
    pred = pred_full[valid]
    true = true_full[valid]
    err = pred - true

    wear_start = float(true[0])
    wear_end = float(true[-1])
    denom = max(wear_end - wear_start, 1e-12)
    progress = np.clip((true - wear_start) / denom, 0.0, 1.0)
    stage_id = np.digitize(progress, bins=[1 / 3, 2 / 3], right=False)

    names = ["early", "mid", "late"]
    rows = []
    for sid, name in enumerate(names):
        m = stage_id == sid
        e = err[m]
        rows.append(
            {
                "stage": name,
                "num_points": int(m.sum()),
                "mae_um": float(np.mean(np.abs(e))),
                "rmse_um": float(np.sqrt(np.mean(e ** 2))),
                "mean_bias_um": float(np.mean(e)),
                "underest_ratio": float(np.mean(e < 0)),
                "true_wear_min_um": float(np.min(true[m])),
                "true_wear_max_um": float(np.max(true[m])),
                "progress_min": float(np.min(progress[m])),
                "progress_max": float(np.max(progress[m])),
            }
        )
    df = pd.DataFrame(rows)
    full_stage = np.full_like(true_full, fill_value=-1, dtype=np.int32)
    full_stage[idx] = stage_id
    return df, full_stage


def plot_stage_curve(out_path: Path, true_full: np.ndarray, pred_full: np.ndarray, stage_full: np.ndarray, seq_len: int) -> None:
    colors = {0: "#5DADE2", 1: "#F5B041", 2: "#E74C3C"}
    labels = {0: "early", 1: "mid", 2: "late"}
    plt.figure(figsize=(11, 4.5))
    plt.plot(true_full, color="black", linewidth=2.0, label="true wear")
    plt.plot(pred_full, color="#1F77B4", linewidth=2.0, label="pred wear")
    for sid in [0, 1, 2]:
        idx = np.where(stage_full == sid)[0]
        if len(idx) == 0:
            continue
        plt.scatter(idx, true_full[idx], s=10, color=colors[sid], alpha=0.9, label=f"{labels[sid]} stage")
    plt.axvline(seq_len, linestyle="--", color="gray", label=f"forecast starts @ t={seq_len}")
    plt.title("Fold1 Current-best: full wear curve with stage coloring")
    plt.xlabel("walk index")
    plt.ylabel("wear (um)")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_stage_residual(out_path: Path, true_full: np.ndarray, pred_full: np.ndarray, stage_full: np.ndarray, seq_len: int) -> None:
    residual = pred_full - true_full
    colors = {0: "#5DADE2", 1: "#F5B041", 2: "#E74C3C"}
    plt.figure(figsize=(11, 4.2))
    plt.axhline(0.0, color="black", linewidth=1.2)
    for sid in [0, 1, 2]:
        idx = np.where(stage_full == sid)[0]
        if len(idx) == 0:
            continue
        plt.scatter(idx, residual[idx], s=12, color=colors[sid], alpha=0.9, label=["early", "mid", "late"][sid])
    plt.axvline(seq_len, linestyle="--", color="gray", label=f"forecast starts @ t={seq_len}")
    plt.title("Fold1 Current-best: residuals by wear stage")
    plt.xlabel("walk index")
    plt.ylabel("pred - true (um)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def write_summary(out_path: Path, df: pd.DataFrame) -> None:
    late = df[df["stage"] == "late"].iloc[0]
    mid = df[df["stage"] == "mid"].iloc[0]
    early = df[df["stage"] == "early"].iloc[0]
    lines = [
        "# Fold1 Current-best 分段误差分析",
        "",
        "对象：`RMS7 + Feat_4 + A2-Random-S2 + SPECTRAL_ENTROPY_CH1` 在 `c1,c4 -> c6` 上的整条磨损曲线。",
        "",
        "## 分段定义",
        "1. 主分析口径按真实 wear 进度三等分：`early=0%~33%`，`mid=33%~66%`，`late=66%~100%`。",
        f"2. 对应真实 wear 区间：",
        f"   - `early`: `{early['true_wear_min_um']:.4f} ~ {early['true_wear_max_um']:.4f}` um",
        f"   - `mid`: `{mid['true_wear_min_um']:.4f} ~ {mid['true_wear_max_um']:.4f}` um",
        f"   - `late`: `{late['true_wear_min_um']:.4f} ~ {late['true_wear_max_um']:.4f}` um",
        "3. `underest_ratio` 定义为：该阶段内满足 `pred < true` 的样本占比。",
        "",
        "## 结论",
        f"1. `late` 段 MAE = `{late['mae_um']:.4f}`，明显高于 `mid={mid['mae_um']:.4f}` 和 `early={early['mae_um']:.4f}`。",
        f"2. `late` 段平均残差 = `{late['mean_bias_um']:.4f}` um，若为负值则表示后段系统性低估。",
        f"3. `late` 段低估比例 = `{late['underest_ratio']:.4f}`，可用于判断后段是否长期预测偏低。",
        "",
        "## 建议",
        "1. 若 `late` 段确为主要误差来源，则下一步优先验证轻量训练域表示对齐（先 `CORAL`，后 `MMD`）。",
        "2. 只有当 `late` 段不仅误差最大，而且平均残差显著为负，且这种负偏差明显强于 `early/mid`，才考虑高磨损区低估惩罚的非对称损失。",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_currentbest"),
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="20260324_fold1_stage_error_currentbest",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_args = build_args(project_root, args.results_subdir)
    setting = exp_args.test_dir
    exp = Exp_Forecast(exp_args)
    result = exp.test(setting, test=1)

    pred_full = result["pred_full"].astype(np.float32)
    true_raw_full = result["true_raw_full_um"].astype(np.float32)
    seq_len = int(exp_args.test_seq_len)

    df, stage_full = compute_stage_metrics(pred_full, true_raw_full, seq_len)
    df.to_csv(out_dir / "fold1_stage_error_metrics.csv", index=False)
    np.savez(
        out_dir / "fold1_stage_error_arrays.npz",
        pred_full=pred_full,
        true_raw_full_um=true_raw_full,
        stage_id=stage_full,
    )

    plot_stage_curve(out_dir / "fold1_stage_curve_colored.png", true_raw_full, pred_full, stage_full, seq_len)
    plot_stage_residual(out_dir / "fold1_stage_residuals.png", true_raw_full, pred_full, stage_full, seq_len)
    write_summary(out_dir / "fold1_stage_error_summary.md", df)

    manifest = {
        "checkpoint_setting": exp_args.test_dir,
        "results_subdir": args.results_subdir,
        "seq_len": seq_len,
        "metrics_csv": str(out_dir / "fold1_stage_error_metrics.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] stage error analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
