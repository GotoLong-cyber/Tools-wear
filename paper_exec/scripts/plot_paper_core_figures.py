#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
CORE_CSV = ROOT / "paper_exec" / "论文级核心对照表_分epoch.csv"
INNER_KNN = (
    ROOT
    / "paper_exec"
    / "train_only_param_selection"
    / "results_rms7_baseline_e300"
    / "inner_loso_knn_global_summary.csv"
)
OUTER_RMS7 = ROOT / "paper_exec" / "results" / "K外层扫描实验_RMS7_Baseline_E300.csv"
OUTER_F133 = ROOT / "paper_exec" / "results" / "K外层扫描实验_Full133_Baseline_E300.csv"
FIG_DIR = ROOT / "paper_exec" / "figures"


def flatten_core_table() -> pd.DataFrame:
    df = pd.read_csv(CORE_CSV, header=[0, 1], encoding="utf-8-sig")
    flat_cols: list[str] = []
    for col in df.columns:
        left = (col[0] or "").strip()
        right = (col[1] or "").strip()
        if left in {"C1", "C4", "C6", "Avg"} and right in {"MAE", "RMSE"}:
            flat_cols.append(f"{left}_{right}")
        elif left:
            flat_cols.append(left)
        elif right:
            flat_cols.append(right)
        else:
            flat_cols.append("")
    df.columns = flat_cols
    df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce").ffill()
    return df


def style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_knn_sensitivity() -> None:
    inner = pd.read_csv(INNER_KNN, encoding="utf-8-sig")
    inner = inner[(inner["beta"] == 1.0) & (inner["late_q"] == 0.0)].copy()
    inner = inner.sort_values("k")

    outer_rms7 = pd.read_csv(OUTER_RMS7, encoding="utf-8-sig").sort_values("k")
    outer_f133 = pd.read_csv(OUTER_F133, encoding="utf-8-sig").sort_values("k")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))

    ax = axes[0]
    ks = inner["k"].to_numpy()
    means = inner["mean_inner_validation_mae"].to_numpy()
    stds = inner["std_inner_validation_mae"].to_numpy()
    ax.plot(ks, means, color="#1f4e79", marker="o", linewidth=2.2, label="Inner-LOSO MAE")
    ax.fill_between(ks, means - stds, means + stds, color="#1f4e79", alpha=0.12)
    best_k = 3
    best_y = float(inner.loc[inner["k"] == best_k, "mean_inner_validation_mae"].iloc[0])
    ax.scatter([best_k], [best_y], color="#8b0000", s=55, zorder=4)
    ax.annotate(
        "Selected k=3",
        xy=(best_k, best_y),
        xytext=(4.2, best_y + 0.07),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"),
        fontsize=9,
        color="#333333",
    )
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Mean Inner Validation MAE")
    ax.set_title("Legal Inner-LOSO Selection")
    style_axes(ax)

    ax = axes[1]
    ax.plot(
        outer_rms7["k"], outer_rms7["avg_mae"],
        color="#355c7d", marker="o", linewidth=2.2, label="Pure RMS7"
    )
    ax.plot(
        outer_f133["k"], outer_f133["avg_mae"],
        color="#c06c84", marker="s", linewidth=2.0, label="Full-133"
    )
    ax.scatter([3], [float(outer_rms7.loc[outer_rms7["k"] == 3, "avg_mae"].iloc[0])], color="#8b0000", s=55, zorder=4)
    ax.set_xlabel("Number of Neighbors (k)")
    ax.set_ylabel("Outer LOCO Avg MAE")
    ax.set_title("Outer-LOCO Robustness")
    style_axes(ax)
    ax.legend(frameon=False)

    fig.suptitle("KNN Parameter Sensitivity", fontsize=13, y=1.02)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "Fig_main_knn_sensitivity.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_main_knn_sensitivity.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_ablation_bar() -> None:
    df = flatten_core_table()
    df = df[df["Epoch"] == 300].copy()
    target = df[df["特征"].str.contains("RMS-7 \\(7D\\), epoch=300", regex=True)]

    order = ["Head-only", "TMA", "KNN", "KNN+TMA"]
    label_map = {
        "Head-only": "Baseline",
        "TMA": "+ TMA",
        "KNN": "+ KNN",
        "KNN+TMA": "+ TMA + KNN",
    }
    target = target[target["Method (模型方法)"].isin(order)].copy()
    target["order"] = target["Method (模型方法)"].map({k: i for i, k in enumerate(order)})
    target = target.sort_values("order")

    labels = [label_map[m] for m in target["Method (模型方法)"]]
    mae = target["Avg_MAE"].to_numpy()
    rmse = target["Avg_RMSE"].to_numpy()
    x = range(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.bar([i - width / 2 for i in x], mae, width=width, color="#4c78a8", label="Avg MAE")
    ax.bar([i + width / 2 for i in x], rmse, width=width, color="#f58518", label="Avg RMSE")

    best_idx = int(pd.Series(mae).idxmin())
    ax.annotate(
        "Best MAE",
        xy=(best_idx - width / 2, mae[best_idx]),
        xytext=(best_idx - 0.35, mae[best_idx] + 0.35),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"),
        fontsize=9,
        color="#333333",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Error")
    ax.set_title("Ablation on Pure RMS7 Mainline (Epoch=300)")
    style_axes(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_main_ablation_bar.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_main_ablation_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_feature_bar() -> None:
    df = flatten_core_table()
    df = df[df["Epoch"] == 300].copy()

    feature_order = [
        "RMS-7 (7D), epoch=300",
        "RMS-7 + PTP-7 (14D)",
        "RMS-7 + PTP-7 + WaveletEnergy-7 (21D)",
        "Full-133 (133D)",
    ]
    short_names = {
        "RMS-7 (7D), epoch=300": "RMS-7",
        "RMS-7 + PTP-7 (14D)": "RMS-7+PTP-7",
        "RMS-7 + PTP-7 + WaveletEnergy-7 (21D)": "RMS-7+PTP-7+WAV-7",
        "Full-133 (133D)": "Full-133",
    }

    head = df[df["Method (模型方法)"] == "Head-only"].copy()
    knn = df[df["Method (模型方法)"] == "KNN"].copy()
    head = head.set_index("特征").loc[feature_order].reset_index()
    knn = knn.set_index("特征").loc[feature_order].reset_index()

    labels = [short_names[f] for f in feature_order]
    x = range(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharex=True)

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], head["Avg_MAE"], width=width, color="#7a5195", label="Head-only")
    ax.bar([i + width / 2 for i in x], knn["Avg_MAE"], width=width, color="#2a9d8f", label="+ KNN")
    ax.set_ylabel("Avg MAE")
    ax.set_title("Average MAE")
    style_axes(ax)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], head["Avg_RMSE"], width=width, color="#7a5195", label="Head-only")
    ax.bar([i + width / 2 for i in x], knn["Avg_RMSE"], width=width, color="#2a9d8f", label="+ KNN")
    ax.set_ylabel("Avg RMSE")
    ax.set_title("Average RMSE")
    style_axes(ax)

    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=12, ha="right")

    fig.suptitle("Feature Comparison Under Unified Epoch-300 Protocol", fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "Fig_main_feature_comparison_bar.png", dpi=320, bbox_inches="tight")
    fig.savefig(FIG_DIR / "Fig_main_feature_comparison_bar.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plot_knn_sensitivity()
    plot_ablation_bar()
    plot_feature_bar()
    print(FIG_DIR / "Fig_main_knn_sensitivity.png")
    print(FIG_DIR / "Fig_main_knn_sensitivity.pdf")
    print(FIG_DIR / "Fig_main_ablation_bar.png")
    print(FIG_DIR / "Fig_main_ablation_bar.pdf")
    print(FIG_DIR / "Fig_main_feature_comparison_bar.png")
    print(FIG_DIR / "Fig_main_feature_comparison_bar.pdf")


if __name__ == "__main__":
    main()
