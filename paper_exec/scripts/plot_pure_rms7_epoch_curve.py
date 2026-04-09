#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


EPOCHS = np.array([100, 150, 200, 250, 300, 400, 500, 600], dtype=float)
RMSE = np.array([8.1548, 5.9181, 5.5836, 5.1319, 5.0665, 5.3717, 6.9473, 8.7141], dtype=float)
MAE = np.array([6.9327, 5.3308, 5.2127, 4.6459, 4.4243, 4.6247, 6.3011, 8.0013], dtype=float)


def draw_panel(ax, x, y, ylabel, color):
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=3,
    )
    ax.fill_between(x, y, y.min() - 0.15, color=color, alpha=0.08, zorder=1)
    ax.axvspan(200, 300, color="#D9EAF7", alpha=0.45, zorder=0)
    ax.scatter([200], [y[list(x).index(200)]], s=95, color="#E69F00", zorder=4, label="Current choice")
    best_idx = int(np.argmin(y))
    ax.scatter([x[best_idx]], [y[best_idx]], s=110, color="#D62728", zorder=5, label="Best on fold1")
    ax.annotate(
        f"best: {int(x[best_idx])}",
        xy=(x[best_idx], y[best_idx]),
        xytext=(12, -18),
        textcoords="offset points",
        fontsize=10,
        color="#A61E1E",
        weight="bold",
    )
    ax.annotate(
        "selected: 200",
        xy=(200, y[list(x).index(200)]),
        xytext=(-54, 18),
        textcoords="offset points",
        fontsize=9,
        color="#9A6700",
        weight="bold",
    )
    ax.set_xlabel("Training Epochs", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(x)
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10)


def main():
    out_dir = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), dpi=220)

    draw_panel(axes[0], EPOCHS, MAE, "MAE (um)", "#1f77b4")
    axes[0].set_title("pure RMS7, fold1, MAE", fontsize=12, weight="bold")

    draw_panel(axes[1], EPOCHS, RMSE, "RMSE (um)", "#2ca02c")
    axes[1].set_title("pure RMS7, fold1, RMSE", fontsize=12, weight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Epoch Sensitivity Under the New pure RMS7 Protocol", fontsize=14, weight="bold", y=1.08)
    fig.tight_layout()

    png_path = out_dir / "pure_rms7_epoch_curve_fold1.png"
    pdf_path = out_dir / "pure_rms7_epoch_curve_fold1.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
