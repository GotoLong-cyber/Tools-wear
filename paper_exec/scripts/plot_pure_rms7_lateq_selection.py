#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
SUMMARY_CSV = PROJECT_ROOT / "paper_exec" / "train_only_param_selection" / "results_rms7_e300_lateq" / "inner_loso_knn_global_summary.csv"
FIG_DIR = PROJECT_ROOT / "paper_exec" / "figures"
OUT_PNG = FIG_DIR / "pure_rms7_lateq_selection_curve.png"
OUT_PDF = FIG_DIR / "pure_rms7_lateq_selection_curve.pdf"


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "late_q": float(row["late_q"]),
                    "mean_mae": float(row["mean_inner_validation_mae"]),
                    "std_mae": float(row["std_inner_validation_mae"]),
                }
            )
    return sorted(rows, key=lambda x: x["late_q"])


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(SUMMARY_CSV)
    best = min(rows, key=lambda x: x["mean_mae"])

    x = np.array([r["late_q"] for r in rows], dtype=float)
    y = np.array([r["mean_mae"] for r in rows], dtype=float)
    e = np.array([r["std_mae"] for r in rows], dtype=float)

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")

    ax.fill_between(x, y - e, y + e, color="#4c78a8", alpha=0.12, linewidth=0)
    ax.plot(
        x,
        y,
        color="#4c78a8",
        linewidth=2.4,
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=1.8,
        zorder=3,
    )
    ax.errorbar(
        x, y, yerr=e,
        fmt="none",
        ecolor="#4c78a8",
        elinewidth=1.0,
        capsize=3.0,
        alpha=0.7,
        zorder=4,
    )

    ax.scatter([best["late_q"]], [best["mean_mae"]], s=140, color="#111111", marker="*", zorder=5)
    ax.scatter([best["late_q"]], [best["mean_mae"]], s=300, facecolors="none", edgecolors="#111111", linewidths=1.2, zorder=4)
    ax.annotate(
        "Selected: late_q=0.0",
        xy=(best["late_q"], best["mean_mae"]),
        xytext=(0.18, best["mean_mae"] + 0.16),
        textcoords="data",
        fontsize=11,
        color="#111111",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#111111"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d0d0", alpha=0.96),
    )

    ax.set_title("Late-Library Quantile Selection on Inner-LOSO Validation", pad=12)
    ax.set_xlabel("Late-Library Quantile late_q")
    ax.set_ylabel("Mean Inner Validation MAE (um)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.1f}" for v in x])
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.16)

    y_min = float(np.min(y - e))
    y_max = float(np.max(y + e))
    margin = 0.14 * (y_max - y_min)
    ax.set_ylim(y_min - margin * 0.25, y_max + margin * 0.45)

    ax.text(
        0.02,
        0.02,
        "Fixed: k=3, beta=0.5; shaded band / error bar shows ±1 std across 6 inner tasks",
        transform=ax.transAxes,
        fontsize=10,
        color="#666666",
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
