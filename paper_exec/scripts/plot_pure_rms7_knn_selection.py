#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
SUMMARY_CSV = PROJECT_ROOT / "paper_exec" / "train_only_param_selection" / "results_rms7_e300" / "inner_loso_knn_global_summary.csv"
FIG_DIR = PROJECT_ROOT / "paper_exec" / "figures"
OUT_PNG = FIG_DIR / "pure_rms7_knn_selection_curve.png"
OUT_PDF = FIG_DIR / "pure_rms7_knn_selection_curve.pdf"


def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "k": int(row["k"]),
                    "beta": float(row["beta"]),
                    "late_q": float(row["late_q"]),
                    "mean_mae": float(row["mean_inner_validation_mae"]),
                    "std_mae": float(row["std_inner_validation_mae"]),
                }
            )
    return rows


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(SUMMARY_CSV)
    rows = [r for r in rows if abs(r["late_q"] - 0.8) < 1e-9]

    beta_order = sorted({r["beta"] for r in rows})
    k_values = sorted({r["k"] for r in rows})
    best = min(rows, key=lambda x: x["mean_mae"])

    color_map = {
        0.0: "#9aa0a6",
        0.3: "#4c78a8",
        0.5: "#e45756",
        0.7: "#72b7b2",
        1.0: "#f2a541",
    }

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

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=220)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfc")

    for beta in beta_order:
        beta_rows = sorted([r for r in rows if abs(r["beta"] - beta) < 1e-9], key=lambda x: x["k"])
        xs = np.array([r["k"] for r in beta_rows], dtype=float)
        ys = np.array([r["mean_mae"] for r in beta_rows], dtype=float)
        es = np.array([r["std_mae"] for r in beta_rows], dtype=float)
        color = color_map.get(beta, "#333333")

        ax.fill_between(xs, ys - es, ys + es, color=color, alpha=0.10, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=rf"$\beta={beta:.1f}$",
            zorder=3,
        )

    ax.scatter(
        [best["k"]],
        [best["mean_mae"]],
        s=140,
        color="#111111",
        marker="*",
        zorder=5,
        label="Selected",
    )
    ax.scatter(
        [best["k"]],
        [best["mean_mae"]],
        s=320,
        facecolors="none",
        edgecolors="#111111",
        linewidths=1.2,
        zorder=4,
    )

    ax.annotate(
        "Selected: k=3, beta=0.5",
        xy=(best["k"], best["mean_mae"]),
        xytext=(4.5, best["mean_mae"] + 0.23),
        textcoords="data",
        fontsize=11,
        color="#111111",
        arrowprops=dict(arrowstyle="->", lw=1.1, color="#111111"),
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d0d0d0", alpha=0.96),
    )

    ax.set_title("Pure RMS7 Inner-LOSO KNN Hyperparameter Selection", pad=12)
    ax.set_xlabel("Number of Neighbors k")
    ax.set_ylabel("Mean Inner Validation MAE (um)")
    ax.set_xticks(k_values)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.28)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.15)

    y_min = min(r["mean_mae"] - r["std_mae"] for r in rows)
    y_max = max(r["mean_mae"] + r["std_mae"] for r in rows)
    margin = 0.12 * (y_max - y_min)
    ax.set_ylim(y_min - margin * 0.35, y_max + margin * 0.65)

    leg = ax.legend(
        loc="upper right",
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        borderpad=0.6,
        handlelength=2.2,
    )
    leg.get_frame().set_edgecolor("#d8d8d8")
    leg.get_frame().set_linewidth(0.8)

    ax.text(
        0.015,
        0.02,
        "Shaded bands show ±1 std across 6 inner tasks",
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
