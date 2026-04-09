#!/usr/bin/env python3
"""Plot legal inner-LOSO hyperparameter selection for pure RMS7 baseline+KNN."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
RESULTS_DIR = PROJECT_ROOT / "paper_exec" / "train_only_param_selection" / "results_rms7_baseline_e300"
FIG_DIR = PROJECT_ROOT / "paper_exec" / "figures"
SUMMARY_CSV = RESULTS_DIR / "inner_loso_knn_global_summary.csv"


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_heatmap_arrays(rows: list[dict], late_q: float, betas: list[float], ks: list[int]) -> np.ndarray:
    arr = np.full((len(betas), len(ks)), np.nan, dtype=float)
    for i, beta in enumerate(betas):
        for j, k in enumerate(ks):
            match = [
                r for r in rows
                if float(r["late_q"]) == late_q and float(r["beta"]) == beta and int(r["k"]) == k
            ]
            if match:
                arr[i, j] = float(match[0]["mean_inner_validation_mae"])
    return arr


def main() -> None:
    rows = load_rows(SUMMARY_CSV)
    ks = sorted({int(r["k"]) for r in rows})
    betas = sorted({float(r["beta"]) for r in rows})
    late_qs = sorted({float(r["late_q"]) for r in rows})

    best = min(rows, key=lambda r: (float(r["mean_inner_validation_mae"]), float(r["std_inner_validation_mae"]), int(r["k"]), float(r["beta"]), float(r["late_q"])))
    best_k = int(best["k"])
    best_beta = float(best["beta"])
    best_q = float(best["late_q"])

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
    })

    fig, axes = plt.subplots(1, len(late_qs), figsize=(12.5, 3.9), constrained_layout=True)
    if len(late_qs) == 1:
        axes = [axes]

    vmin = min(float(r["mean_inner_validation_mae"]) for r in rows)
    vmax = max(float(r["mean_inner_validation_mae"]) for r in rows)

    for ax, late_q in zip(axes, late_qs):
        arr = build_heatmap_arrays(rows, late_q, betas, ks)
        im = ax.imshow(arr, cmap="YlGnBu_r", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"late_q = {late_q:.1f}")
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(betas)), [f"{b:.1f}" for b in betas])
        ax.set_xlabel("Number of Neighbors k")
        if ax is axes[0]:
            ax.set_ylabel("Blend Weight beta")

        for i, beta in enumerate(betas):
            for j, k in enumerate(ks):
                value = arr[i, j]
                if np.isnan(value):
                    continue
                color = "white" if value < (vmin + vmax) / 2 else "#16324f"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)
                if k == best_k and abs(beta - best_beta) < 1e-9 and abs(late_q - best_q) < 1e-9:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="#c7253e", linewidth=2.2))

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#666666")

    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label("Mean Inner Validation MAE (um)")

    fig.suptitle("Legal Hyperparameter Selection on Baseline + KNN (pure RMS7, epoch=300)", y=1.03, fontsize=13)
    fig.text(
        0.5, -0.02,
        f"Selected: k={best_k}, beta={best_beta:.1f}, late_q={best_q:.1f}",
        ha="center", fontsize=10, color="#8b1e3f"
    )

    png = FIG_DIR / "pure_rms7_baseline_knn_selection_heatmap.png"
    pdf = FIG_DIR / "pure_rms7_baseline_knn_selection_heatmap.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
