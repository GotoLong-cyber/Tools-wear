#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
INPUT = ROOT / "paper_exec" / "results" / "K外层扫描实验_Full133_Baseline_E300.csv"
OUT_PNG = ROOT / "paper_exec" / "figures" / "FigA3_outer_loco_k_sweep_full133.png"
OUT_PDF = ROOT / "paper_exec" / "figures" / "FigA3_outer_loco_k_sweep_full133.pdf"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open("r", encoding="utf-8-sig", newline="")))
    ks = [int(r["k"]) for r in rows]
    maes = [float(r["avg_mae"]) for r in rows]
    rmses = [float(r["avg_rmse"]) for r in rows]

    plt.figure(figsize=(7.0, 4.4))
    plt.plot(ks, maes, marker="o", linewidth=2.0, markersize=6, color="#355c7d", label="Avg MAE")
    plt.plot(ks, rmses, marker="s", linewidth=1.8, markersize=5.5, color="#c06c84", label="Avg RMSE")

    selected_k = 3
    idx = ks.index(selected_k)
    plt.scatter([selected_k], [maes[idx]], color="#8b0000", s=60, zorder=4)
    plt.annotate(
        "Legal selection: k=3",
        xy=(selected_k, maes[idx]),
        xytext=(4.2, maes[idx] + 0.04),
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"),
        fontsize=10,
        color="#333333",
    )

    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Outer LOCO Error")
    plt.title("Outer-LOCO k Sweep on Full-133 + Baseline + KNN")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
