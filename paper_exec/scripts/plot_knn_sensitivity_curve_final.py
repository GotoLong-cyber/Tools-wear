#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
INPUT = ROOT / "paper_exec" / "results" / "knn_outer_k_sweep_rms7_baseline_e300.csv"
OUT_PNG = ROOT / "paper_exec" / "figures" / "knn_sensitivity_curve.png"
OUT_PDF = ROOT / "paper_exec" / "figures" / "knn_sensitivity_curve.pdf"


def main() -> None:
    rows = list(csv.DictReader(INPUT.open("r", encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["k"]))

    ks = [int(r["k"]) for r in rows]
    maes = [float(r["avg_mae"]) for r in rows]
    stds = [
        statistics.pstdev(
            [float(r["fold1_mae"]), float(r["fold2_mae"]), float(r["fold3_mae"])]
        )
        for r in rows
    ]

    plt.figure(figsize=(6.6, 4.2))
    plt.plot(ks, maes, color="#1f4e79", marker="o", linewidth=2.0, markersize=6)
    lower = [m - s for m, s in zip(maes, stds)]
    upper = [m + s for m, s in zip(maes, stds)]
    plt.fill_between(ks, lower, upper, color="#9dc3e6", alpha=0.25)
    selected_y = maes[ks.index(3)]
    plt.scatter([3], [selected_y], color="#b00020", s=55, zorder=3)
    plt.annotate("Legal selection: k=3", xy=(3, selected_y), xytext=(3.9, selected_y + 0.03),
                 arrowprops=dict(arrowstyle="->", lw=0.8, color="#444444"),
                 fontsize=10, color="#333333")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Mean Outer LOCO MAE (um)")
    plt.title("Outer-LOCO Sensitivity of k (beta=1.0, late_q=0.0)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    print(OUT_PNG)
    print(OUT_PDF)


if __name__ == "__main__":
    main()
