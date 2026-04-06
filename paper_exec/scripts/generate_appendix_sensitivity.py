#!/usr/bin/env python3
"""Generate appendix sensitivity assets from inner-LOSO validation summaries.

This script uses train-only inner-LOSO validation outputs only. It does not read
outer-test results for model selection, so it is safe for robustness appendix use.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER_EXEC = ROOT / "paper_exec"
CSV_DIR = PAPER_EXEC / "csv"
TABLE_DIR = PAPER_EXEC / "tables"
FIG_DIR = PAPER_EXEC / "figures"
SRC = PAPER_EXEC / "train_only_param_selection" / "results" / "inner_loso_knn_global_summary.csv"


def write_md_table(df: pd.DataFrame, path: Path, title: str) -> None:
    cols = list(df.columns)
    lines = [f"# {title}", "", "| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = pd.read_csv(SRC)
    df = df.sort_values(["mean_inner_validation_mae", "std_inner_validation_mae", "k"]).reset_index(drop=True)
    best = df.iloc[0]
    out = df.copy()
    out["rank"] = range(1, len(out) + 1)
    out["delta_to_best"] = out["mean_inner_validation_mae"] - float(best["mean_inner_validation_mae"])
    out = out[["rank", "k", "beta", "late_q", "mean_inner_validation_mae", "std_inner_validation_mae", "delta_to_best", "num_inner_tasks"]]
    out = out.round({"beta": 2, "late_q": 2, "mean_inner_validation_mae": 4, "std_inner_validation_mae": 4, "delta_to_best": 4})

    csv_path = CSV_DIR / "appendix_inner_loso_sensitivity.csv"
    out.to_csv(csv_path, index=False)

    top8 = out.head(8).copy()
    write_md_table(top8, TABLE_DIR / "表A2_InnerLOSO敏感性.md", "Table A2 Inner-LOSO Sensitivity Summary")
    write_md_table(top8, TABLE_DIR / "附录_InnerLOSO敏感性.md", "Appendix Inner-LOSO Sensitivity Summary")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)

    for q, group in df.groupby("late_q"):
        g = group.sort_values("beta")
        axes[0].plot(g["beta"], g["mean_inner_validation_mae"], marker="o", linewidth=2, label=f"late_q={q}")
    axes[0].scatter([best["beta"]], [best["mean_inner_validation_mae"]], color="red", s=70, zorder=5)
    axes[0].set_xlabel("beta")
    axes[0].set_ylabel("Mean inner-validation MAE")
    axes[0].set_title("Sensitivity over beta and late_q")
    axes[0].grid(alpha=0.25, linestyle="--")
    axes[0].legend(frameon=False)

    for beta, group in df.groupby("beta"):
        g = group.sort_values("k")
        axes[1].plot(g["k"], g["mean_inner_validation_mae"], marker="o", linewidth=2, label=f"beta={beta}")
    axes[1].scatter([best["k"]], [best["mean_inner_validation_mae"]], color="red", s=70, zorder=5)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Mean inner-validation MAE")
    axes[1].set_title("Sensitivity over k")
    axes[1].grid(alpha=0.25, linestyle="--")
    axes[1].legend(frameon=False)

    stem = FIG_DIR / "FigA1_inner_loso_sensitivity"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
