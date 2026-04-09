#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
INNER = (
    ROOT
    / "paper_exec"
    / "train_only_param_selection"
    / "results_rms7_baseline_e300"
    / "inner_loso_knn_global_summary.csv"
)
OUTER = ROOT / "paper_exec" / "results" / "knn_outer_k_sweep_rms7_baseline_e300.csv"
OUTER_F133 = ROOT / "paper_exec" / "results" / "K外层扫描实验_Full133_Baseline_E300.csv"
OUTPUT = ROOT / "paper_exec" / "results" / "KNN参数敏感性分析汇总.csv"


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    inner_rows = load_csv(INNER)
    outer_rows = load_csv(OUTER)
    outer_f133_rows = load_csv(OUTER_F133)

    fieldnames = [
        "analysis_scope",
        "protocol",
        "feature_set",
        "backbone",
        "epoch",
        "k",
        "beta",
        "late_q",
        "mean_inner_validation_mae",
        "std_inner_validation_mae",
        "num_inner_tasks",
        "fold1_mae",
        "fold2_mae",
        "fold3_mae",
        "avg_outer_mae",
        "fold1_rmse",
        "fold2_rmse",
        "fold3_rmse",
        "avg_outer_rmse",
        "selected_for_mainline",
        "notes",
    ]

    rows: list[dict[str, str]] = []

    for row in inner_rows:
        selected = (
            row["k"] == "3" and row["beta"] == "1.0" and row["late_q"] == "0.0"
        )
        rows.append(
            {
                "analysis_scope": "inner_loso_grid",
                "protocol": "source-only inner-LOSO",
                "feature_set": "Pure RMS7",
                "backbone": "Baseline + KNN",
                "epoch": "300",
                "k": row["k"],
                "beta": row["beta"],
                "late_q": row["late_q"],
                "mean_inner_validation_mae": row["mean_inner_validation_mae"],
                "std_inner_validation_mae": row["std_inner_validation_mae"],
                "num_inner_tasks": row["num_inner_tasks"],
                "fold1_mae": "",
                "fold2_mae": "",
                "fold3_mae": "",
                "avg_outer_mae": "",
                "fold1_rmse": "",
                "fold2_rmse": "",
                "fold3_rmse": "",
                "avg_outer_rmse": "",
                "selected_for_mainline": "yes" if selected else "no",
                "notes": (
                    "Legal parameter search on the final mainline."
                    if selected
                    else "Full inner-LOSO search record for appendix."
                ),
            }
        )

    for row in outer_rows:
        selected = row["k"] == "3"
        rows.append(
            {
                "analysis_scope": "outer_loco_k_sweep",
                "protocol": "outer LOCO 3-fold",
                "feature_set": "Pure RMS7",
                "backbone": "Baseline + KNN",
                "epoch": "300",
                "k": row["k"],
                "beta": "1.0",
                "late_q": "0.0",
                "mean_inner_validation_mae": "",
                "std_inner_validation_mae": "",
                "num_inner_tasks": "",
                "fold1_mae": row["fold1_mae"],
                "fold2_mae": row["fold2_mae"],
                "fold3_mae": row["fold3_mae"],
                "avg_outer_mae": row["avg_mae"],
                "fold1_rmse": row["fold1_rmse"],
                "fold2_rmse": row["fold2_rmse"],
                "fold3_rmse": row["fold3_rmse"],
                "avg_outer_rmse": row["avg_rmse"],
                "selected_for_mainline": "yes" if selected else "no",
                "notes": (
                    "Outer-test k sweep at fixed beta=1.0 and late_q=0.0. "
                    "For robustness analysis only; not used to redefine the legal mainline."
                ),
            }
        )

    for row in outer_f133_rows:
        selected = row["k"] == "3"
        rows.append(
            {
                "analysis_scope": "outer_loco_k_sweep",
                "protocol": "outer LOCO 3-fold",
                "feature_set": "Full-133",
                "backbone": "Baseline + KNN",
                "epoch": "300",
                "k": row["k"],
                "beta": "1.0",
                "late_q": "0.0",
                "mean_inner_validation_mae": "",
                "std_inner_validation_mae": "",
                "num_inner_tasks": "",
                "fold1_mae": row["fold1_mae"],
                "fold2_mae": row["fold2_mae"],
                "fold3_mae": row["fold3_mae"],
                "avg_outer_mae": row["avg_mae"],
                "fold1_rmse": row["fold1_rmse"],
                "fold2_rmse": row["fold2_rmse"],
                "fold3_rmse": row["fold3_rmse"],
                "avg_outer_rmse": row["avg_rmse"],
                "selected_for_mainline": "yes" if selected else "no",
                "notes": (
                    "Outer-test k sweep on Full-133 at fixed beta=1.0 and late_q=0.0. "
                    "Appendix-only comparison against the RMS7 mainline."
                ),
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(OUTPUT)


if __name__ == "__main__":
    main()
