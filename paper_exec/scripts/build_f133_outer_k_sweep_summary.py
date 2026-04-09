#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
RESULTS_ROOT = ROOT / "results"
OUT_CSV = ROOT / "paper_exec" / "results" / "K外层扫描实验_Full133_Baseline_E300.csv"
OUT_MD = ROOT / "paper_exec" / "tables" / "K外层扫描实验_Full133_Baseline_E300.md"

FOLDS = {
    "fold1": "c6",
    "fold2": "c1",
    "fold3": "c4",
}
K_VALUES = [1, 3, 5, 7, 9, 11, 15]


def load_metric(path: Path) -> tuple[float, float]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["mode"].startswith("delta-knn-only@"):
            return float(row["rmse_full_raw"]), float(row["mae_full_raw"])
    raise RuntimeError(f"delta-knn-only row not found in {path}")


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for k in K_VALUES:
        fold_metrics: dict[str, tuple[float, float]] = {}
        for fold in FOLDS:
            metrics_path = (
                RESULTS_ROOT
                / f"20260410_f133ksweep_{fold}_k{k}"
                / f"knn_delta_f133ksweep_{fold}_k{k}_overall_metrics.csv"
            )
            rmse, mae = load_metric(metrics_path)
            fold_metrics[fold] = (rmse, mae)

        avg_rmse = sum(v[0] for v in fold_metrics.values()) / 3.0
        avg_mae = sum(v[1] for v in fold_metrics.values()) / 3.0
        rows.append(
            {
                "k": str(k),
                "fold1_rmse": f"{fold_metrics['fold1'][0]:.4f}",
                "fold1_mae": f"{fold_metrics['fold1'][1]:.4f}",
                "fold2_rmse": f"{fold_metrics['fold2'][0]:.4f}",
                "fold2_mae": f"{fold_metrics['fold2'][1]:.4f}",
                "fold3_rmse": f"{fold_metrics['fold3'][0]:.4f}",
                "fold3_mae": f"{fold_metrics['fold3'][1]:.4f}",
                "avg_rmse": f"{avg_rmse:.4f}",
                "avg_mae": f"{avg_mae:.4f}",
                "selected_by_inner_loso": "yes" if k == 3 else "no",
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], out: Path) -> None:
    fieldnames = [
        "k",
        "fold1_mae",
        "fold1_rmse",
        "fold2_mae",
        "fold2_rmse",
        "fold3_mae",
        "fold3_rmse",
        "avg_mae",
        "avg_rmse",
        "selected_by_inner_loso",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "k": row["k"],
                    "fold1_mae": row["fold1_mae"],
                    "fold1_rmse": row["fold1_rmse"],
                    "fold2_mae": row["fold2_mae"],
                    "fold2_rmse": row["fold2_rmse"],
                    "fold3_mae": row["fold3_mae"],
                    "fold3_rmse": row["fold3_rmse"],
                    "avg_mae": row["avg_mae"],
                    "avg_rmse": row["avg_rmse"],
                    "selected_by_inner_loso": row["selected_by_inner_loso"],
                }
            )


def write_md(rows: list[dict[str, str]], out: Path) -> None:
    lines = [
        "## 表A：Full-133 主线在外层 LOCO 下的 k 扫描",
        "",
        "- 协议：`Full-133 + Baseline + KNN`",
        "- 固定：`beta=1.0, late_q=0.0, epoch=300`",
        "- 说明：`k=3` 为当前合法 inner-LOSO 主线选参得到的参考点；本表用于附录外层稳健性分析。",
        "",
        "| k | Fold1 MAE | Fold2 MAE | Fold3 MAE | 平均 MAE | Fold1 RMSE | Fold2 RMSE | Fold3 RMSE | 平均 RMSE | 主线参数 |",
        "|---|----------:|----------:|----------:|---------:|-----------:|-----------:|-----------:|----------:|---------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['k']} | {row['fold1_mae']} | {row['fold2_mae']} | {row['fold3_mae']} | "
            f"{row['avg_mae']} | {row['fold1_rmse']} | {row['fold2_rmse']} | {row['fold3_rmse']} | "
            f"{row['avg_rmse']} | {'yes' if row['k'] == '3' else ''} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows, OUT_CSV)
    write_md(rows, OUT_MD)
    print(OUT_CSV)
    print(OUT_MD)


if __name__ == "__main__":
    main()
