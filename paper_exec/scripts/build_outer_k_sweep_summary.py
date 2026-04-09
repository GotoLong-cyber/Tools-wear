#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
RMS7_INPUT = ROOT / "paper_exec" / "results" / "knn_outer_k_sweep_rms7_baseline_e300.csv"
RMS7_OUTPUT = ROOT / "paper_exec" / "results" / "K外层扫描实验_RMS7_Baseline_E300.csv"
RMS7_MD = ROOT / "paper_exec" / "tables" / "K外层扫描实验_RMS7_Baseline_E300.md"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict[str, str]], out: Path) -> None:
    fieldnames = [
        "k",
        "fold1_mae", "fold1_rmse",
        "fold2_mae", "fold2_rmse",
        "fold3_mae", "fold3_rmse",
        "avg_mae", "avg_rmse",
        "selected_by_inner_loso",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "k": r["k"],
                "fold1_mae": r["fold1_mae"],
                "fold1_rmse": r["fold1_rmse"],
                "fold2_mae": r["fold2_mae"],
                "fold2_rmse": r["fold2_rmse"],
                "fold3_mae": r["fold3_mae"],
                "fold3_rmse": r["fold3_rmse"],
                "avg_mae": r["avg_mae"],
                "avg_rmse": r["avg_rmse"],
                "selected_by_inner_loso": "yes" if r["k"] == "3" else "no",
            })


def write_md(rows: list[dict[str, str]], out: Path) -> None:
    lines = [
        "## 表A：RMS7 主线在外层 LOCO 下的 k 扫描",
        "",
        "- 协议：`Pure RMS7 + Baseline + KNN`",
        "- 固定：`beta=1.0, late_q=0.0, epoch=300`",
        "- 说明：`k=3` 为合法 inner-LOSO 选参得到的主结果参数；本表仅作 outer-test 稳健性分析。",
        "",
        "| k | Fold1 MAE | Fold2 MAE | Fold3 MAE | 平均 MAE | Fold1 RMSE | Fold2 RMSE | Fold3 RMSE | 平均 RMSE | 主线参数 |",
        "|---|----------:|----------:|----------:|---------:|-----------:|-----------:|-----------:|----------:|---------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['k']} | {float(r['fold1_mae']):.4f} | {float(r['fold2_mae']):.4f} | {float(r['fold3_mae']):.4f} | "
            f"{float(r['avg_mae']):.4f} | {float(r['fold1_rmse']):.4f} | {float(r['fold2_rmse']):.4f} | {float(r['fold3_rmse']):.4f} | "
            f"{float(r['avg_rmse']):.4f} | {'yes' if r['k']=='3' else ''} |"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows(RMS7_INPUT)
    write_csv(rows, RMS7_OUTPUT)
    write_md(rows, RMS7_MD)
    print(RMS7_OUTPUT)
    print(RMS7_MD)


if __name__ == "__main__":
    main()
