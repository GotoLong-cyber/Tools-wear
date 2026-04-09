#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
INPUT = ROOT / "paper_exec" / "论文级核心对照表_分epoch.csv"
OUT_CSV = ROOT / "paper_exec" / "tables" / "论文级核心对照表_打勾矩阵_epoch300.csv"
OUT_TEX = ROOT / "paper_exec" / "tables" / "论文级核心对照表_打勾矩阵_epoch300.tex"


CHECK = "✓"
BLANK = "-"


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols: list[str] = []
    for col in df.columns:
        if not isinstance(col, tuple):
            flat_cols.append(str(col))
            continue
        left = (col[0] or "").strip()
        right = (col[1] or "").strip()
        if left in {"C1", "C4", "C6", "Avg"} and right in {"MAE", "RMSE"}:
            flat_cols.append(f"{left}_{right}")
        elif left:
            flat_cols.append(left)
        elif right:
            flat_cols.append(right)
        else:
            flat_cols.append("")
    df = df.copy()
    df.columns = flat_cols
    return df


def load_input() -> pd.DataFrame:
    df = pd.read_csv(INPUT, header=[0, 1], encoding="utf-8-sig")
    df = flatten_columns(df)

    # Forward-fill the merged epoch column produced by the wide CSV export.
    df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce").ffill()

    # Normalize numeric columns.
    for col in ["C1_MAE", "C1_RMSE", "C4_MAE", "C4_RMSE", "C6_MAE", "C6_RMSE", "Avg_MAE", "Avg_RMSE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Epoch"] == 300].copy()

    feature = df["特征"].fillna("")
    method = df["Method (模型方法)"].fillna("")

    matrix = pd.DataFrame(index=df.index)
    matrix["Base Feat (7D)"] = feature.apply(
        lambda x: CHECK if ("RMS-7" in x and "133" not in x) else BLANK
    )
    matrix["Complex Feat"] = feature.apply(
        lambda x: (
            "✓(133D)" if "133D" in x else
            "✓(21D)" if "21D" in x else
            "✓(14D)" if "14D" in x else
            BLANK
        )
    )
    matrix["Reg. Head"] = method.apply(
        lambda x: CHECK if ("Head-only" in x or "blend" in x) else BLANK
    )
    matrix["TMA"] = method.apply(lambda x: CHECK if "TMA" in x else BLANK)
    matrix["KNN Retrieval"] = method.apply(lambda x: CHECK if "KNN" in x else BLANK)

    def pack(mae_col: str, rmse_col: str) -> pd.Series:
        return df.apply(lambda row: f"{row[mae_col]:.4f} / {row[rmse_col]:.4f}", axis=1)

    matrix["C1 Target (MAE/RMSE)"] = pack("C1_MAE", "C1_RMSE")
    matrix["C4 Target (MAE/RMSE)"] = pack("C4_MAE", "C4_RMSE")
    matrix["C6 Target (MAE/RMSE)"] = pack("C6_MAE", "C6_RMSE")
    matrix["Avg (MAE/RMSE)"] = pack("Avg_MAE", "Avg_RMSE")

    c6_best_idx = df["C6_MAE"].idxmin()
    avg_best_idx = df["Avg_MAE"].idxmin()
    matrix.loc[c6_best_idx, "C6 Target (MAE/RMSE)"] += " *"
    matrix.loc[avg_best_idx, "Avg (MAE/RMSE)"] += " *"

    return matrix.reset_index(drop=True)


def to_latex(df: pd.DataFrame) -> str:
    latex_df = df.copy()
    latex_df["C6 Target (MAE/RMSE)"] = latex_df["C6 Target (MAE/RMSE)"].str.replace(
        " *", r" $^{\star}$", regex=False
    )
    latex_df["Avg (MAE/RMSE)"] = latex_df["Avg (MAE/RMSE)"].str.replace(
        " *", r" $^{\star}$", regex=False
    )

    headers = list(latex_df.columns)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation check-mark matrix under the unified epoch-300 protocol.}",
        r"\label{tab:ablation_checkmark_matrix_epoch300}",
        r"\begin{tabular}{lllllcccc}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]

    for _, row in latex_df.iterrows():
        cells = [str(row[h]) for h in headers]
        lines.append(" & ".join(cells) + r" \\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    df = load_input()
    matrix = build_matrix(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    latex_str = to_latex(matrix)
    OUT_TEX.write_text(latex_str + "\n", encoding="utf-8")

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(matrix.to_string(index=False))
    print("\n" + "=" * 80 + "\n")
    print(latex_str)
    print("\n" + "=" * 80 + "\n")
    print(f"Saved CSV to: {OUT_CSV}")
    print(f"Saved LaTeX to: {OUT_TEX}")


if __name__ == "__main__":
    main()
