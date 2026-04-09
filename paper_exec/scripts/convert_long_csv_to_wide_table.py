#!/usr/bin/env python3
"""Convert the long-format experiment CSV into a paper-style wide table.

Input columns:
    ['feature_set', 'dim', 'epoch', 'setting', 'fold', 'rmse', 'mae', 'notes']

Output columns (MultiIndex):
    Method (模型方法) | 特征 | 损失函数 | C1(MAE/RMSE) | C4(MAE/RMSE) | C6(MAE/RMSE) | Note (备注)

Fold mapping:
    fold2 -> C1
    fold3 -> C4
    fold1 -> C6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


FOLD_TO_DOMAIN = {
    "fold2": "C1",
    "fold3": "C4",
    "fold1": "C6",
}

FEATURE_NOTES = {
    "full133": "7通道×19统计特征（时域+频域）",
    "pure_rms7": "7通道RMS",
    "rms7_ptp7": "7通道RMS + 7通道PTP",
    "rms7_wav7": "7通道RMS + 7通道小波能量",
    "rms7_ptp7_wav7": "7通道RMS + 7通道PTP + 7通道小波能量",
}

FEATURE_DISPLAY = {
    "full133": "Full-133",
    "pure_rms7": "RMS-7",
    "rms7_ptp7": "RMS-7 + PTP-7",
    "rms7_wav7": "RMS-7 + WaveletEnergy-7",
    "rms7_ptp7_wav7": "RMS-7 + PTP-7 + WaveletEnergy-7",
}

METHOD_DISPLAY = {
    "Baseline": "TimerXL (Head-only)",
    "TMA": "TimerXL + TMA",
    "RetrievalBackbone": "TimerXL + TMA (Retrieval Backbone)",
    "KNN_knn_only": "TimerXL + KNN (knn-only)",
    "KNN_blend": "TimerXL + KNN (blend)",
    "TMA+KNN_knn_only": "TimerXL + TMA + KNN (knn-only)",
    "TMA+KNN_blend": "TimerXL + TMA + KNN (blend)",
    "Baseline+KNN_knn_only_q0": "TimerXL + KNN (knn-only, late_q=0.0)",
    "Baseline+KNN_blend_q0": "TimerXL + KNN (blend, late_q=0.0)",
    "TMA+KNN_knn_only_q0": "TimerXL + TMA + KNN (knn-only, late_q=0.0)",
    "TMA+KNN_blend_q0": "TimerXL + TMA + KNN (blend, late_q=0.0)",
}

FEATURE_ORDER = {
    "pure_rms7": 1,
    "rms7_ptp7": 2,
    "rms7_wav7": 3,
    "rms7_ptp7_wav7": 4,
    "full133": 5,
}

METHOD_ORDER = {
    "Baseline": 1,
    "TMA": 2,
    "RetrievalBackbone": 3,
    "KNN_knn_only": 4,
    "Baseline+KNN_knn_only_q0": 4,
    "KNN_blend": 5,
    "Baseline+KNN_blend_q0": 5,
    "TMA+KNN_knn_only": 6,
    "TMA+KNN_knn_only_q0": 6,
    "TMA+KNN_blend": 7,
    "TMA+KNN_blend_q0": 7,
}


def _clean_note(values: Iterable[object]) -> str:
    notes: List[str] = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text not in notes:
            notes.append(text)
    return "; ".join(notes)


def build_feature_label(df: pd.DataFrame) -> pd.Series:
    """Create a feature label.

    The user's requested row key is (setting, feature_set, dim). In the current
    source CSV there are cases such as pure_rms7 + Baseline/TMA that appear at
    both epoch 200 and epoch 300. To avoid collapsing different epochs into one
    row, append epoch only for duplicated (setting, feature_set, dim) groups.
    """

    epoch_counts = (
        df.groupby(["setting", "feature_set", "dim"])["epoch"]
        .transform("nunique")
        .fillna(1)
    )

    labels = []
    for _, row in df.iterrows():
        dim = int(row["dim"])
        epoch = str(row["epoch"]).strip()
        feature_name = FEATURE_DISPLAY.get(str(row["feature_set"]), str(row["feature_set"]))
        base = f"{feature_name} ({dim}D"
        if epoch_counts.loc[row.name] > 1:
            labels.append(f"{base}, epoch={epoch})")
        else:
            labels.append(f"{base})")
    return pd.Series(labels, index=df.index, name="feature_label")


def convert_long_to_wide(input_csv: Path, loss_name: str = "MSE(wear) + λ_m L_mono + λ_s L_smooth") -> pd.DataFrame:
    df = pd.read_csv(input_csv)

    required = {"feature_set", "dim", "epoch", "setting", "fold", "rmse", "mae", "notes"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df[df["fold"].isin(FOLD_TO_DOMAIN)].copy()
    df["feature_label"] = build_feature_label(df)

    group_keys = ["setting", "feature_label"]
    rows: List[Dict[Tuple[str, str], object]] = []

    for (setting, feature_label), group in df.groupby(group_keys, sort=False):
        row: Dict[Tuple[str, str], object] = {
            ("Method (模型方法)", ""): METHOD_DISPLAY.get(str(setting), str(setting)),
            ("特征", ""): feature_label,
            ("损失函数", ""): loss_name,
            ("Note (备注)", ""): FEATURE_NOTES.get(
                str(group["feature_set"].iloc[0]),
                _clean_note(group["notes"]),
            ),
            ("__sort__", "epoch"): int(str(group["epoch"].iloc[0])),
            ("__sort__", "feature"): FEATURE_ORDER.get(str(group["feature_set"].iloc[0]), 999),
            ("__sort__", "method"): METHOD_ORDER.get(str(setting), 999),
        }

        for _, r in group.iterrows():
            domain = FOLD_TO_DOMAIN[r["fold"]]
            row[(domain, "MAE")] = r["mae"]
            row[(domain, "RMSE")] = r["rmse"]

        rows.append(row)

    columns = pd.MultiIndex.from_tuples(
        [
            ("Method (模型方法)", ""),
            ("特征", ""),
            ("损失函数", ""),
            ("C1", "MAE"),
            ("C1", "RMSE"),
            ("C4", "MAE"),
            ("C4", "RMSE"),
            ("C6", "MAE"),
            ("C6", "RMSE"),
            ("Note (备注)", ""),
        ]
    )

    wide_df = pd.DataFrame(rows)
    wide_df = wide_df.reindex(columns=columns)
    sort_epoch = pd.DataFrame(rows)[("__sort__", "epoch")]
    sort_feature = pd.DataFrame(rows)[("__sort__", "feature")]
    sort_method = pd.DataFrame(rows)[("__sort__", "method")]
    wide_df = wide_df.assign(
        __sort_epoch=sort_epoch.values,
        __sort_feature=sort_feature.values,
        __sort_method=sort_method.values,
    ).sort_values(
        ["__sort_epoch", "__sort_feature", "__sort_method", ("Method (模型方法)", "")]
    ).drop(columns=["__sort_epoch", "__sort_feature", "__sort_method"])
    return wide_df


def highlight_best_markdown(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with best values wrapped by markdown bold markers.

    Since CSV does not support real font styling, this makes the minima easy to
    spot after import. Best is defined as the minimum value within each metric
    column: C1/C4/C6 x MAE/RMSE.
    """

    out = wide_df.copy()
    metric_cols = [
        ("C1", "MAE"),
        ("C1", "RMSE"),
        ("C4", "MAE"),
        ("C4", "RMSE"),
        ("C6", "MAE"),
        ("C6", "RMSE"),
    ]

    for col in metric_cols:
        numeric = pd.to_numeric(wide_df[col], errors="coerce")
        if numeric.notna().any():
            best = numeric.min()
            formatted = []
            for value in wide_df[col]:
                num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                if pd.isna(num):
                    formatted.append(value)
                elif abs(num - best) < 1e-12:
                    formatted.append(f"**{num:.4f}**")
                else:
                    formatted.append(f"{num:.4f}")
            out[col] = formatted
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_csv",
        default="/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/新协议特征结果总对照.csv",
        help="Path to the source long-format CSV.",
    )
    parser.add_argument(
        "--output_csv",
        default="/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/新协议特征结果总对照_宽表.csv",
        help="Path to the output wide-format CSV.",
    )
    parser.add_argument(
        "--highlight_output_csv",
        default="/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM/paper_exec/新协议特征结果总对照_宽表_最优高亮.csv",
        help="Optional CSV path for markdown-highlighted best values.",
    )
    parser.add_argument(
        "--loss_name",
        default="MSE(wear) + λ_m L_mono + λ_s L_smooth",
        help='Value written to the "损失函数" column.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)

    wide_df = convert_long_to_wide(input_csv=input_csv, loss_name=args.loss_name)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    wide_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    highlight_df = highlight_best_markdown(wide_df)
    Path(args.highlight_output_csv).parent.mkdir(parents=True, exist_ok=True)
    highlight_df.to_csv(args.highlight_output_csv, index=False, encoding="utf-8-sig")

    print(f"Saved wide-format table to: {output_csv}")
    print(f"Saved highlighted table to: {args.highlight_output_csv}")
    print(f"Rows: {len(wide_df)}")


if __name__ == "__main__":
    main()
