#!/usr/bin/env python3
"""Build a clean epoch-grouped core comparison table from the long result CSV."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/jyc23/zml25/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
INPUT_CSV = ROOT / "paper_exec" / "新协议特征结果总对照.csv"
OUTPUT_CSV = ROOT / "paper_exec" / "论文级核心对照表_分epoch.csv"

LOSS_TEXT = "MSE(wear) + λ_m L_mono + λ_s L_smooth"

FEATURE_ORDER = [
    "pure_rms7",
    "rms7_ptp7",
    "rms7_wav7",
    "rms7_ptp7_wav7",
    "full133",
]

FEATURE_LABELS = {
    "pure_rms7": "RMS-7 (7D)",
    "rms7_ptp7": "RMS-7 + PTP-7 (14D)",
    "rms7_wav7": "RMS-7 + WaveletEnergy-7 (14D)",
    "rms7_ptp7_wav7": "RMS-7 + PTP-7 + WaveletEnergy-7 (21D)",
    "full133": "Full-133 (133D)",
}

FEATURE_NOTES = {
    "pure_rms7": "7通道RMS",
    "rms7_ptp7": "7通道RMS + 7通道PTP",
    "rms7_wav7": "7通道RMS + 7通道小波能量",
    "rms7_ptp7_wav7": "7通道RMS + 7通道PTP + 7通道小波能量",
    "full133": "7通道×19统计特征（时域+频域）",
}

METHOD_ORDER = ["Baseline", "TMA", "KNN", "blend", "KNN+TMA", "KNN+TMA+blend"]

METHOD_LABELS = {
    "Baseline": "Head-only",
    "TMA": "TMA",
    "KNN": "KNN",
    "blend": "blend",
    "KNN+TMA": "KNN+TMA",
    "KNN+TMA+blend": "KNN+TMA+blend",
}

FOLD_TO_CONDITION = {
    "fold2": "C1",
    "fold3": "C4",
    "fold1": "C6",
}


def load_and_repair_rows() -> list[dict]:
    rows = list(csv.DictReader(INPUT_CSV.open("r", encoding="utf-8-sig")))
    repaired = []
    for r in rows:
        # Repair malformed 21D baseline rows accidentally appended in wrong columns.
        if (
            r["feature_set"] == "rms7_ptp7_wav7"
            and r["epoch"] == "Baseline"
            and r["setting"] in {"fold2", "fold3", "avg"}
        ):
            repaired.append({
                "feature_set": "rms7_ptp7_wav7",
                "dim": r["dim"],
                "epoch": "200",
                "setting": "Baseline",
                "fold": r["setting"],
                "rmse": r["fold"],
                "mae": r["rmse"],
                "notes": r["mae"],
            })
        else:
            repaired.append(r)
    return repaired


def normalize_method_key(setting: str) -> str | None:
    mapping = {
        "Baseline": "Baseline",
        "TMA": "TMA",
        "KNN_knn_only": "KNN",
        "KNN_blend": "blend",
        "Baseline+KNN_knn_only_q0": "KNN",
        "Baseline+KNN_blend_q0": "blend",
        "TMA+KNN_knn_only": "KNN+TMA",
        "TMA+KNN_blend": "KNN+TMA+blend",
        "TMA+KNN_knn_only_q0": "KNN+TMA",
        "TMA+KNN_blend_q0": "KNN+TMA+blend",
    }
    return mapping.get(setting)


def priority(setting: str) -> int:
    if setting.endswith("_q0"):
        return 2
    return 1


def build_table(rows: list[dict]) -> list[list[str]]:
    valid_rows = []
    for r in rows:
        if r["epoch"] not in {"200", "300"}:
            continue
        if r["feature_set"] not in FEATURE_ORDER:
            continue
        method_key = normalize_method_key(r["setting"])
        if method_key is None:
            continue
        valid_rows.append({
            **r,
            "method_key": method_key,
        })

    grouped: dict[tuple[str, str, str], dict[str, tuple[str, str]]] = {}
    chosen_priority: dict[tuple[str, str, str], int] = {}
    for r in valid_rows:
        key = (r["epoch"], r["feature_set"], r["method_key"])
        p = priority(r["setting"])
        if key in chosen_priority and p < chosen_priority[key]:
            continue
        if key not in chosen_priority or p > chosen_priority[key]:
            grouped[key] = {}
            chosen_priority[key] = p
        bucket = grouped.setdefault(key, {})
        if r["fold"] in FOLD_TO_CONDITION:
            bucket[FOLD_TO_CONDITION[r["fold"]]] = (r["mae"], r["rmse"])

    rows_out: list[list[str]] = [
        ["Epoch", "Method (模型方法)", "特征", "损失函数", "C1", "C1", "C4", "C4", "C6", "C6", "Avg", "Avg", "Note (备注)"],
        ["", "", "", "", "MAE", "RMSE", "MAE", "RMSE", "MAE", "RMSE", "MAE", "RMSE", ""],
    ]

    for epoch in ["200", "300"]:
        first_epoch_row = True
        for feature in FEATURE_ORDER:
            for method_key in METHOD_ORDER:
                key = (epoch, feature, method_key)
                if key not in grouped:
                    continue
                vals = grouped[key]
                c1 = vals.get("C1", ("", ""))
                c4 = vals.get("C4", ("", ""))
                c6 = vals.get("C6", ("", ""))
                mae_vals = [v for v in [c1[0], c4[0], c6[0]] if v != ""]
                rmse_vals = [v for v in [c1[1], c4[1], c6[1]] if v != ""]
                avg_mae = f"{sum(float(v) for v in mae_vals)/len(mae_vals):.4f}" if mae_vals else ""
                avg_rmse = f"{sum(float(v) for v in rmse_vals)/len(rmse_vals):.4f}" if rmse_vals else ""
                rows_out.append([
                    epoch if first_epoch_row else "",
                    METHOD_LABELS[method_key],
                    FEATURE_LABELS[feature] + (f", epoch={epoch}" if feature == "pure_rms7" else ""),
                    LOSS_TEXT,
                    c1[0],
                    c1[1],
                    c4[0],
                    c4[1],
                    c6[0],
                    c6[1],
                    avg_mae,
                    avg_rmse,
                    FEATURE_NOTES[feature],
                ])
                first_epoch_row = False
        rows_out.append([""] * 13)

    # drop trailing blank
    if rows_out and all(v == "" for v in rows_out[-1]):
        rows_out.pop()
    return rows_out


def main() -> None:
    rows = load_and_repair_rows()
    out_rows = build_table(rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)
    print(OUTPUT_CSV)


if __name__ == "__main__":
    main()
