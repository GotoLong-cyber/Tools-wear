#!/usr/bin/env python3
"""A1: time-scale evidence analysis for c1/c4/c6 wear curves.

This script does not train models. It only reads wear CSV files and outputs
curve plots + quantitative tables for time-scale mismatch diagnosis.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class RunWear:
    run: str
    cut: np.ndarray
    wear: np.ndarray
    wear_monotonic: np.ndarray


def resolve_wear_csv(dataset_dir: Path, run: str) -> Path:
    candidates = [
        dataset_dir / run / f"{run}_wear.csv",
        dataset_dir / "passlevel_full133_npz" / f"{run}_wear.csv",
        dataset_dir / f"{run}_wear.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"Cannot find wear csv for run={run}. Tried: {candidates}")


def read_wear_csv(path: Path, run: str, wear_agg: str) -> RunWear:
    cuts: List[float] = []
    mat: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        wear_cols = [c for c in fields if c != "cut"]
        if not wear_cols:
            raise ValueError(f"{path}: no wear columns")

        for i, row in enumerate(reader):
            cut_val = float(row["cut"]) if "cut" in row and row["cut"] != "" else float(i + 1)
            vals = [float(row[c]) for c in wear_cols]
            cuts.append(cut_val)
            mat.append(vals)

    cut = np.asarray(cuts, dtype=np.float64)
    arr = np.asarray(mat, dtype=np.float64)

    if wear_agg in wear_cols:
        idx = wear_cols.index(wear_agg)
        wear = arr[:, idx]
    elif wear_agg == "mean":
        wear = arr.mean(axis=1)
    else:
        wear = arr.max(axis=1)

    wear_mono = np.maximum.accumulate(wear)
    return RunWear(run=run, cut=cut, wear=wear, wear_monotonic=wear_mono)


def first_crossing_cut(cut: np.ndarray, wear_mono: np.ndarray, threshold: float) -> float:
    idx = np.where(wear_mono >= threshold)[0]
    if idx.size == 0:
        return float("nan")
    return float(cut[idx[0]])


def segment_slope(cut: np.ndarray, wear: np.ndarray, lo: float, hi: float) -> float:
    n = len(cut)
    i0 = int(np.floor(lo * n))
    i1 = int(np.ceil(hi * n))
    i0 = max(0, min(i0, n - 2))
    i1 = max(i0 + 2, min(i1, n))
    x = cut[i0:i1]
    y = wear[i0:i1]
    if len(x) < 2:
        return float("nan")
    p = np.polyfit(x, y, 1)
    return float(p[0])


def plot_curves(packs: Dict[str, RunWear], out: Path) -> None:
    plt.figure(figsize=(9, 5))
    for run, p in packs.items():
        plt.plot(p.cut, p.wear, linewidth=1.8, label=f"{run} raw")
        plt.plot(p.cut, p.wear_monotonic, linewidth=1.2, linestyle="--", alpha=0.85, label=f"{run} mono")
    plt.xlabel("cut index")
    plt.ylabel("wear (um)")
    plt.title("A1 Time-Scale Check: wear curves (raw + monotonic envelope)")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_normalized(packs: Dict[str, RunWear], out: Path) -> None:
    plt.figure(figsize=(9, 5))
    for run, p in packs.items():
        y = p.wear_monotonic
        y_norm = (y - y.min()) / (max(1e-12, y.max() - y.min()))
        plt.plot(p.cut, y_norm, linewidth=2.0, label=run)
    plt.xlabel("cut index")
    plt.ylabel("normalized wear progress [0,1]")
    plt.title("A1 Time-Scale Check: normalized wear progression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_threshold_bars(cross_table: Dict[str, Dict[str, float]], out: Path, title: str) -> None:
    runs = list(cross_table.keys())
    thr_keys = list(next(iter(cross_table.values())).keys())

    x = np.arange(len(thr_keys), dtype=np.float64)
    width = 0.22

    plt.figure(figsize=(10, 5.5))
    for i, run in enumerate(runs):
        vals = [cross_table[run][k] for k in thr_keys]
        plt.bar(x + (i - 1) * width, vals, width=width, label=run)

    plt.xticks(x, thr_keys, rotation=0)
    plt.ylabel("first crossing cut")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def plot_stage_slopes(metrics: List[dict], out: Path) -> None:
    runs = [m["run"] for m in metrics]
    early = [m["slope_early"] for m in metrics]
    mid = [m["slope_mid"] for m in metrics]
    late = [m["slope_late"] for m in metrics]

    x = np.arange(len(runs))
    w = 0.24

    plt.figure(figsize=(9, 5))
    plt.bar(x - w, early, width=w, label="early(0-33%)")
    plt.bar(x, mid, width=w, label="mid(33-66%)")
    plt.bar(x + w, late, width=w, label="late(66-100%)")
    plt.xticks(x, runs)
    plt.ylabel("slope (um per cut)")
    plt.title("A1 Time-Scale Check: stage slopes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()


def write_csv(path: Path, rows: List[dict], headers: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("dataset"))
    parser.add_argument("--runs", nargs="+", default=["c1", "c4", "c6"])
    parser.add_argument("--train_runs", nargs="+", default=["c1", "c4"])
    parser.add_argument("--wear_agg", type=str, default="max", help="max|mean|column name")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    now = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = args.out_dir or Path("feature_alignment_diagnosis") / "outputs" / f"A1_time_scale_{now}"
    out_dir.mkdir(parents=True, exist_ok=True)

    packs: Dict[str, RunWear] = {}
    src_files: Dict[str, str] = {}
    for run in args.runs:
        p = resolve_wear_csv(args.dataset_dir, run)
        src_files[run] = str(p)
        packs[run] = read_wear_csv(p, run=run, wear_agg=args.wear_agg)

    # thresholds from training runs only (for reproducibility and anti-leakage discipline)
    train_vec = np.concatenate([packs[r].wear for r in args.train_runs])
    q_vals = np.quantile(train_vec, [0.25, 0.5, 0.75, 0.9])
    q_names = ["train_q25", "train_q50", "train_q75", "train_q90"]
    train_thresholds = {k: float(v) for k, v in zip(q_names, q_vals)}

    metrics_rows: List[dict] = []
    cross_train: Dict[str, Dict[str, float]] = {}
    cross_self: Dict[str, Dict[str, float]] = {}

    for run in args.runs:
        p = packs[run]
        n = len(p.cut)
        c_first = float(p.cut[0])
        c_last = float(p.cut[-1])

        self_min = float(p.wear.min())
        self_max = float(p.wear.max())
        self_range = max(1e-12, self_max - self_min)

        # crossings on train thresholds
        train_crossings = {
            k: first_crossing_cut(p.cut, p.wear_monotonic, thr)
            for k, thr in train_thresholds.items()
        }
        cross_train[run] = train_crossings

        # crossings on self-relative progress
        self_thr = {
            "self_p70": self_min + 0.70 * self_range,
            "self_p80": self_min + 0.80 * self_range,
            "self_p90": self_min + 0.90 * self_range,
        }
        self_crossings = {
            k: first_crossing_cut(p.cut, p.wear_monotonic, thr)
            for k, thr in self_thr.items()
        }
        cross_self[run] = self_crossings

        row = {
            "run": run,
            "n_points": n,
            "cut_start": c_first,
            "cut_end": c_last,
            "wear_start": float(p.wear[0]),
            "wear_end": float(p.wear[-1]),
            "wear_delta": float(p.wear[-1] - p.wear[0]),
            "slope_global": float((p.wear[-1] - p.wear[0]) / max(1e-12, c_last - c_first)),
            "slope_early": segment_slope(p.cut, p.wear_monotonic, 0.0, 0.33),
            "slope_mid": segment_slope(p.cut, p.wear_monotonic, 0.33, 0.66),
            "slope_late": segment_slope(p.cut, p.wear_monotonic, 0.66, 1.0),
            "cut_train_q25": train_crossings["train_q25"],
            "cut_train_q50": train_crossings["train_q50"],
            "cut_train_q75": train_crossings["train_q75"],
            "cut_train_q90": train_crossings["train_q90"],
            "cut_self_p70": self_crossings["self_p70"],
            "cut_self_p80": self_crossings["self_p80"],
            "cut_self_p90": self_crossings["self_p90"],
        }
        metrics_rows.append(row)

    # sort rows by run order
    metrics_rows.sort(key=lambda x: args.runs.index(x["run"]))

    # write tables
    write_csv(
        out_dir / "a1_time_scale_metrics.csv",
        metrics_rows,
        headers=list(metrics_rows[0].keys()),
    )

    with (out_dir / "a1_train_thresholds.json").open("w", encoding="utf-8") as f:
        json.dump(train_thresholds, f, ensure_ascii=False, indent=2)

    # crossings table (wide)
    rows_cross_train = []
    for run in args.runs:
        r = {"run": run}
        r.update(cross_train[run])
        rows_cross_train.append(r)
    write_csv(
        out_dir / "a1_crossing_train_thresholds.csv",
        rows_cross_train,
        headers=["run"] + q_names,
    )

    rows_cross_self = []
    for run in args.runs:
        r = {"run": run}
        r.update(cross_self[run])
        rows_cross_self.append(r)
    write_csv(
        out_dir / "a1_crossing_self_thresholds.csv",
        rows_cross_self,
        headers=["run", "self_p70", "self_p80", "self_p90"],
    )

    # plots
    plot_curves(packs, out_dir / "a1_wear_curves_raw_and_mono.png")
    plot_normalized(packs, out_dir / "a1_wear_curves_normalized_progress.png")
    plot_threshold_bars(
        cross_train,
        out_dir / "a1_crossing_train_thresholds.png",
        title="A1: first crossing cut on train-derived wear thresholds",
    )
    plot_threshold_bars(
        cross_self,
        out_dir / "a1_crossing_self_progress.png",
        title="A1: first crossing cut on self progress thresholds",
    )
    plot_stage_slopes(metrics_rows, out_dir / "a1_stage_slopes.png")

    # short markdown summary
    # compare c6 vs c1/c4 on train q75 and q90 as key evidence
    row_map = {r["run"]: r for r in metrics_rows}
    q75 = {
        r: row_map[r]["cut_train_q75"] for r in args.runs if "cut_train_q75" in row_map[r]
    }
    q90 = {
        r: row_map[r]["cut_train_q90"] for r in args.runs if "cut_train_q90" in row_map[r]
    }

    def fmt(v: float) -> str:
        if np.isnan(v):
            return "NA"
        return f"{v:.0f}"

    md = []
    md.append(f"# A1 时间尺度证据化摘要（{datetime.now().strftime('%Y-%m-%d %H:%M')}）")
    md.append("")
    md.append("## 输入与口径")
    md.append(f"- runs: {args.runs}")
    md.append(f"- train_runs: {args.train_runs}")
    md.append(f"- wear聚合口径: {args.wear_agg}（与训练默认 `wear_agg=max` 一致）")
    md.append("")
    md.append("## 关键阈值（由训练域 c1/c4 拟合）")
    for k in q_names:
        md.append(f"- {k}: {train_thresholds[k]:.4f} um")
    md.append("")
    md.append("## 核心观察")
    md.append(
        f"- 在 train_q75 阈值上，c1/c4/c6 首次到达 cut 分别为：{fmt(q75.get('c1', np.nan))} / {fmt(q75.get('c4', np.nan))} / {fmt(q75.get('c6', np.nan))}。"
    )
    md.append(
        f"- 在 train_q90 阈值上，c1/c4/c6 首次到达 cut 分别为：{fmt(q90.get('c1', np.nan))} / {fmt(q90.get('c4', np.nan))} / {fmt(q90.get('c6', np.nan))}。"
    )
    md.append("- 若 c6 在高阈值处明显更早达到，说明存在时间尺度错位（更早进入高磨损阶段）。")
    md.append("- 该结论用于后续切片策略设计，不直接回流到当前折训练拟合。")
    md.append("")
    md.append("## 产物")
    md.append("- `a1_time_scale_metrics.csv`")
    md.append("- `a1_crossing_train_thresholds.csv`")
    md.append("- `a1_crossing_self_thresholds.csv`")
    md.append("- `a1_wear_curves_raw_and_mono.png`")
    md.append("- `a1_wear_curves_normalized_progress.png`")
    md.append("- `a1_crossing_train_thresholds.png`")
    md.append("- `a1_crossing_self_progress.png`")
    md.append("- `a1_stage_slopes.png`")

    (out_dir / "a1_summary.md").write_text("\n".join(md), encoding="utf-8")

    manifest = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dataset_dir": str(args.dataset_dir),
        "runs": args.runs,
        "train_runs": args.train_runs,
        "wear_agg": args.wear_agg,
        "source_files": src_files,
        "out_dir": str(out_dir),
    }
    with (out_dir / "a1_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[A1][OK] outputs -> {out_dir}")


if __name__ == "__main__":
    main()
