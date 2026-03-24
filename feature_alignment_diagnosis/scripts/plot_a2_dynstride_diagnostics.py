#!/usr/bin/env python3
"""
Plot diagnostics for revised A2 dynstride.

This script does not visualize a global "alignment" because the revised A2
does not warp sequences onto a new axis. Instead, it visualizes:
1. Raw and monotonic wear curves.
2. Seq-len local wear slope curves.
3. Fold-specific slope thresholds and stride eligibility.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEQ_LEN = 96
STRIDE_CANDIDATES = [1, 2, 3]
STRIDE_QUANTILES = {2: 0.5, 3: 0.25}
FOLDS = {
    "fold1": {"train_runs": ["c1", "c4"], "test_run": "c6"},
    "fold2": {"train_runs": ["c4", "c6"], "test_run": "c1"},
    "fold3": {"train_runs": ["c1", "c6"], "test_run": "c4"},
}


@dataclass
class FoldSummary:
    fold: str
    train_runs: str
    test_run: str
    stride: int
    threshold: float
    eligible_windows: int
    eligible_ratio: float


def read_wear_max(csv_path: Path) -> np.ndarray:
    vals = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        wear_cols = [c for c in cols if c != "cut"]
        if not wear_cols:
            raise ValueError(f"{csv_path}: no wear columns")
        for row in rd:
            vals.append(max(float(row[c]) for c in wear_cols))
    return np.asarray(vals, dtype=np.float64)


def moving_hist_slope(y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.maximum.accumulate(y.astype(np.float64))
    T = len(y)
    n = T - seq_len - 16 + 1
    slopes = []
    starts = []
    for s in range(max(n, 0)):
        last_hist = s + seq_len - 1
        slope = (y[last_hist] - y[s]) / max(seq_len - 1, 1)
        starts.append(s)
        slopes.append(slope)
    return np.asarray(starts, dtype=np.int32), np.asarray(slopes, dtype=np.float64)


def build_fold_thresholds(wear_by_run: dict[str, np.ndarray], train_runs: list[str]) -> tuple[dict[int, float], dict[str, dict[int, float]]]:
    slope_values = []
    run_slopes = {}
    for run in train_runs:
        starts, slopes = moving_hist_slope(wear_by_run[run], SEQ_LEN)
        run_slopes[run] = {int(s): float(v) for s, v in zip(starts, slopes)}
        slope_values.extend(slopes.tolist())

    slope_values = np.asarray(slope_values, dtype=np.float64)
    thresholds = {}
    for stride in STRIDE_CANDIDATES:
        if stride == 1:
            continue
        thresholds[stride] = float(np.quantile(slope_values, STRIDE_QUANTILES[stride]))
    return thresholds, run_slopes


def window_fits(T: int, s_begin: int, stride: int) -> bool:
    last_x = s_begin + (SEQ_LEN - 1) * stride
    return (last_x + 16) < T


def eligible_stride_mask(T: int, slope_map: dict[int, float], thresholds: dict[int, float]) -> dict[int, np.ndarray]:
    starts = np.arange(max(T - SEQ_LEN - 16 + 1, 0))
    out = {}
    for stride, thr in thresholds.items():
        mask = []
        for s in starts:
            slope = slope_map.get(int(s), None)
            ok = slope is not None and slope <= thr and window_fits(T, int(s), stride)
            mask.append(ok)
        out[stride] = np.asarray(mask, dtype=bool)
    return out


def plot_global_wear(wear_by_run: dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for run in ["c1", "c4", "c6"]:
        y = wear_by_run[run]
        plt.plot(np.arange(1, len(y) + 1), y, label=f"{run} raw", linewidth=2)
        plt.plot(np.arange(1, len(y) + 1), np.maximum.accumulate(y), linestyle="--", alpha=0.9, label=f"{run} mono")
    plt.xlabel("Cut Index")
    plt.ylabel("Wear (um)")
    plt.title("A2 dynstride: Raw vs Monotonic Wear Curves")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "a2_dynstride_wear_curves_raw_mono.png", dpi=180)
    plt.close()


def plot_global_progress(wear_by_run: dict[str, np.ndarray], out_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    for run in ["c1", "c4", "c6"]:
        y = np.maximum.accumulate(wear_by_run[run])
        prog = (y - y[0]) / max(y[-1] - y[0], 1e-12)
        plt.plot(np.arange(1, len(y) + 1), prog, label=run, linewidth=2)
    plt.xlabel("Cut Index")
    plt.ylabel("Wear Progress [0,1]")
    plt.title("A2 dynstride: Wear Progress Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "a2_dynstride_wear_progress_curves.png", dpi=180)
    plt.close()


def plot_fold_diagnostics(
    fold: str,
    wear_by_run: dict[str, np.ndarray],
    train_runs: list[str],
    test_run: str,
    thresholds: dict[int, float],
    run_slopes: dict[str, dict[int, float]],
    out_dir: Path,
) -> list[FoldSummary]:
    fold_dir = out_dir / fold
    fold_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[FoldSummary] = []

    plt.figure(figsize=(10, 5))
    for run in train_runs + [test_run]:
        starts, slopes = moving_hist_slope(wear_by_run[run], SEQ_LEN)
        plt.plot(starts + 1, slopes, label=run, linewidth=2)
    for stride, thr in thresholds.items():
        plt.axhline(thr, linestyle="--", linewidth=1.5, label=f"stride={stride} thr={thr:.4f}")
    plt.xlabel("Window Start")
    plt.ylabel(f"Local Slope over {SEQ_LEN} cuts (um/cut)")
    plt.title(f"{fold}: Local Wear Slope and Dynstride Thresholds")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fold_dir / f"{fold}_local_slope_thresholds.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(len(train_runs), 1, figsize=(10, 3.5 * len(train_runs)), sharex=True)
    if len(train_runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, train_runs):
        y = np.maximum.accumulate(wear_by_run[run])
        starts = np.arange(max(len(y) - SEQ_LEN - 16 + 1, 0))
        ax.plot(np.arange(1, len(y) + 1), y, color="black", linewidth=2, label=f"{run} mono wear")
        slope_map = run_slopes[run]
        masks = eligible_stride_mask(len(y), slope_map, thresholds)
        for stride, color in [(2, "#2b8cbe"), (3, "#e34a33")]:
            if stride not in masks:
                continue
            mask = masks[stride]
            elig_starts = starts[mask]
            if len(elig_starts) == 0:
                continue
            ax.scatter(elig_starts + 1, y[elig_starts], s=12, alpha=0.8, color=color, label=f"eligible stride={stride}")
            summaries.append(
                FoldSummary(
                    fold=fold,
                    train_runs=",".join(train_runs),
                    test_run=test_run,
                    stride=stride,
                    threshold=float(thresholds[stride]),
                    eligible_windows=int(mask.sum()),
                    eligible_ratio=float(mask.mean()) if len(mask) else 0.0,
                )
            )
        ax.set_ylabel("Wear (um)")
        ax.set_title(f"{fold}: {run} eligible train windows")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Window Start / Cut Index")
    plt.tight_layout()
    plt.savefig(fold_dir / f"{fold}_eligible_windows_on_wear_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    all_train_slopes = []
    for run in train_runs:
        all_train_slopes.extend(run_slopes[run].values())
    all_train_slopes = np.asarray(all_train_slopes, dtype=np.float64)
    plt.hist(all_train_slopes, bins=40, color="#9ecae1", edgecolor="white")
    for stride, thr in thresholds.items():
        plt.axvline(thr, linestyle="--", linewidth=2, label=f"stride={stride} thr={thr:.4f}")
    plt.xlabel("Local Slope (um/cut)")
    plt.ylabel("Count")
    plt.title(f"{fold}: Train Slope Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fold_dir / f"{fold}_train_slope_hist.png", dpi=180)
    plt.close()

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("feature_alignment_diagnosis/outputs"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (root / args.output_dir / f"A2_dynstride_diagnostics_{ts}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    wear_by_run = {
        run: read_wear_max(root / "dataset" / run / f"{run}_wear.csv")
        for run in ["c1", "c4", "c6"]
    }

    plot_global_wear(wear_by_run, out_dir)
    plot_global_progress(wear_by_run, out_dir)

    summary_rows: list[FoldSummary] = []
    manifest = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "seq_len": SEQ_LEN,
        "stride_candidates": STRIDE_CANDIDATES,
        "stride_quantiles": STRIDE_QUANTILES,
        "feature_set": ["Feat_3", "Feat_7", "Feat_11", "Feat_15", "Feat_19", "Feat_23", "Feat_27", "Feat_4"],
        "note": "Revised A2 dynstride stays in time domain; figures diagnose local slope and eligible train windows instead of global alignment.",
        "folds": {},
    }

    for fold, cfg in FOLDS.items():
        thresholds, run_slopes = build_fold_thresholds(wear_by_run, cfg["train_runs"])
        summary_rows.extend(
            plot_fold_diagnostics(
                fold=fold,
                wear_by_run=wear_by_run,
                train_runs=cfg["train_runs"],
                test_run=cfg["test_run"],
                thresholds=thresholds,
                run_slopes=run_slopes,
                out_dir=out_dir,
            )
        )
        manifest["folds"][fold] = {
            "train_runs": cfg["train_runs"],
            "test_run": cfg["test_run"],
            "thresholds": {str(k): float(v) for k, v in thresholds.items()},
        }

    with (out_dir / "a2_dynstride_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    with (out_dir / "a2_dynstride_summary.csv").open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["fold", "train_runs", "test_run", "stride", "threshold", "eligible_windows", "eligible_ratio"])
        for row in summary_rows:
            wr.writerow(
                [
                    row.fold,
                    row.train_runs,
                    row.test_run,
                    row.stride,
                    f"{row.threshold:.6f}",
                    row.eligible_windows,
                    f"{row.eligible_ratio:.6f}",
                ]
            )

    print(f"[A2-DYNSTRIDE][OK] outputs: {out_dir}")


if __name__ == "__main__":
    main()
