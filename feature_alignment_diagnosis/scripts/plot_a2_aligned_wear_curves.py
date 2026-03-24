#!/usr/bin/env python3
"""
Plot A2 stage-aligned wear curves (raw vs aligned) for fold1/2/3.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
    if not vals:
        raise ValueError(f"{csv_path}: empty file")
    return np.asarray(vals, dtype=np.float64)


def normalize_progress(y: np.ndarray) -> np.ndarray:
    y_mono = np.maximum.accumulate(y)
    y0 = float(y_mono[0])
    yr = float(y_mono[-1] - y_mono[0])
    if yr <= 1e-12:
        return np.zeros_like(y_mono)
    return (y_mono - y0) / yr


def plot_one_fold(
    fold_name: str,
    fold_dir: Path,
    raw_wear_by_run: dict[str, np.ndarray],
    out_dir: Path,
) -> dict:
    manifest_path = fold_dir / "a2_stagealign_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    aligned_info = {r["run"]: bool(r["aligned"]) for r in manifest.get("records", [])}
    run_order = manifest.get("runs", ["c1", "c4", "c6"])

    aligned_wear_dir = fold_dir / "wear_csv"
    aligned_wear_by_run = {
        run: read_wear_max(aligned_wear_dir / f"{run}_wear.csv") for run in run_order
    }

    # Plot raw wear values.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=False)
    for ax, run in zip(axes, run_order):
        y_raw = raw_wear_by_run[run]
        y_aln = aligned_wear_by_run[run]
        x = np.arange(1, len(y_raw) + 1)
        ax.plot(x, y_raw, label=f"{run} raw", linewidth=1.8, color="#1f77b4")
        if aligned_info.get(run, False):
            ax.plot(x, y_aln, label=f"{run} aligned", linewidth=1.6, linestyle="--", color="#d62728")
        else:
            ax.plot(x, y_aln, label=f"{run} unchanged(test)", linewidth=1.4, linestyle="--", color="#2ca02c")
        ax.set_title(f"{run} ({'aligned' if aligned_info.get(run, False) else 'unchanged'})")
        ax.set_xlabel("cut index")
        ax.set_ylabel("wear (um)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"A2 {fold_name}: wear curves (raw vs aligned)")
    fig.tight_layout()
    raw_png = out_dir / f"{fold_name}_a2_wear_raw_vs_aligned.png"
    fig.savefig(raw_png, bbox_inches="tight")
    plt.close(fig)

    # Plot normalized progress values.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=True)
    for ax, run in zip(axes, run_order):
        y_raw = normalize_progress(raw_wear_by_run[run])
        y_aln = normalize_progress(aligned_wear_by_run[run])
        x = np.arange(1, len(y_raw) + 1)
        ax.plot(x, y_raw, label=f"{run} raw", linewidth=1.8, color="#1f77b4")
        if aligned_info.get(run, False):
            ax.plot(x, y_aln, label=f"{run} aligned", linewidth=1.6, linestyle="--", color="#d62728")
        else:
            ax.plot(x, y_aln, label=f"{run} unchanged(test)", linewidth=1.4, linestyle="--", color="#2ca02c")
        ax.set_title(f"{run} progress")
        ax.set_xlabel("cut index")
        ax.set_ylabel("normalized progress [0,1]")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"A2 {fold_name}: normalized wear progress (raw vs aligned)")
    fig.tight_layout()
    norm_png = out_dir / f"{fold_name}_a2_wear_progress_vs_aligned.png"
    fig.savefig(norm_png, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "fold": fold_name,
        "fold_dir": str(fold_dir),
        "raw_plot": str(raw_png),
        "progress_plot": str(norm_png),
        "aligned_runs": [r for r, ok in aligned_info.items() if ok],
        "unchanged_runs": [r for r, ok in aligned_info.items() if not ok],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Optional output dir. Default: feature_alignment_diagnosis/outputs/A2_aligned_wear_curves_<time>",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "feature_alignment_diagnosis" / "outputs" / f"A2_aligned_wear_curves_{now}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_wear_by_run = {
        run: read_wear_max(root / "dataset" / run / f"{run}_wear.csv") for run in ["c1", "c4", "c6"]
    }
    fold_dirs = {
        "fold1": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold1",
        "fold2": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold2",
        "fold3": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold3",
    }

    all_summary = []
    for fold_name, fold_dir in fold_dirs.items():
        all_summary.append(plot_one_fold(fold_name, fold_dir, raw_wear_by_run, out_dir))

    summary_path = out_dir / "a2_aligned_curve_summary.json"
    summary_path.write_text(json.dumps(all_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] output_dir: {out_dir}")
    print(f"[OK] summary: {summary_path}")
    for row in all_summary:
        print(f"[OK] {row['fold']} raw: {row['raw_plot']}")
        print(f"[OK] {row['fold']} progress: {row['progress_plot']}")


if __name__ == "__main__":
    main()

