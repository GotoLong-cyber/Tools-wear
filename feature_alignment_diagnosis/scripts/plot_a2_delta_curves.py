#!/usr/bin/env python3
"""
Plot A2 aligned-vs-raw delta curves for fold1/2/3.
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
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "feature_alignment_diagnosis" / "outputs" / f"A2_aligned_wear_curves_{now}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = {r: read_wear_max(root / "dataset" / r / f"{r}_wear.csv") for r in ["c1", "c4", "c6"]}
    fold_dirs = {
        "fold1": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold1",
        "fold2": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold2",
        "fold3": root / "dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold3",
    }

    summary_rows = []

    for fold, fold_dir in fold_dirs.items():
        manifest = json.loads((fold_dir / "a2_stagealign_manifest.json").read_text(encoding="utf-8"))
        aligned_map = {r["run"]: bool(r["aligned"]) for r in manifest.get("records", [])}
        runs = manifest.get("runs", ["c1", "c4", "c6"])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), dpi=150, sharey=True)
        for ax, run in zip(axes, runs):
            y_raw = raw[run]
            y_aln = read_wear_max(fold_dir / "wear_csv" / f"{run}_wear.csv")
            delta = y_aln - y_raw
            x = np.arange(1, len(delta) + 1)

            if aligned_map.get(run, False):
                color = "#d62728"
                label = f"{run} delta(aligned-raw)"
            else:
                color = "#2ca02c"
                label = f"{run} delta(unchanged)"

            ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
            ax.plot(x, delta, color=color, linewidth=1.5, label=label)
            ax.set_title(f"{run} ({'aligned' if aligned_map.get(run, False) else 'unchanged'})")
            ax.set_xlabel("cut index")
            ax.set_ylabel("delta wear (um)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)

            summary_rows.append(
                {
                    "fold": fold,
                    "run": run,
                    "aligned": aligned_map.get(run, False),
                    "delta_mean_um": float(np.mean(delta)),
                    "delta_mae_um": float(np.mean(np.abs(delta))),
                    "delta_p95_um": float(np.percentile(np.abs(delta), 95)),
                    "delta_max_abs_um": float(np.max(np.abs(delta))),
                }
            )

        fig.suptitle(f"A2 {fold}: delta curve (aligned - raw)")
        fig.tight_layout()
        fig.savefig(output_dir / f"{fold}_a2_delta_curve.png", bbox_inches="tight")
        plt.close(fig)

    # Aggregate |delta| bar plot.
    labels = [f"{r['fold']}-{r['run']}" for r in summary_rows]
    vals = [r["delta_mae_um"] for r in summary_rows]
    colors = ["#d62728" if r["aligned"] else "#2ca02c" for r in summary_rows]

    plt.figure(figsize=(12, 4.6), dpi=150)
    x = np.arange(len(labels))
    plt.bar(x, vals, color=colors)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("|delta| mean (um)")
    plt.title("A2 delta magnitude by fold/run")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "a2_delta_mae_bar.png", bbox_inches="tight")
    plt.close()

    summary_csv = output_dir / "a2_delta_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold",
            "run",
            "aligned",
            "delta_mean_um",
            "delta_mae_um",
            "delta_p95_um",
            "delta_max_abs_um",
        ]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(summary_rows)

    print(f"[OK] output_dir: {output_dir}")
    print(f"[OK] summary_csv: {summary_csv}")
    for fold in fold_dirs:
        print(f"[OK] {output_dir / f'{fold}_a2_delta_curve.png'}")
    print(f"[OK] {output_dir / 'a2_delta_mae_bar.png'}")


if __name__ == "__main__":
    main()

