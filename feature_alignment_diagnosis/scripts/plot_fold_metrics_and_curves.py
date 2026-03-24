#!/usr/bin/env python3
import argparse
import csv
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

PATTERN = re.compile(
    r"\[Metric\]\[(window|fullcurve|fullcurve_raw)\] "
    r"mse\(um\^2\):([0-9eE+\-.]+), rmse\(um\):([0-9eE+\-.]+), mae\(um\):([0-9eE+\-.]+)"
)


def parse_metrics(log_path: Path):
    data = {}
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for metric_type, mse, rmse, mae in PATTERN.findall(text):
        data[metric_type] = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
        }
    return data


def plot_bars(folds, metric_map, out_file: Path):
    labels = [f["name"] for f in folds]
    x = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=150)
    for ax, metric_name in zip(axes, ["mae", "rmse", "mse"]):
        vals_window = [metric_map[f["name"]]["window"][metric_name] for f in folds]
        vals_fc = [metric_map[f["name"]]["fullcurve"][metric_name] for f in folds]
        vals_raw = [metric_map[f["name"]]["fullcurve_raw"][metric_name] for f in folds]

        ax.bar(x - width, vals_window, width=width, label="window")
        ax.bar(x, vals_fc, width=width, label="fullcurve")
        ax.bar(x + width, vals_raw, width=width, label="fullcurve_raw")

        ax.set_title(metric_name.upper())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[0].set_ylabel("metric value")
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper center", ncol=3)
    fig.suptitle("Cross-Fold Metrics Comparison (RMS7 + Feat_4)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def plot_curve_montage(folds, out_file: Path):
    fig, axes = plt.subplots(1, len(folds), figsize=(6 * len(folds), 5), dpi=150)
    if len(folds) == 1:
        axes = [axes]

    for ax, fold in zip(axes, folds):
        img = mpimg.imread(str(fold["curve_png"]))
        ax.imshow(img)
        ax.set_title(fold["name"])
        ax.axis("off")

    fig.suptitle("Wear Curve Visualization by Fold", y=0.98)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def write_csv(folds, metric_map, out_csv: Path):
    rows = []
    for fold in folds:
        name = fold["name"]
        for t in ["window", "fullcurve", "fullcurve_raw"]:
            m = metric_map[name][t]
            rows.append(
                {
                    "fold": name,
                    "metric_type": t,
                    "mae_um": m["mae"],
                    "rmse_um": m["rmse"],
                    "mse_um2": m["mse"],
                    "log_file": str(fold["log"]),
                }
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "metric_type", "mae_um", "rmse_um", "mse_um2", "log_file"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Generate visualization for 3-fold experiments.")
    parser.add_argument("--root", type=Path, required=True, help="OpenLTM root directory")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else root / "feature_alignment_diagnosis" / "outputs" / f"crossfold_feat4_visual_{now}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = [
        {
            "name": "c1c4->c6",
            "log": root / "results" / "longrun_PHM_c1c4_to_c6_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1.log",
            "curve_png": root
            / "results"
            / "forecast_PHM_c1c4_to_c6_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
            / "wear_full_curve_trueRaw_predWindows.png",
        },
        {
            "name": "c1c6->c4",
            "log": root / "results" / "longrun_PHM_c1c6_to_c4_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1.log",
            "curve_png": root
            / "results"
            / "forecast_PHM_c1c6_to_c4_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
            / "wear_full_curve_trueRaw_predWindows.png",
        },
        {
            "name": "c4c6->c1",
            "log": root / "results" / "longrun_PHM_c4c6_to_c1_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1.log",
            "curve_png": root
            / "results"
            / "forecast_PHM_c4c6_to_c1_rms7_plus_feat4_dual_seed2026_e1000_bt96_gpu1_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
            / "wear_full_curve_trueRaw_predWindows.png",
        },
    ]

    for fold in folds:
        if not fold["log"].exists():
            raise FileNotFoundError(f"Missing log file: {fold['log']}")
        if not fold["curve_png"].exists():
            raise FileNotFoundError(f"Missing curve image: {fold['curve_png']}")

    metric_map = {}
    for fold in folds:
        parsed = parse_metrics(fold["log"])
        required = {"window", "fullcurve", "fullcurve_raw"}
        if not required.issubset(parsed.keys()):
            raise RuntimeError(f"Metrics missing in {fold['log']}: got {list(parsed.keys())}")
        metric_map[fold["name"]] = parsed

    write_csv(folds, metric_map, output_dir / "crossfold_metrics.csv")
    plot_bars(folds, metric_map, output_dir / "crossfold_metrics_bar.png")
    plot_curve_montage(folds, output_dir / "crossfold_wear_curve_montage.png")

    print(f"[OK] visualization generated in: {output_dir}")
    print(f"[OK] csv: {output_dir / 'crossfold_metrics.csv'}")
    print(f"[OK] bar: {output_dir / 'crossfold_metrics_bar.png'}")
    print(f"[OK] montage: {output_dir / 'crossfold_wear_curve_montage.png'}")


if __name__ == "__main__":
    main()
