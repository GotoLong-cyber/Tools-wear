#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM")
OUT_ROOT = ROOT / "paper_figures"


@dataclass
class CopySpec:
    target_subdir: str
    filename: str
    source: Path
    title: str
    purpose: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_figures(specs: list[CopySpec]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in specs:
        target_dir = OUT_ROOT / spec.target_subdir
        ensure_dir(target_dir)
        target_path = target_dir / spec.filename
        shutil.copy2(spec.source, target_path)
        rows.append(
            {
                "category": spec.target_subdir,
                "filename": spec.filename,
                "title": spec.title,
                "purpose": spec.purpose,
                "source": str(spec.source.relative_to(ROOT)),
            }
        )
    return rows


def read_overall_metrics(path: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            result[row["mode"]] = {
                "mae": float(row["mae_full_raw"]),
                "rmse": float(row["rmse_full_raw"]),
            }
    return result


def read_stage_metrics(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"]
            stage = row["stage"]
            result.setdefault(mode, {})[stage] = {
                "mae": float(row.get("MAE", row.get("mae_um"))),
                "rmse": float(row.get("RMSE", row.get("rmse_um"))),
                "mean_residual": float(row.get("mean_residual", row.get("mean_residual_um"))),
                "underest_ratio": float(row["underest_ratio"]),
            }
    return result


def add_bar_labels(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_three_fold_main_results(manifest_rows: list[dict[str, str]]) -> None:
    fold_files = {
        "fold1": ROOT / "results/20260325_RetrievalV21_formal/knn_delta_fold1_overall_metrics.csv",
        "fold2": ROOT / "results/20260325_RetrievalV21_formal/knn_delta_fold2_overall_metrics.csv",
        "fold3": ROOT / "results/20260325_RetrievalV21_formal/knn_delta_fold3_overall_metrics.csv",
    }
    folds = list(fold_files.keys())
    head_mae, retr_mae, head_rmse, retr_rmse = [], [], [], []
    for fold in folds:
        metrics = read_overall_metrics(fold_files[fold])
        head_mae.append(metrics["head-only"]["mae"])
        retr_mae.append(metrics["delta-blend@k5_b05"]["mae"])
        head_rmse.append(metrics["head-only"]["rmse"])
        retr_rmse.append(metrics["delta-blend@k5_b05"]["rmse"])

    x = range(len(folds))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    bars1 = axes[0].bar([i - width / 2 for i in x], head_mae, width, label="Current-best", color="#6b7280")
    bars2 = axes[0].bar([i + width / 2 for i in x], retr_mae, width, label="Retrieval V2.1", color="#1d4ed8")
    axes[0].set_title("Three-Fold MAE")
    axes[0].set_xticks(list(x), folds)
    axes[0].set_ylabel("MAE (um)")
    axes[0].legend(frameon=False)
    add_bar_labels(axes[0], bars1)
    add_bar_labels(axes[0], bars2)

    bars3 = axes[1].bar([i - width / 2 for i in x], head_rmse, width, label="Current-best", color="#9ca3af")
    bars4 = axes[1].bar([i + width / 2 for i in x], retr_rmse, width, label="Retrieval V2.1", color="#2563eb")
    axes[1].set_title("Three-Fold RMSE")
    axes[1].set_xticks(list(x), folds)
    axes[1].set_ylabel("RMSE (um)")
    add_bar_labels(axes[1], bars3)
    add_bar_labels(axes[1], bars4)

    target = OUT_ROOT / "03_main_results" / "Fig08_RetrievalV21_ThreeFold_MAE_RMSE.png"
    ensure_dir(target.parent)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest_rows.append(
        {
            "category": "03_main_results",
            "filename": target.name,
            "title": "Retrieval V2.1 three-fold MAE/RMSE comparison",
            "purpose": "Main quantitative result figure for paper results section.",
            "source": "generated from results/20260325_RetrievalV21_formal/knn_delta_fold*_overall_metrics.csv",
        }
    )


def plot_fold1_stage_repair(manifest_rows: list[dict[str, str]]) -> None:
    stage_file = ROOT / "results/20260325_RetrievalV21_formal/knn_delta_fold1_stage_metrics.csv"
    metrics = read_stage_metrics(stage_file)
    stages = ["early", "mid", "late"]
    head_mae = [metrics["head-only"][s]["mae"] for s in stages]
    retr_mae = [metrics["delta-blend@k5_b05"][s]["mae"] for s in stages]
    head_res = [metrics["head-only"][s]["mean_residual"] for s in stages]
    retr_res = [metrics["delta-blend@k5_b05"][s]["mean_residual"] for s in stages]

    x = range(len(stages))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    bars1 = axes[0].bar([i - width / 2 for i in x], head_mae, width, label="Current-best", color="#6b7280")
    bars2 = axes[0].bar([i + width / 2 for i in x], retr_mae, width, label="Retrieval V2.1", color="#1d4ed8")
    axes[0].set_title("Fold1 Stage MAE")
    axes[0].set_xticks(list(x), stages)
    axes[0].set_ylabel("MAE (um)")
    axes[0].legend(frameon=False)
    add_bar_labels(axes[0], bars1)
    add_bar_labels(axes[0], bars2)

    bars3 = axes[1].bar([i - width / 2 for i in x], head_res, width, label="Current-best", color="#9ca3af")
    bars4 = axes[1].bar([i + width / 2 for i in x], retr_res, width, label="Retrieval V2.1", color="#2563eb")
    axes[1].axhline(0.0, color="#111827", linewidth=1.0, linestyle="--")
    axes[1].set_title("Fold1 Stage Mean Residual")
    axes[1].set_xticks(list(x), stages)
    axes[1].set_ylabel("Mean residual (um)")
    add_bar_labels(axes[1], bars3)
    add_bar_labels(axes[1], bars4)

    target = OUT_ROOT / "04_mechanism" / "Fig11_Fold1_StageRepair_MAE_Residual.png"
    ensure_dir(target.parent)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest_rows.append(
        {
            "category": "04_mechanism",
            "filename": target.name,
            "title": "Fold1 stage repair by Retrieval V2.1",
            "purpose": "Mechanism figure showing late-stage MAE and residual reduction.",
            "source": "generated from results/20260325_RetrievalV21_formal/knn_delta_fold1_stage_metrics.csv",
        }
    )


def plot_retrieval_sensitivity(manifest_rows: list[dict[str, str]]) -> None:
    configs = {
        "K=3,q80": ROOT / "results/20260325_KNNSens3Fold_k3_q80",
        "K=5,q80": ROOT / "results/20260325_RetrievalV21_formal",
        "K=10,q80": ROOT / "results/20260325_KNNSens3Fold_k10_q80",
        "K=5,q0": ROOT / "results/20260325_KNNSens3Fold_k5_q0",
        "beta=0.3": ROOT / "results/20260325_KNNSens3Fold_k5_q80_b03",
        "beta=0.5": ROOT / "results/20260325_RetrievalV21_formal",
    }

    def avg_metric(folder: Path, metric: str) -> float:
        vals = []
        for fold in ("fold1", "fold2", "fold3"):
            overall = read_overall_metrics(folder / f"knn_delta_{fold}_overall_metrics.csv")
            blend_key = next(key for key in overall if key.startswith("delta-blend@"))
            vals.append(overall[blend_key][metric])
        return sum(vals) / len(vals)

    kq_labels = ["K=3,q80", "K=5,q80", "K=10,q80", "K=5,q0"]
    kq_mae = [avg_metric(configs[label], "mae") for label in kq_labels]
    beta_labels = ["beta=0.3", "beta=0.5"]
    beta_mae = [avg_metric(configs[label], "mae") for label in beta_labels]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    bars1 = axes[0].bar(kq_labels, kq_mae, color=["#93c5fd", "#2563eb", "#60a5fa", "#1e40af"])
    axes[0].set_title("Sensitivity to K and q")
    axes[0].set_ylabel("Average MAE (um)")
    axes[0].tick_params(axis="x", rotation=20)
    add_bar_labels(axes[0], bars1)

    bars2 = axes[1].bar(beta_labels, beta_mae, color=["#94a3b8", "#0f766e"])
    axes[1].set_title("Sensitivity to beta")
    axes[1].set_ylabel("Average MAE (um)")
    add_bar_labels(axes[1], bars2)

    target = OUT_ROOT / "05_ablation" / "Fig12_RetrievalV21_Sensitivity_K_q_beta.png"
    ensure_dir(target.parent)
    fig.savefig(target, dpi=300, bbox_inches="tight")
    plt.close(fig)

    manifest_rows.append(
        {
            "category": "05_ablation",
            "filename": target.name,
            "title": "Retrieval V2.1 sensitivity to K, q, and beta",
            "purpose": "Ablation figure showing robustness to key retrieval hyperparameters.",
            "source": "generated from formal/sensitivity three-fold overall metrics",
        }
    )


def write_manifest(rows: list[dict[str, str]]) -> None:
    ensure_dir(OUT_ROOT)
    csv_path = OUT_ROOT / "paper_figures_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "filename", "title", "purpose", "source"])
        writer.writeheader()
        writer.writerows(rows)

    md_path = OUT_ROOT / "README.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Paper Figures\n\n")
        f.write("This directory contains paper-ready figures copied or generated from the final experiment artifacts.\n\n")
        f.write("| Category | Filename | Title | Purpose | Source |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for row in rows:
            f.write(
                f"| `{row['category']}` | `{row['filename']}` | {row['title']} | {row['purpose']} | `{row['source']}` |\n"
            )


def main() -> None:
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    ensure_dir(OUT_ROOT)

    copy_specs = [
        CopySpec(
            "01_problem_setup",
            "Fig01_A1_C1C4C6_TimeScale_CurveComparison.png",
            ROOT / "feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_c1c4c6_curve_compare.png",
            "C1/C4/C6 wear curve comparison",
            "Motivates cross-condition time-scale mismatch.",
        ),
        CopySpec(
            "01_problem_setup",
            "Fig02_A1_C1C4C6_NormalizedProgressCurves.png",
            ROOT / "feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_wear_curves_normalized_progress.png",
            "Normalized progress wear curves",
            "Shows normalized trajectory mismatch across tools.",
        ),
        CopySpec(
            "01_problem_setup",
            "Fig03_A1_C1C4C6_StageSlopeComparison.png",
            ROOT / "feature_alignment_diagnosis/outputs/A1_time_scale_20260323_1539/a1_stage_slopes.png",
            "Stage slope comparison",
            "Shows stage-wise degradation slope differences.",
        ),
        CopySpec(
            "02_method",
            "Fig04_A2_Baseline_vs_A2_Structure.svg",
            ROOT / "feature_alignment_diagnosis/outputs/A2_dynstride_diagnostics_20260323_1743/baseline_vs_a2final_structure.svg",
            "Baseline vs A2 structure",
            "Method-side structure comparison for A2 augmentation stage.",
        ),
        CopySpec(
            "03_main_results",
            "Fig05_RetrievalV21_Fold1_CurrentBest_vs_Retrieval.png",
            ROOT / "results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold1.png",
            "Fold1 current-best vs Retrieval V2.1",
            "Main result curve for hardest fold.",
        ),
        CopySpec(
            "03_main_results",
            "Fig06_RetrievalV21_Fold2_CurrentBest_vs_Retrieval.png",
            ROOT / "results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold2.png",
            "Fold2 current-best vs Retrieval V2.1",
            "Main result curve for fold2.",
        ),
        CopySpec(
            "03_main_results",
            "Fig07_RetrievalV21_Fold3_CurrentBest_vs_Retrieval.png",
            ROOT / "results/20260325_RetrievalV21_formal/wear_full_curve_knn_delta_compare_fold3.png",
            "Fold3 current-best vs Retrieval V2.1",
            "Main result curve for fold3.",
        ),
        CopySpec(
            "04_mechanism",
            "Fig09_Fold1_CurrentBest_StageCurve.png",
            ROOT / "feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_currentbest/fold1_stage_curve_colored.png",
            "Fold1 stage-colored wear curve",
            "Highlights early/mid/late partition on current-best fold1.",
        ),
        CopySpec(
            "04_mechanism",
            "Fig10_Fold1_CurrentBest_StageResiduals.png",
            ROOT / "feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_currentbest/fold1_stage_residuals.png",
            "Fold1 stage residual curve",
            "Shows late-stage systematic underestimation before retrieval correction.",
        ),
    ]

    rows = copy_figures(copy_specs)
    plot_three_fold_main_results(rows)
    plot_fold1_stage_repair(rows)
    plot_retrieval_sensitivity(rows)
    write_manifest(rows)


if __name__ == "__main__":
    main()
