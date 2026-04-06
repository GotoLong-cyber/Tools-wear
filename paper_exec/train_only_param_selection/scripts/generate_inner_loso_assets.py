#!/usr/bin/env python3
"""Generate supplementary inner-LOSO assets.

This script does not affect the formal mainline results. It only supplements
the inner-LOSO hyperparameter-selection protocol with:
1. per-inner-task MAE/RMSE summaries under the final global configuration
2. per-inner-task wear-curve figures for visual inspection
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper_exec.train_only_param_selection.scripts.select_knn_hparams_inner_loso import (
    FOLD_SPECS,
    prepare_inner_context,
)
from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_delta_retrieval import (
    cosine_knn_predict_with_meta,
    select_library_mask,
)
from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_retrieval import (
    metrics_from_full_curve,
    reconstruct_full_curve,
    stage_metrics,
)


def _sanitize_inner_id(inner_id: str) -> str:
    return inner_id.replace(":", "_").replace("->", "_to_")


def _compute_selected_prediction(ctx: dict, k: int, beta: float, late_q: float):
    lib_mask, lib_thr = select_library_mask(
        ctx["train_current_last"], threshold_um=0.0, quantile=float(late_q)
    )
    lib_repr = ctx["train_repr"][lib_mask]
    lib_delta_targets = ctx["train_delta_targets"][lib_mask]
    delta_knn, _, _ = cosine_knn_predict_with_meta(
        lib_repr, lib_delta_targets, ctx["query_repr"], k=int(k)
    )
    delta_head = ctx["head_pred"] - ctx["query_current_last"][:, None]
    blend_pred = ctx["head_pred"] + float(beta) * (delta_knn - delta_head)
    pred_full, pred_steps, true_raw_full = reconstruct_full_curve(
        blend_pred, ctx["query_targets"], ctx["query_data"], seq_len=ctx["seq_len"]
    )
    return pred_full, pred_steps, true_raw_full, float(lib_thr)


def _plot_curves(inner_id: str, true_curve: np.ndarray, head_curve: np.ndarray, selected_curve: np.ndarray, out_base: Path) -> None:
    x = np.arange(len(true_curve))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(x, true_curve, label="真实磨损", linewidth=2.0, color="#1f1f1f")
    ax.plot(x, head_curve, label="head-only", linewidth=1.8, color="#1f77b4")
    ax.plot(x, selected_curve, label="global-selected", linewidth=1.8, color="#d62728")
    ax.set_title(f"Inner LOSO: {inner_id}")
    ax.set_xlabel("时间步")
    ax.set_ylabel("磨损值 (um)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".png"), dpi=180)
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--config_json", type=Path, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = args.results_dir.resolve()
    config_json = args.config_json.resolve()
    figures_dir = project_root / "paper_exec" / "train_only_param_selection" / "figures"
    manifests_dir = project_root / "paper_exec" / "train_only_param_selection" / "manifests"
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(config_json.read_text(encoding="utf-8"))
    selected_k = int(config["k"])
    selected_beta = float(config["beta"])
    selected_late_q = float(config["late_library_quantile"])

    task_rows = []
    stage_rows = []

    for spec in FOLD_SPECS:
        src_runs = [x.strip() for x in spec.train_runs.split(",") if x.strip()]
        for library_run, query_run in [(src_runs[0], src_runs[1]), (src_runs[1], src_runs[0])]:
            ctx = prepare_inner_context(
                project_root, spec, library_run, query_run,
                results_subdir="paper_exec_inner_loso_assets"
            )
            inner_id = ctx["id"]

            head_pred_full, _, true_raw_full = reconstruct_full_curve(
                ctx["head_pred"], ctx["query_targets"], ctx["query_data"], seq_len=ctx["seq_len"]
            )
            head_metrics = metrics_from_full_curve(head_pred_full, true_raw_full, seq_len=ctx["seq_len"])
            for row in stage_metrics(head_pred_full, true_raw_full, ctx["stage_info"]):
                stage_rows.append({
                    "inner_id": inner_id,
                    "outer_fold": spec.fold,
                    "library_run": library_run,
                    "query_run": query_run,
                    "setting": "head-only",
                    "k": "",
                    "beta": "",
                    "late_q": "",
                    **row,
                })
            task_rows.append({
                "inner_id": inner_id,
                "outer_fold": spec.fold,
                "library_run": library_run,
                "query_run": query_run,
                "setting": "head-only",
                "k": "",
                "beta": "",
                "late_q": "",
                "library_wear_threshold_um": "",
                **head_metrics,
            })

            sel_pred_full, _, _, lib_thr = _compute_selected_prediction(
                ctx, selected_k, selected_beta, selected_late_q
            )
            sel_metrics = metrics_from_full_curve(sel_pred_full, true_raw_full, seq_len=ctx["seq_len"])
            for row in stage_metrics(sel_pred_full, true_raw_full, ctx["stage_info"]):
                stage_rows.append({
                    "inner_id": inner_id,
                    "outer_fold": spec.fold,
                    "library_run": library_run,
                    "query_run": query_run,
                    "setting": f"global-selected delta-blend@k{selected_k}_b{str(selected_beta).replace('.', '')}",
                    "k": selected_k,
                    "beta": selected_beta,
                    "late_q": selected_late_q,
                    **row,
                })
            task_rows.append({
                "inner_id": inner_id,
                "outer_fold": spec.fold,
                "library_run": library_run,
                "query_run": query_run,
                "setting": f"global-selected delta-blend@k{selected_k}_b{str(selected_beta).replace('.', '')}",
                "k": selected_k,
                "beta": selected_beta,
                "late_q": selected_late_q,
                "library_wear_threshold_um": lib_thr,
                **sel_metrics,
            })

            out_base = figures_dir / f"inner_loso_curve_{_sanitize_inner_id(inner_id)}"
            _plot_curves(inner_id, true_raw_full, head_pred_full, sel_pred_full, out_base)

    task_csv = results_dir / "inner_loso_task_metrics.csv"
    with task_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "inner_id", "outer_fold", "library_run", "query_run", "setting",
            "k", "beta", "late_q", "library_wear_threshold_um",
            "mse_full_raw", "rmse_full_raw", "mae_full_raw", "valid_points",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(task_rows)

    stage_csv = results_dir / "inner_loso_task_stage_metrics.csv"
    with stage_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "inner_id", "outer_fold", "library_run", "query_run", "setting",
            "k", "beta", "late_q", "stage", "num_points",
            "wear_min_um", "wear_max_um", "mae_um", "rmse_um",
            "mean_residual_um", "underest_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stage_rows)

    df = pd.DataFrame(task_rows)
    summary_md = manifests_dir / "InnerLOSO曲线资产汇总.md"
    lines = [
        "# Inner LOSO 曲线与指标补充说明",
        "",
        "本文件补充 `inner LOSO` 选参协议下缺失的两类资产：",
        "1. 每个 inner task 在最终全局固定参数下的 MAE/RMSE 汇总",
        "2. 每个 inner task 的真实曲线、head-only 曲线与 global-selected 曲线图",
        "",
        f"- 全局固定参数：`k={selected_k}, beta={selected_beta}, late_q={selected_late_q}`",
        f"- 指标文件：`{task_csv}`",
        f"- 分阶段文件：`{stage_csv}`",
        "",
        "## 每个 inner task 的全局配置结果",
        "",
    ]
    selected_df = df[df["setting"].str.startswith("global-selected")].copy()
    for _, row in selected_df.iterrows():
        lines.append(
            f"- {row['inner_id']}: MAE={row['mae_full_raw']:.4f}, RMSE={row['rmse_full_raw']:.4f}, "
            f"library_run={row['library_run']}, query_run={row['query_run']}"
        )
    lines.extend([
        "",
        "## 曲线图文件",
        "",
    ])
    for path in sorted(figures_dir.glob("inner_loso_curve_*.png")):
        lines.append(f"- `{path}`")
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INNER-LOSO-ASSETS][OK] task_csv={task_csv}")
    print(f"[INNER-LOSO-ASSETS][OK] stage_csv={stage_csv}")
    print(f"[INNER-LOSO-ASSETS][OK] figures_dir={figures_dir}")


if __name__ == "__main__":
    main()
