#!/usr/bin/env python3
"""Train-only validation protocol for retrieval hyperparameter selection.

This script never touches outer test folds for model selection. It evaluates
KNN retrieval hyperparameters on source-domain validation splits only, using
clean retrieval-backbone checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_provider.data_factory import data_provider
from exp.exp_forecast import Exp_Forecast
from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_retrieval import (
    build_args,
    build_stage_info,
    extract_raw_target_sequences,
    extract_repr_and_head_preds,
    load_checkpoint,
    make_eval_loader,
    metrics_from_full_curve,
    reconstruct_full_curve,
    stage_metrics,
)
from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_delta_retrieval import (
    cosine_knn_predict_with_meta,
    extract_current_last_wear,
    select_library_mask,
)


@dataclass(frozen=True)
class FoldSpec:
    fold: str
    train_runs: str
    test_runs: str
    runtime_root: str
    checkpoint_path: str


FOLD_SPECS = [
    FoldSpec(
        fold="fold1",
        train_runs="c1,c4",
        test_runs="c6",
        runtime_root="dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_tma_clean_fold1_gpu3",
        checkpoint_path="checkpoints/forecast_PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_TMAClean_dual_seed2026_e1000_bt96_gpu3_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth",
    ),
    FoldSpec(
        fold="fold2",
        train_runs="c4,c6",
        test_runs="c1",
        runtime_root="dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_tma_clean_fold2_gpu3",
        checkpoint_path="checkpoints/forecast_PHM_c4c6_to_c1_rms7_plus_feat4_plus_se1_TMAClean_dual_seed2026_e1000_bt96_gpu3_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth",
    ),
    FoldSpec(
        fold="fold3",
        train_runs="c1,c6",
        test_runs="c4",
        runtime_root="dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_tma_clean_fold3_gpu3",
        checkpoint_path="checkpoints/forecast_PHM_c1c6_to_c4_rms7_plus_feat4_plus_se1_TMAClean_dual_seed2026_e1000_bt96_gpu3_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth",
    ),
]


def parse_int_grid(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_float_grid(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def prepare_fold_context(project_root: Path, spec: FoldSpec, results_subdir: str) -> dict:
    runtime_root = (project_root / spec.runtime_root).resolve()
    checkpoint_path = (project_root / spec.checkpoint_path).resolve()
    cfg = build_args(project_root, runtime_root, checkpoint_path, results_subdir)
    cfg.train_runs = spec.train_runs
    cfg.test_runs = spec.test_runs

    exp = Exp_Forecast(cfg)
    load_checkpoint(exp, checkpoint_path)

    args_train = copy.deepcopy(cfg)
    args_train.train_stride_candidates = "1"
    args_train.train_stride_quantiles = ""

    train_data, _ = data_provider(args_train, "train")
    val_data, _ = data_provider(cfg, "val")
    train_loader = make_eval_loader(train_data, cfg.batch_size)
    val_loader = make_eval_loader(val_data, cfg.batch_size)

    train_repr, _ = extract_repr_and_head_preds(exp, train_data, train_loader)
    val_repr, head_pred = extract_repr_and_head_preds(exp, val_data, val_loader)

    horizon = int(cfg.test_pred_len)
    seq_len = int(cfg.test_seq_len)
    train_targets = extract_raw_target_sequences(train_data, horizon=horizon, seq_len=seq_len)
    val_targets = extract_raw_target_sequences(val_data, horizon=horizon, seq_len=seq_len)
    train_current_last = extract_current_last_wear(train_data, seq_len=seq_len)
    val_current_last = extract_current_last_wear(val_data, seq_len=seq_len)
    train_delta_targets = train_targets - train_current_last[:, None]

    head_pred_full, _, true_raw_full = reconstruct_full_curve(head_pred, val_targets, val_data, seq_len=seq_len)
    head_metrics = metrics_from_full_curve(head_pred_full, true_raw_full, seq_len=seq_len)
    stage_info = build_stage_info(true_raw_full, seq_len=seq_len)
    head_stage = stage_metrics(head_pred_full, true_raw_full, stage_info)

    return {
        "cfg": cfg,
        "train_data": train_data,
        "val_data": val_data,
        "train_repr": train_repr,
        "val_repr": val_repr,
        "head_pred": head_pred,
        "train_delta_targets": train_delta_targets,
        "train_current_last": train_current_last,
        "val_current_last": val_current_last,
        "val_targets": val_targets,
        "head_pred_full": head_pred_full,
        "true_raw_full": true_raw_full,
        "head_metrics": head_metrics,
        "head_stage": head_stage,
        "seq_len": seq_len,
        "stage_info": stage_info,
    }


def evaluate_grid_on_context(ctx: dict, k_grid: list[int], beta_grid: list[float], q_grid: list[float]) -> tuple[list[dict], list[dict]]:
    overall_rows: list[dict] = []
    stage_rows: list[dict] = []
    lib_cache: dict[float, tuple[np.ndarray, np.ndarray, float]] = {}
    knn_cache: dict[tuple[float, int], tuple[np.ndarray, np.ndarray]] = {}

    head_metrics = ctx["head_metrics"]
    overall_rows.append({
        "fold": "",
        "k": "",
        "beta": "",
        "late_q": "",
        "mode": "head-only",
        **head_metrics,
    })

    for row in ctx["head_stage"]:
        stage_rows.append({
            "fold": "",
            "k": "",
            "beta": "",
            "late_q": "",
            "mode": "head-only",
            **row,
        })

    for late_q in q_grid:
        if late_q not in lib_cache:
            lib_mask, lib_thr = select_library_mask(
                ctx["train_current_last"],
                threshold_um=0.0,
                quantile=float(late_q),
            )
            lib_cache[late_q] = (
                ctx["train_repr"][lib_mask],
                ctx["train_delta_targets"][lib_mask],
                float(lib_thr),
            )

        lib_repr, lib_delta_targets, lib_thr = lib_cache[late_q]
        for k in k_grid:
            cache_key = (late_q, k)
            if cache_key not in knn_cache:
                delta_knn, _, _ = cosine_knn_predict_with_meta(
                    lib_repr,
                    lib_delta_targets,
                    ctx["val_repr"],
                    k=int(k),
                )
                knn_cache[cache_key] = (delta_knn, np.asarray([], dtype=np.float32))
            delta_knn, _ = knn_cache[cache_key]

            knn_abs = ctx["val_current_last"][:, None] + delta_knn
            knn_mode = f"delta-knn-only@k{int(k)}"
            knn_pred_full, _, _ = reconstruct_full_curve(knn_abs, ctx["val_targets"], ctx["val_data"], seq_len=ctx["seq_len"])
            knn_metrics = metrics_from_full_curve(knn_pred_full, ctx["true_raw_full"], seq_len=ctx["seq_len"])
            overall_rows.append({
                "fold": "",
                "k": int(k),
                "beta": "",
                "late_q": float(late_q),
                "mode": knn_mode,
                "library_wear_threshold_um": lib_thr,
                **knn_metrics,
            })

            if not any(
                r["mode"] == knn_mode and r["k"] == int(k) and r["late_q"] == float(late_q)
                for r in stage_rows
            ):
                for row in stage_metrics(knn_pred_full, ctx["true_raw_full"], ctx["stage_info"]):
                    stage_rows.append({
                        "fold": "",
                        "k": int(k),
                        "beta": "",
                        "late_q": float(late_q),
                        "mode": knn_mode,
                        **row,
                    })

            delta_head = ctx["head_pred"] - ctx["val_current_last"][:, None]
            for beta in beta_grid:
                blend_pred = ctx["head_pred"] + float(beta) * (delta_knn - delta_head)
                mode = f"delta-blend@k{int(k)}_b{str(beta).replace('.', '')}"
                pred_full, _, _ = reconstruct_full_curve(blend_pred, ctx["val_targets"], ctx["val_data"], seq_len=ctx["seq_len"])
                cur_metrics = metrics_from_full_curve(pred_full, ctx["true_raw_full"], seq_len=ctx["seq_len"])
                overall_rows.append({
                    "fold": "",
                    "k": int(k),
                    "beta": float(beta),
                    "late_q": float(late_q),
                    "mode": mode,
                    "library_wear_threshold_um": lib_thr,
                    **cur_metrics,
                })
                for row in stage_metrics(pred_full, ctx["true_raw_full"], ctx["stage_info"]):
                    stage_rows.append({
                        "fold": "",
                        "k": int(k),
                        "beta": float(beta),
                        "late_q": float(late_q),
                        "mode": mode,
                        **row,
                    })

    return overall_rows, stage_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path, required=True)
    parser.add_argument("--k_grid", type=str, default="3,5,10")
    parser.add_argument("--beta_grid", type=str, default="0.3,0.5,0.7")
    parser.add_argument("--late_q_grid", type=str, default="0.0,0.8")
    parser.add_argument("--selection_metric", type=str, default="mae_full_raw")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    k_grid = parse_int_grid(args.k_grid)
    beta_grid = parse_float_grid(args.beta_grid)
    q_grid = parse_float_grid(args.late_q_grid)

    grid_rows: list[dict] = []
    stage_rows_all: list[dict] = []
    best_rows: list[dict] = []

    for spec in FOLD_SPECS:
        ctx = prepare_fold_context(project_root, spec, results_subdir="paper_exec_train_only_param_select")
        overall_rows, stage_rows = evaluate_grid_on_context(ctx, k_grid, beta_grid, q_grid)

        for row in overall_rows:
            row["fold"] = spec.fold
            grid_rows.append(row)
        for row in stage_rows:
            row["fold"] = spec.fold
            stage_rows_all.append(row)

        candidate_rows = [
            r for r in overall_rows
            if str(r["mode"]).startswith("delta-blend@")
        ]
        best_row = min(candidate_rows, key=lambda x: float(x[args.selection_metric]))
        best_rows.append({
            "fold": spec.fold,
            "selected_k": int(best_row["k"]),
            "selected_beta": float(best_row["beta"]),
            "selected_late_q": float(best_row["late_q"]),
            "selected_mode": best_row["mode"],
            "selected_metric": float(best_row[args.selection_metric]),
        })

    grid_csv = results_dir / "train_only_knn_grid_all.csv"
    with grid_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold", "k", "beta", "late_q", "mode",
            "library_wear_threshold_um", "mse_full_raw", "rmse_full_raw",
            "mae_full_raw", "valid_points",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid_rows:
            out = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(out)

    stage_csv = results_dir / "train_only_knn_stage_all.csv"
    with stage_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "fold", "k", "beta", "late_q", "mode", "stage", "num_points",
            "wear_min_um", "wear_max_um", "mae_um", "rmse_um",
            "mean_residual_um", "underest_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stage_rows_all:
            out = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(out)

    best_csv = results_dir / "train_only_knn_fold_best.csv"
    with best_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["fold", "selected_k", "selected_beta", "selected_late_q", "selected_mode", "selected_metric"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(best_rows)

    summary_rows = []
    grouped = {}
    for row in grid_rows:
        if not str(row["mode"]).startswith("delta-blend@"):
            continue
        key = (int(row["k"]), float(row["beta"]), float(row["late_q"]))
        grouped.setdefault(key, []).append(float(row[args.selection_metric]))

    for (k, beta, late_q), vals in grouped.items():
        if len(vals) != len(FOLD_SPECS):
            continue
        summary_rows.append({
            "k": k,
            "beta": beta,
            "late_q": late_q,
            "mean_validation_mae": float(np.mean(vals)),
            "std_validation_mae": float(np.std(vals)),
            "num_folds": len(vals),
        })

    summary_rows.sort(key=lambda x: (x["mean_validation_mae"], x["std_validation_mae"], x["k"], x["beta"], x["late_q"]))

    summary_csv = results_dir / "train_only_knn_global_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["k", "beta", "late_q", "mean_validation_mae", "std_validation_mae", "num_folds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    selected = summary_rows[0]
    selected_json = {
        "k": int(selected["k"]),
        "beta": float(selected["beta"]),
        "late_library_quantile": float(selected["late_q"]),
        "selection_metric": args.selection_metric,
        "selection_protocol": "train-only validation across source-domain folds",
        "selection_source": "paper_exec/train_only_param_selection/results/train_only_knn_global_summary.csv",
        "note": "Selected without using outer test folds.",
    }
    selected_json_path = results_dir / "selected_knn_config_train_only.json"
    selected_json_path.write_text(json.dumps(selected_json, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = project_root / "paper_exec" / "train_only_param_selection" / "manifests" / "仅训练选参汇总.md"
    lines = [
        "# Train-Only Selection Summary",
        "",
        f"- 候选 `k`: `{k_grid}`",
        f"- 候选 `beta`: `{beta_grid}`",
        f"- 候选 `late_q`: `{q_grid}`",
        f"- 选择指标: `{args.selection_metric}`",
        "",
        "## 每折最优",
        "",
    ]
    for row in best_rows:
        lines.append(
            f"- {row['fold']}: k={row['selected_k']}, beta={row['selected_beta']}, late_q={row['selected_late_q']}, "
            f"metric={row['selected_metric']:.4f}"
        )
    lines.extend([
        "",
        "## 全局固定参数",
        "",
        f"- k={selected_json['k']}",
        f"- beta={selected_json['beta']}",
        f"- late_q={selected_json['late_library_quantile']}",
        f"- mean_validation_mae={selected['mean_validation_mae']:.4f}",
        f"- std_validation_mae={selected['std_validation_mae']:.4f}",
        "",
        "## 输出文件",
        "",
        f"- `{grid_csv}`",
        f"- `{stage_csv}`",
        f"- `{best_csv}`",
        f"- `{summary_csv}`",
        f"- `{selected_json_path}`",
    ])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[TRAIN-ONLY-SELECT][OK] grid_csv={grid_csv}")
    print(f"[TRAIN-ONLY-SELECT][OK] summary_csv={summary_csv}")
    print(f"[TRAIN-ONLY-SELECT][OK] selected={selected_json}")


if __name__ == "__main__":
    main()
