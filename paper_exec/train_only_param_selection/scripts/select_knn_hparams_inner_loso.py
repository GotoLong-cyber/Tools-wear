#!/usr/bin/env python3
"""Inner leave-one-source-run-out validation for retrieval hyperparameter selection.

For each outer fold with two source runs:
- use source run A train split as retrieval library
- use the other source run B full run as pseudo-target query set
- repeat B->A

This keeps hyperparameter selection away from outer test folds while preserving
cross-condition mismatch inside the source domain.
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


def prepare_inner_context(project_root: Path, spec: FoldSpec, library_run: str, query_run: str, results_subdir: str) -> dict:
    runtime_root = (project_root / spec.runtime_root).resolve()
    checkpoint_path = (project_root / spec.checkpoint_path).resolve()

    cfg = build_args(project_root, runtime_root, checkpoint_path, results_subdir)
    cfg.train_runs = spec.train_runs
    cfg.test_runs = spec.test_runs

    exp = Exp_Forecast(cfg)
    load_checkpoint(exp, checkpoint_path)

    args_train = copy.deepcopy(cfg)
    args_train.train_runs = library_run
    args_train.test_runs = query_run
    args_train.train_stride_candidates = "1"
    args_train.train_stride_quantiles = ""

    args_query = copy.deepcopy(cfg)
    args_query.train_runs = library_run
    args_query.test_runs = query_run

    train_data, _ = data_provider(args_train, "train")
    query_data, _ = data_provider(args_query, "test")
    train_loader = make_eval_loader(train_data, cfg.batch_size)
    query_loader = make_eval_loader(query_data, cfg.batch_size)

    train_repr, _ = extract_repr_and_head_preds(exp, train_data, train_loader)
    query_repr, head_pred = extract_repr_and_head_preds(exp, query_data, query_loader)

    horizon = int(cfg.test_pred_len)
    seq_len = int(cfg.test_seq_len)
    train_targets = extract_raw_target_sequences(train_data, horizon=horizon, seq_len=seq_len)
    query_targets = extract_raw_target_sequences(query_data, horizon=horizon, seq_len=seq_len)
    train_current_last = extract_current_last_wear(train_data, seq_len=seq_len)
    query_current_last = extract_current_last_wear(query_data, seq_len=seq_len)
    train_delta_targets = train_targets - train_current_last[:, None]

    head_pred_full, _, true_raw_full = reconstruct_full_curve(head_pred, query_targets, query_data, seq_len=seq_len)
    head_metrics = metrics_from_full_curve(head_pred_full, true_raw_full, seq_len=seq_len)
    stage_info = build_stage_info(true_raw_full, seq_len=seq_len)
    head_stage = stage_metrics(head_pred_full, true_raw_full, stage_info)

    return {
        "id": f"{spec.fold}:{library_run}->{query_run}",
        "outer_fold": spec.fold,
        "library_run": library_run,
        "query_run": query_run,
        "train_data": train_data,
        "query_data": query_data,
        "train_repr": train_repr,
        "query_repr": query_repr,
        "head_pred": head_pred,
        "train_delta_targets": train_delta_targets,
        "train_current_last": train_current_last,
        "query_current_last": query_current_last,
        "query_targets": query_targets,
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
    knn_cache: dict[tuple[float, int], np.ndarray] = {}

    overall_rows.append({
        "inner_id": ctx["id"],
        "outer_fold": ctx["outer_fold"],
        "library_run": ctx["library_run"],
        "query_run": ctx["query_run"],
        "k": "",
        "beta": "",
        "late_q": "",
        "mode": "head-only",
        **ctx["head_metrics"],
    })
    for row in ctx["head_stage"]:
        stage_rows.append({
            "inner_id": ctx["id"],
            "outer_fold": ctx["outer_fold"],
            "library_run": ctx["library_run"],
            "query_run": ctx["query_run"],
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
            key = (late_q, k)
            if key not in knn_cache:
                delta_knn, _, _ = cosine_knn_predict_with_meta(
                    lib_repr,
                    lib_delta_targets,
                    ctx["query_repr"],
                    k=int(k),
                )
                knn_cache[key] = delta_knn
            delta_knn = knn_cache[key]
            delta_head = ctx["head_pred"] - ctx["query_current_last"][:, None]

            knn_abs = ctx["query_current_last"][:, None] + delta_knn
            knn_mode = f"delta-knn-only@k{int(k)}"
            knn_pred_full, _, _ = reconstruct_full_curve(knn_abs, ctx["query_targets"], ctx["query_data"], seq_len=ctx["seq_len"])
            knn_metrics = metrics_from_full_curve(knn_pred_full, ctx["true_raw_full"], seq_len=ctx["seq_len"])
            overall_rows.append({
                "inner_id": ctx["id"],
                "outer_fold": ctx["outer_fold"],
                "library_run": ctx["library_run"],
                "query_run": ctx["query_run"],
                "k": int(k),
                "beta": "",
                "late_q": float(late_q),
                "mode": knn_mode,
                "library_wear_threshold_um": lib_thr,
                **knn_metrics,
            })

            if not any(
                r["mode"] == knn_mode and r["k"] == int(k) and r["late_q"] == float(late_q) and r["inner_id"] == ctx["id"]
                for r in stage_rows
            ):
                for row in stage_metrics(knn_pred_full, ctx["true_raw_full"], ctx["stage_info"]):
                    stage_rows.append({
                        "inner_id": ctx["id"],
                        "outer_fold": ctx["outer_fold"],
                        "library_run": ctx["library_run"],
                        "query_run": ctx["query_run"],
                        "k": int(k),
                        "beta": "",
                        "late_q": float(late_q),
                        "mode": knn_mode,
                        **row,
                    })

            for beta in beta_grid:
                blend_pred = ctx["head_pred"] + float(beta) * (delta_knn - delta_head)
                mode = f"delta-blend@k{int(k)}_b{str(beta).replace('.', '')}"
                pred_full, _, _ = reconstruct_full_curve(blend_pred, ctx["query_targets"], ctx["query_data"], seq_len=ctx["seq_len"])
                cur_metrics = metrics_from_full_curve(pred_full, ctx["true_raw_full"], seq_len=ctx["seq_len"])
                overall_rows.append({
                    "inner_id": ctx["id"],
                    "outer_fold": ctx["outer_fold"],
                    "library_run": ctx["library_run"],
                    "query_run": ctx["query_run"],
                    "k": int(k),
                    "beta": float(beta),
                    "late_q": float(late_q),
                    "mode": mode,
                    "library_wear_threshold_um": lib_thr,
                    **cur_metrics,
                })
                for row in stage_metrics(pred_full, ctx["true_raw_full"], ctx["stage_info"]):
                    stage_rows.append({
                        "inner_id": ctx["id"],
                        "outer_fold": ctx["outer_fold"],
                        "library_run": ctx["library_run"],
                        "query_run": ctx["query_run"],
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
    inner_best_rows: list[dict] = []

    for spec in FOLD_SPECS:
        src_runs = [x.strip() for x in spec.train_runs.split(",") if x.strip()]
        assert len(src_runs) == 2
        for library_run, query_run in [(src_runs[0], src_runs[1]), (src_runs[1], src_runs[0])]:
            ctx = prepare_inner_context(project_root, spec, library_run, query_run, results_subdir="paper_exec_inner_loso_param_select")
            overall_rows, stage_rows = evaluate_grid_on_context(ctx, k_grid, beta_grid, q_grid)
            grid_rows.extend(overall_rows)
            stage_rows_all.extend(stage_rows)
            candidate_rows = [r for r in overall_rows if str(r["mode"]).startswith("delta-blend@")]
            best_row = min(candidate_rows, key=lambda x: float(x[args.selection_metric]))
            inner_best_rows.append({
                "inner_id": ctx["id"],
                "outer_fold": ctx["outer_fold"],
                "library_run": ctx["library_run"],
                "query_run": ctx["query_run"],
                "selected_k": int(best_row["k"]),
                "selected_beta": float(best_row["beta"]),
                "selected_late_q": float(best_row["late_q"]),
                "selected_mode": best_row["mode"],
                "selected_metric": float(best_row[args.selection_metric]),
            })

    grid_csv = results_dir / "inner_loso_knn_grid_all.csv"
    with grid_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "inner_id", "outer_fold", "library_run", "query_run",
            "k", "beta", "late_q", "mode", "library_wear_threshold_um",
            "mse_full_raw", "rmse_full_raw", "mae_full_raw", "valid_points",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    stage_csv = results_dir / "inner_loso_knn_stage_all.csv"
    with stage_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "inner_id", "outer_fold", "library_run", "query_run",
            "k", "beta", "late_q", "mode", "stage", "num_points",
            "wear_min_um", "wear_max_um", "mae_um", "rmse_um",
            "mean_residual_um", "underest_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stage_rows_all:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    inner_best_csv = results_dir / "inner_loso_knn_inner_best.csv"
    with inner_best_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "inner_id", "outer_fold", "library_run", "query_run",
            "selected_k", "selected_beta", "selected_late_q",
            "selected_mode", "selected_metric",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inner_best_rows)

    grouped = {}
    for row in grid_rows:
        if not str(row["mode"]).startswith("delta-blend@"):
            continue
        key = (int(row["k"]), float(row["beta"]), float(row["late_q"]))
        grouped.setdefault(key, []).append(float(row[args.selection_metric]))

    summary_rows = []
    expected = len(inner_best_rows)
    for (k, beta, late_q), vals in grouped.items():
        if len(vals) != expected:
            continue
        summary_rows.append({
            "k": k,
            "beta": beta,
            "late_q": late_q,
            "mean_inner_validation_mae": float(np.mean(vals)),
            "std_inner_validation_mae": float(np.std(vals)),
            "num_inner_tasks": len(vals),
        })
    summary_rows.sort(key=lambda x: (x["mean_inner_validation_mae"], x["std_inner_validation_mae"], x["k"], x["beta"], x["late_q"]))

    summary_csv = results_dir / "inner_loso_knn_global_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["k", "beta", "late_q", "mean_inner_validation_mae", "std_inner_validation_mae", "num_inner_tasks"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    selected = summary_rows[0]
    selected_json = {
        "k": int(selected["k"]),
        "beta": float(selected["beta"]),
        "late_library_quantile": float(selected["late_q"]),
        "selection_metric": args.selection_metric,
        "selection_protocol": "inner leave-one-source-run-out validation across source-domain runs",
        "selection_source": "paper_exec/train_only_param_selection/results/inner_loso_knn_global_summary.csv",
        "note": "Selected without using outer test folds; query runs are held-out source runs only.",
    }
    selected_json_path = results_dir / "selected_knn_config_inner_loso.json"
    selected_json_path.write_text(json.dumps(selected_json, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_md = project_root / "paper_exec" / "train_only_param_selection" / "manifests" / "InnerLOSO选参汇总.md"
    lines = [
        "# Inner LOSO Selection Summary",
        "",
        f"- 候选 `k`: `{k_grid}`",
        f"- 候选 `beta`: `{beta_grid}`",
        f"- 候选 `late_q`: `{q_grid}`",
        f"- 选择指标: `{args.selection_metric}`",
        "",
        "## 每个 inner task 最优",
        "",
    ]
    for row in inner_best_rows:
        lines.append(
            f"- {row['inner_id']}: k={row['selected_k']}, beta={row['selected_beta']}, "
            f"late_q={row['selected_late_q']}, metric={row['selected_metric']:.4f}"
        )
    lines.extend([
        "",
        "## 全局固定参数",
        "",
        f"- k={selected_json['k']}",
        f"- beta={selected_json['beta']}",
        f"- late_q={selected_json['late_library_quantile']}",
        f"- mean_inner_validation_mae={selected['mean_inner_validation_mae']:.4f}",
        f"- std_inner_validation_mae={selected['std_inner_validation_mae']:.4f}",
        "",
        "## 输出文件",
        "",
        f"- `{grid_csv}`",
        f"- `{stage_csv}`",
        f"- `{inner_best_csv}`",
        f"- `{summary_csv}`",
        f"- `{selected_json_path}`",
    ])
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[INNER-LOSO-SELECT][OK] summary_csv={summary_csv}")
    print(f"[INNER-LOSO-SELECT][OK] selected={selected_json}")


if __name__ == "__main__":
    main()
