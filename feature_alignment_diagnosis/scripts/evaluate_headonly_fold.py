#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
    save_curve_plot,
    stage_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--runtime_root", type=Path, required=True)
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--results_subdir", type=str, required=True)
    parser.add_argument("--train_runs", type=str, required=True)
    parser.add_argument("--test_runs", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--wear_agg", type=str, default="mean", choices=["max", "mean"])
    parser.add_argument("--split_ratio", type=float, default=1.0)
    parser.add_argument("--time_gap", type=int, default=0)
    parser.add_argument("--n_vars", type=int, required=True)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_root = (project_root / args.runtime_root).resolve()
    checkpoint_path = (project_root / args.checkpoint_path).resolve()
    results_root = project_root / "results" / args.results_subdir
    results_root.mkdir(parents=True, exist_ok=True)

    cfg = build_args(
        project_root,
        runtime_root,
        checkpoint_path,
        args.results_subdir,
        wear_agg=args.wear_agg,
        split_ratio=args.split_ratio,
        time_gap=args.time_gap,
        n_vars=args.n_vars,
    )
    cfg.train_runs = str(args.train_runs)
    cfg.test_runs = str(args.test_runs)

    exp = Exp_Forecast(cfg)
    load_checkpoint(exp, checkpoint_path)

    test_data, _ = data_provider(cfg, "test")
    test_loader = make_eval_loader(test_data, cfg.batch_size)

    _, head_pred = extract_repr_and_head_preds(exp, test_data, test_loader)
    horizon = int(cfg.test_pred_len)
    seq_len = int(cfg.test_seq_len)
    test_targets = extract_raw_target_sequences(test_data, horizon=horizon, seq_len=seq_len)

    head_pred_full, _, true_raw_full = reconstruct_full_curve(head_pred, test_targets, test_data, seq_len=seq_len)
    overall = {"mode": "head-only", **metrics_from_full_curve(head_pred_full, true_raw_full, seq_len=seq_len)}
    stage_info = build_stage_info(true_raw_full, seq_len=seq_len)
    stage_rows = [{"mode": "head-only", **row} for row in stage_metrics(head_pred_full, true_raw_full, stage_info)]

    overall_path = results_root / f"headonly_{args.tag}_overall_metrics.csv"
    stage_path = results_root / f"headonly_{args.tag}_stage_metrics.csv"
    with overall_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["mode", "mse_full_raw", "rmse_full_raw", "mae_full_raw", "valid_points"])
        wr.writeheader()
        wr.writerow(overall)
    with stage_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "stage",
                "num_points",
                "wear_min_um",
                "wear_max_um",
                "mae_um",
                "rmse_um",
                "mean_residual_um",
                "underest_ratio",
            ],
        )
        wr.writeheader()
        wr.writerows(stage_rows)

    save_curve_plot(
        results_root / f"wear_full_curve_headonly_compare_{args.tag}.png",
        true_raw_full,
        {"head-only": head_pred_full},
        seq_len,
    )
    print(f"[HEADONLY][OK] results_root={results_root}")
    print(f"[HEADONLY][OK] mae_full_raw={overall['mae_full_raw']:.4f}")


if __name__ == "__main__":
    main()
