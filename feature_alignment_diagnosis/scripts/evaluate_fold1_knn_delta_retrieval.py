#!/usr/bin/env python3
# Hyperparameters are fixed a priori (k=5, beta=0.5, late_q=0.80).
# No test-set tuning is performed. All parameters loaded from knn_config.json.
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feature_alignment_diagnosis.scripts.evaluate_fold1_knn_retrieval import (
    build_args,
    build_stage_info,
    cosine_knn_predict,
    extract_raw_target_sequences,
    extract_repr_and_head_preds,
    load_checkpoint,
    make_eval_loader,
    metrics_from_full_curve,
    reconstruct_full_curve,
    save_curve_plot,
    stage_metrics,
    write_summary_md,
)
from data_provider.data_factory import data_provider
from exp.exp_forecast import Exp_Forecast

# Fixed defaults (also mirrored in knn_config.json)
FIXED_K = 5
FIXED_BETA = 0.5
FIXED_LATE_LIBRARY_QUANTILE = 0.80

DEFAULT_KNN_CONFIG = Path(__file__).resolve().with_name("knn_config.json")


def load_fixed_knn_config(config_path: Path) -> tuple[int, float, float, dict]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    k = int(cfg.get("k", FIXED_K))
    beta = float(cfg.get("beta", FIXED_BETA))
    late_q = float(cfg.get("late_library_quantile", FIXED_LATE_LIBRARY_QUANTILE))
    if k <= 0:
        raise ValueError(f"invalid k in {config_path}: {k}")
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"invalid beta in {config_path}: {beta}")
    if not (0.0 <= late_q <= 1.0):
        raise ValueError(f"invalid late_library_quantile in {config_path}: {late_q}")
    return k, beta, late_q, cfg


def extract_current_last_wear(dataset, seq_len: int) -> np.ndarray:
    vals = []
    for item in dataset.index_map:
        if len(item) == 2:
            fn, s_begin = item
            stride = 1
        else:
            fn, s_begin, stride = item
        run = Path(fn).stem
        last_t = int(s_begin) + (seq_len - 1) * int(stride)
        raw_y = np.asarray(dataset.raw_wear_um[run], dtype=np.float32)
        vals.append(float(raw_y[last_t]))
    return np.asarray(vals, dtype=np.float32)


def select_library_mask(current_last_um: np.ndarray, threshold_um: float, quantile: float) -> tuple[np.ndarray, float]:
    if threshold_um > 0:
        thr = float(threshold_um)
    else:
        thr = float(np.quantile(current_last_um, quantile))
    mask = current_last_um >= thr
    return mask, thr


def cosine_knn_predict_with_meta(
    train_repr: np.ndarray, train_targets: np.ndarray, test_repr: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eps = 1e-8
    train_norm = train_repr / np.linalg.norm(train_repr, axis=1, keepdims=True).clip(min=eps)
    test_norm = test_repr / np.linalg.norm(test_repr, axis=1, keepdims=True).clip(min=eps)
    sims = test_norm @ train_norm.T
    dists = 1.0 - sims
    k = int(min(k, train_repr.shape[0]))
    topk_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    preds = []
    min_dists = []
    mean_topk_dists = []
    for i in range(test_repr.shape[0]):
        idx = topk_idx[i]
        local_d = dists[i, idx]
        local_w = 1.0 / np.clip(local_d, 1e-6, None)
        local_w = local_w / np.sum(local_w)
        pred = np.sum(train_targets[idx] * local_w[:, None], axis=0)
        preds.append(pred.astype(np.float32))
        min_dists.append(float(np.min(local_d)))
        mean_topk_dists.append(float(np.mean(local_d)))
    return (
        np.stack(preds, axis=0),
        np.asarray(min_dists, dtype=np.float32),
        np.asarray(mean_topk_dists, dtype=np.float32),
    )


def distance_to_dynamic_beta(
    distances: np.ndarray,
    qlo: float,
    qhi: float,
    beta_min: float,
    beta_max: float,
) -> np.ndarray:
    # WARNING:
    # This maps beta using batch-level distance quantiles over test windows.
    # It is non-causal for strict online deployment.
    # Formal paper results should use gate_mode='none'.
    lo = float(np.quantile(distances, qlo))
    hi = float(np.quantile(distances, qhi))
    if hi <= lo + 1e-8:
        return np.full_like(distances, fill_value=float(beta_max), dtype=np.float32)
    scores = (hi - distances) / (hi - lo)
    scores = np.clip(scores, 0.0, 1.0)
    beta = float(beta_min) + scores * (float(beta_max) - float(beta_min))
    return beta.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--runtime_root", type=Path, default=Path("dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_fold1_gpu0"))
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/forecast_PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu0_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"))
    parser.add_argument("--results_subdir", type=str, default="20260324_KNNDeltaFold1")
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--k", type=int, default=5, help="deprecated, ignored; use knn_config.json")
    parser.add_argument("--betas", type=str, default="", help="deprecated, ignored; use knn_config.json")
    parser.add_argument("--library_wear_threshold_um", type=float, default=150.0)
    parser.add_argument("--library_wear_quantile", type=float, default=0.80, help="deprecated, ignored; use knn_config.json")
    parser.add_argument("--blend_mode", type=str, default="delta_residual", choices=["delta_add", "delta_residual"])
    parser.add_argument("--gate_mode", type=str, default="none", choices=["none", "distance_linear"])
    parser.add_argument("--gate_stat", type=str, default="min", choices=["min", "mean_topk"])
    parser.add_argument("--gate_qlo", type=float, default=0.10)
    parser.add_argument("--gate_qhi", type=float, default=0.90)
    parser.add_argument("--gate_beta_min", type=float, default=0.0)
    parser.add_argument("--gate_beta_max", type=float, default=1.0)
    parser.add_argument("--train_runs", type=str, default="c1,c4")
    parser.add_argument("--test_runs", type=str, default="c6")
    parser.add_argument("--tag", type=str, default="fold1")
    parser.add_argument("--knn_config", type=Path, default=DEFAULT_KNN_CONFIG)
    parser.add_argument("--wear_agg", type=str, default="max", choices=["max", "mean"])
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_root = (project_root / args.runtime_root).resolve()
    checkpoint_path = (project_root / args.checkpoint_path).resolve()
    results_root = project_root / "results" / args.results_subdir
    results_root.mkdir(parents=True, exist_ok=True)
    config_path = args.knn_config if args.knn_config.is_absolute() else (project_root / args.knn_config)
    fixed_k, fixed_beta, fixed_late_q, knn_cfg = load_fixed_knn_config(config_path.resolve())
    if str(args.betas).strip():
        print(f"[KNN-DELTA][INFO] --betas is ignored; using fixed config from {config_path.resolve()}")
    print(f"[KNN-DELTA][INFO] fixed k={fixed_k}, beta={fixed_beta}, late_q={fixed_late_q}")

    cfg = build_args(project_root, runtime_root, checkpoint_path, args.results_subdir, wear_agg=args.wear_agg)
    cfg.train_runs = str(args.train_runs)
    cfg.test_runs = str(args.test_runs)
    exp = Exp_Forecast(cfg)
    load_checkpoint(exp, checkpoint_path)

    args_train = cfg.__class__(**vars(cfg))
    args_train.train_stride_candidates = "1"
    args_train.train_stride_quantiles = ""
    train_data, _ = data_provider(args_train, "train")
    test_data, _ = data_provider(cfg, "test")
    train_loader = make_eval_loader(train_data, cfg.batch_size)
    test_loader = make_eval_loader(test_data, cfg.batch_size)

    train_repr, _ = extract_repr_and_head_preds(exp, train_data, train_loader)
    test_repr, head_pred = extract_repr_and_head_preds(exp, test_data, test_loader)

    horizon = int(cfg.test_pred_len)
    seq_len = int(cfg.test_seq_len)
    train_targets = extract_raw_target_sequences(train_data, horizon=horizon, seq_len=seq_len)
    test_targets = extract_raw_target_sequences(test_data, horizon=horizon, seq_len=seq_len)
    train_current_last = extract_current_last_wear(train_data, seq_len=seq_len)
    test_current_last = extract_current_last_wear(test_data, seq_len=seq_len)

    train_delta_targets = train_targets - train_current_last[:, None]

    # lib_mask, lib_thr = select_library_mask(
    #     train_current_last,
    #     threshold_um=float(args.library_wear_threshold_um),
    #     quantile=float(args.library_wear_quantile),
    # )
    lib_mask, lib_thr = select_library_mask(
        train_current_last,
        threshold_um=float(args.library_wear_threshold_um),
        quantile=float(fixed_late_q),
    )
    if int(lib_mask.sum()) == 0:
        # Fallback: if the fixed threshold is above all train-window current wear,
        # rebuild a late-ish sub-library by quantile rather than silently using an empty library.
        lib_mask, lib_thr = select_library_mask(
            train_current_last,
            threshold_um=-1.0,
            quantile=float(fixed_late_q),
        )
    lib_repr = train_repr[lib_mask]
    lib_delta_targets = train_delta_targets[lib_mask]
    if int(lib_repr.shape[0]) == 0:
        raise RuntimeError("Delta-retrieval library is empty after threshold/quantile selection.")

    overall_rows = []
    stage_rows = []
    pred_curves = {}

    head_pred_full, _, true_raw_full = reconstruct_full_curve(head_pred, test_targets, test_data, seq_len=seq_len)
    head_metrics = metrics_from_full_curve(head_pred_full, true_raw_full, seq_len=seq_len)
    overall_rows.append({"mode": "head-only", **head_metrics})
    stage_info = build_stage_info(true_raw_full, seq_len=seq_len)
    for row in stage_metrics(head_pred_full, true_raw_full, stage_info):
        stage_rows.append({"mode": "head-only", **row})
    pred_curves["head-only"] = head_pred_full

    if args.distance != "cosine":
        raise NotImplementedError("Only cosine distance is supported in Retrieval V2 first round.")

    delta_knn, min_dists, mean_topk_dists = cosine_knn_predict_with_meta(
        lib_repr, lib_delta_targets, test_repr, k=int(fixed_k)
    )
    knn_abs = test_current_last[:, None] + delta_knn

    knn_mode = f"delta-knn-only@k{int(fixed_k)}"
    knn_pred_full, _, _ = reconstruct_full_curve(knn_abs, test_targets, test_data, seq_len=seq_len)
    knn_metrics = metrics_from_full_curve(knn_pred_full, true_raw_full, seq_len=seq_len)
    overall_rows.append({"mode": knn_mode, **knn_metrics})
    for row in stage_metrics(knn_pred_full, true_raw_full, stage_info):
        stage_rows.append({"mode": knn_mode, **row})
    pred_curves[knn_mode] = knn_pred_full

    if args.blend_mode == "delta_add":
        blend_pred = head_pred + fixed_beta * delta_knn
    else:
        delta_head = head_pred - test_current_last[:, None]
        blend_pred = head_pred + fixed_beta * (delta_knn - delta_head)
    mode = f"delta-blend@k{int(fixed_k)}_b{str(fixed_beta).replace('.', '')}"
    pred_full, _, _ = reconstruct_full_curve(blend_pred, test_targets, test_data, seq_len=seq_len)
    cur_metrics = metrics_from_full_curve(pred_full, true_raw_full, seq_len=seq_len)
    overall_rows.append({"mode": mode, **cur_metrics})
    for row in stage_metrics(pred_full, true_raw_full, stage_info):
        stage_rows.append({"mode": mode, **row})
    pred_curves[mode] = pred_full

    if args.gate_mode == "distance_linear":
        # WARNING:
        # gate_mode != "none" uses statistics from the whole test-window batch.
        # Keep this branch for exploratory analysis only; do not use it for formal claims.
        warnings.warn(
            "gate_mode != 'none' uses batch-level test statistics and is non-causal; "
            "do not use for paper results.",
            UserWarning,
            stacklevel=2,
        )
        gate_dist = min_dists if str(args.gate_stat) == "min" else mean_topk_dists
        beta_dyn = distance_to_dynamic_beta(
            gate_dist,
            qlo=float(args.gate_qlo),
            qhi=float(args.gate_qhi),
            beta_min=float(args.gate_beta_min),
            beta_max=float(args.gate_beta_max),
        )
        # gate_blend = head_pred + beta_dyn[:, None] * delta_knn
        if args.blend_mode == "delta_add":
            gate_blend = head_pred + beta_dyn[:, None] * delta_knn
            gate_mode_name = f"delta-gated@k{int(fixed_k)}_add"
        else:
            delta_head = head_pred - test_current_last[:, None]
            gate_blend = head_pred + beta_dyn[:, None] * (delta_knn - delta_head)
            gate_mode_name = f"delta-gated@k{int(fixed_k)}_res"
        pred_full, _, _ = reconstruct_full_curve(gate_blend, test_targets, test_data, seq_len=seq_len)
        gate_metrics = metrics_from_full_curve(pred_full, true_raw_full, seq_len=seq_len)
        overall_rows.append({"mode": gate_mode_name, **gate_metrics})
        for row in stage_metrics(pred_full, true_raw_full, stage_info):
            stage_rows.append({"mode": gate_mode_name, **row})
        pred_curves[gate_mode_name] = pred_full

    overall_rows.sort(key=lambda x: x["mae_full_raw"])
    best_curve_keys = ["head-only", knn_mode, mode]
    tag = str(args.tag).strip() or "fold"
    save_curve_plot(
        results_root / f"wear_full_curve_knn_delta_compare_{tag}.png",
        true_raw_full,
        {k: pred_curves[k] for k in best_curve_keys if k in pred_curves},
        seq_len,
    )
    overall_path = results_root / f"knn_delta_{tag}_overall_metrics.csv"
    stage_path = results_root / f"knn_delta_{tag}_stage_metrics.csv"
    summary_path = results_root / f"knn_delta_{tag}_summary.md"
    manifest_path = results_root / f"knn_delta_{tag}_manifest.json"

    with overall_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["mode", "mse_full_raw", "rmse_full_raw", "mae_full_raw", "valid_points"])
        wr.writeheader()
        wr.writerows(overall_rows)

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

    write_summary_md(summary_path, overall_rows, stage_rows, mode)
    manifest = {
        "runtime_root": str(runtime_root),
        "checkpoint_path": str(checkpoint_path),
        "distance": args.distance,
        "k": int(fixed_k),
        "beta": float(fixed_beta),
        "late_library_quantile": float(fixed_late_q),
        "blend_mode": args.blend_mode,
        "gate_mode": args.gate_mode,
        "gate_stat": args.gate_stat,
        "gate_qlo": float(args.gate_qlo),
        "gate_qhi": float(args.gate_qhi),
        "gate_beta_min": float(args.gate_beta_min),
        "gate_beta_max": float(args.gate_beta_max),
        "library_size": int(lib_mask.sum()),
        "library_wear_threshold_um": float(lib_thr),
        "knn_config_path": str(config_path.resolve()),
        "knn_config": knn_cfg,
        "selected_blend_name": mode,
        "selected_blend_mae_full_raw": cur_metrics["mae_full_raw"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[KNN-DELTA][OK] results_root={results_root}")
    print(f"[KNN-DELTA][OK] library_size={int(lib_mask.sum())} threshold_um={lib_thr:.4f}")
    print(f"[KNN-DELTA][OK] selected_blend={mode} mae_full_raw={cur_metrics['mae_full_raw']:.4f}")


if __name__ == "__main__":
    main()
