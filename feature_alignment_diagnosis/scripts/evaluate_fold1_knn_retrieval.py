#!/usr/bin/env python3
# Hyperparameters are fixed a priori (k=5, beta=0.5, late_q=0.80).
# No test-set tuning is performed. All parameters loaded from knn_config.json.
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


def build_args(
    project_root: Path,
    runtime_root: Path,
    checkpoint_path: Path,
    results_subdir: str,
    wear_agg: str = "max",
) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="forecast",
        is_training=0,
        model_id="PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2_knn_eval",
        model="timer_xl",
        seed=2026,
        data="PHM_MergedMultivariateNpy",
        root_path=str(runtime_root),
        data_path=".",
        checkpoints=str(project_root / "checkpoints"),
        results_subdir=results_subdir,
        test_flag="T",
        seq_len=96,
        input_token_len=96,
        output_token_len=16,
        test_seq_len=96,
        test_pred_len=16,
        dropout=0.1,
        e_layers=8,
        d_model=1024,
        n_heads=8,
        d_ff=2048,
        activation="relu",
        covariate=True,
        node_num=100,
        node_list=[23, 37, 40],
        use_norm=False,
        nonautoregressive=True,
        test_dir="",
        test_file_name=checkpoint_path.name,
        output_attention=False,
        visualize=False,
        flash_attention=False,
        target_only=True,
        target_idx=-1,
        adaptation=True,
        pretrain_model_path=str(project_root / "checkpoint" / "checkpoint.pth"),
        subset_rand_ratio=1.0,
        freeze_backbone=True,
        lam_mono=0.01,
        lam_smooth=0.00001,
        lam_coral=0.0,
        lam_mmd=0.0,
        lam_asym=0.0,
        asym_wear_threshold_um=150.0,
        asym_wear_quantile=0.66,
        asym_alpha=2.0,
        unfreeze_last_n=1,
        keep_features_path="",
        train_runs="c1,c4",
        test_runs="c6",
        split_ratio=0.8,
        time_gap=0,
        wear_agg=str(wear_agg),
        mask_future_features_in_y=False,
        enable_dual_loader=1,
        train_stride_candidates="1,2",
        train_stride_quantiles="0.5",
        train_stride_use_monotonic_wear=1,
        train_stride_policy="random",
        train_stride_random_seed=2026,
        train_window_weight_policy="none",
        train_window_weight_quantile=0.5,
        train_window_weight_seed=2026,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=96,
        patience=100,
        learning_rate=1e-4,
        des="test",
        loss="MSE",
        lradj="type1",
        cosine=True,
        tmax=10,
        weight_decay=0.0,
        valid_last=False,
        last_token=True,
        gpu=0,
        ddp=False,
        dp=False,
        devices="0",
        device_ids=[0],
        gpt_layers=6,
        patch_size=16,
        kernel_size=25,
        stride=8,
        n_vars=10,
        factor=2,
        mode="mix_channel",
        AP_levels=0,
        use_decoder=True,
        d_mode="common_channel",
        layers=8,
        hidden_dim=16,
        ts_vocab_size=1000,
        domain_des="PHM2010 milling wear forecasting.",
        llm_model="LLAMA",
        llm_layers=6,
    )


def load_checkpoint(exp: Exp_Forecast, checkpoint_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    base_model = exp.model.module if hasattr(exp.model, "module") else exp.model
    model_state = base_model.state_dict()
    checkpoint = exp._align_state_for_model(checkpoint, model_state)
    base_model.load_state_dict(checkpoint, strict=False)
    exp.model.eval()


def make_eval_loader(dataset, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )


def wear_scale_params(dataset, target_idx: int) -> tuple[float, float]:
    csc = int(dataset.scaler.mean_.shape[0])
    tidx = target_idx if target_idx >= 0 else (csc + target_idx)
    return float(dataset.scaler.mean_[tidx]), float(dataset.scaler.scale_[tidx])


def extract_repr_and_head_preds(exp: Exp_Forecast, dataset, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    device = exp.device
    mean_t, scale_t = wear_scale_params(dataset, exp.args.target_idx)

    repr_list = []
    pred_list = []
    exp.model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            model_out = exp.model(batch_x, batch_x_mark, batch_y_mark, return_features=True)
            if not (isinstance(model_out, tuple) and len(model_out) == 2 and isinstance(model_out[1], dict)):
                raise RuntimeError("Model did not return shared features for KNN evaluation.")
            outputs, aux = model_out
            hidden_repr = aux.get("shared_repr", None)
            if hidden_repr is None:
                raise RuntimeError("shared_repr is missing in model aux outputs.")

            if outputs.ndim == 2:
                pred_w = outputs[:, -exp.args.test_pred_len :]
            else:
                out_last = outputs[:, -exp.args.test_pred_len :, :]
                cout = out_last.shape[-1]
                widx = int(exp.args.target_idx)
                widx = widx if widx >= 0 else (cout + widx)
                pred_w = out_last[:, :, 0] if cout == 1 else out_last[:, :, widx]

            repr_list.append(hidden_repr.detach().cpu().numpy().astype(np.float32))
            pred_list.append((pred_w.detach().cpu().numpy().astype(np.float32) * scale_t) + mean_t)

    return np.concatenate(repr_list, axis=0), np.concatenate(pred_list, axis=0)


def extract_raw_target_sequences(dataset, horizon: int, seq_len: int) -> np.ndarray:
    seqs = []
    for item in dataset.index_map:
        if len(item) == 2:
            fn, s_begin = item
            stride = 1
        else:
            fn, s_begin, stride = item
        run = Path(fn).stem
        base_t = int(s_begin) + (seq_len - 1) * int(stride) + 1
        raw_y = np.asarray(dataset.raw_wear_um[run], dtype=np.float32)
        seq = raw_y[base_t: base_t + horizon]
        if len(seq) != horizon:
            raise RuntimeError(f"Unexpected target length for {run} start={s_begin}: got {len(seq)} expected {horizon}")
        seqs.append(seq.astype(np.float32))
    return np.stack(seqs, axis=0)


def reconstruct_full_curve(preds_inv: np.ndarray, trues_inv: np.ndarray, dataset, seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    index_map = dataset.index_map
    fn0 = index_map[0][0]
    run0 = Path(fn0).stem
    length = int(dataset.raw[fn0].shape[0])
    horizon = int(preds_inv.shape[1])

    pred_bucket = [[] for _ in range(length)]
    true_bucket = [[] for _ in range(length)]
    for ii, item in enumerate(index_map):
        if len(item) == 2:
            _, s_begin = item
            stride = 1
        else:
            _, s_begin, stride = item
        base_t = int(s_begin) + (seq_len - 1) * int(stride) + 1
        for k in range(horizon):
            t = base_t + k
            if 0 <= t < length:
                pred_bucket[t].append(float(preds_inv[ii, k]))
                true_bucket[t].append(float(trues_inv[ii, k]))

    pred_full = np.full((length,), np.nan, dtype=np.float32)
    true_full = np.full((length,), np.nan, dtype=np.float32)
    for t in range(length):
        if pred_bucket[t]:
            pred_full[t] = float(np.mean(pred_bucket[t]))
        if true_bucket[t]:
            true_full[t] = float(np.mean(true_bucket[t]))

    true_raw_full = np.asarray(dataset.raw_wear_um[run0], dtype=np.float32)
    return pred_full, true_full, true_raw_full


def metrics_from_full_curve(pred_full: np.ndarray, true_raw_full: np.ndarray, seq_len: int) -> dict:
    valid = np.isfinite(pred_full) & np.isfinite(true_raw_full)
    valid = valid & (np.arange(len(pred_full)) >= seq_len)
    err = pred_full[valid] - true_raw_full[valid]
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    return {
        "valid_points": int(valid.sum()),
        "mse_full_raw": mse,
        "rmse_full_raw": rmse,
        "mae_full_raw": mae,
    }


def build_stage_info(true_raw_full: np.ndarray, seq_len: int) -> list[dict]:
    valid = np.isfinite(true_raw_full) & (np.arange(len(true_raw_full)) >= seq_len)
    vals = true_raw_full[valid]
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    stages = [
        ("early", -np.inf, q1),
        ("mid", q1, q2),
        ("late", q2, np.inf),
    ]

    infos = []
    for name, lo, hi in stages:
        if np.isneginf(lo):
            mask = valid & (true_raw_full <= hi)
        elif np.isposinf(hi):
            mask = valid & (true_raw_full > lo)
        else:
            mask = valid & (true_raw_full > lo) & (true_raw_full <= hi)
        wear_vals = true_raw_full[mask]
        infos.append(
            {
                "stage": name,
                "mask": mask,
                "wear_min_um": float(np.min(wear_vals)),
                "wear_max_um": float(np.max(wear_vals)),
                "num_points": int(mask.sum()),
            }
        )
    return infos


def stage_metrics(pred_full: np.ndarray, true_raw_full: np.ndarray, stage_infos: list[dict]) -> list[dict]:
    rows = []
    for info in stage_infos:
        mask = info["mask"] & np.isfinite(pred_full)
        err = pred_full[mask] - true_raw_full[mask]
        rows.append(
            {
                "stage": info["stage"],
                "num_points": int(mask.sum()),
                "wear_min_um": info["wear_min_um"],
                "wear_max_um": info["wear_max_um"],
                "mae_um": float(np.mean(np.abs(err))),
                "rmse_um": float(np.sqrt(np.mean(err ** 2))),
                "mean_residual_um": float(np.mean(err)),
                "underest_ratio": float(np.mean(pred_full[mask] < true_raw_full[mask])),
            }
        )
    return rows


def cosine_knn_predict(train_repr: np.ndarray, train_targets: np.ndarray, test_repr: np.ndarray, k: int) -> np.ndarray:
    eps = 1e-8
    train_norm = train_repr / np.linalg.norm(train_repr, axis=1, keepdims=True).clip(min=eps)
    test_norm = test_repr / np.linalg.norm(test_repr, axis=1, keepdims=True).clip(min=eps)
    sims = test_norm @ train_norm.T
    dists = 1.0 - sims
    k = int(min(k, train_repr.shape[0]))
    topk_idx = np.argpartition(dists, kth=k - 1, axis=1)[:, :k]

    preds = []
    for i in range(test_repr.shape[0]):
        idx = topk_idx[i]
        local_d = dists[i, idx]
        local_w = 1.0 / np.clip(local_d, 1e-6, None)
        local_w = local_w / np.sum(local_w)
        pred = np.sum(train_targets[idx] * local_w[:, None], axis=0)
        preds.append(pred.astype(np.float32))
    return np.stack(preds, axis=0)


def save_curve_plot(save_path: Path, true_raw_full: np.ndarray, curves: dict[str, np.ndarray], seq_len: int) -> None:
    plt.figure(figsize=(11, 4.5))
    plt.plot(true_raw_full, label="true(raw wear, um)", linewidth=2.0, color="black")
    for name, values in curves.items():
        plt.plot(values, label=name, linewidth=1.5)
    plt.axvline(seq_len, linestyle="--", color="gray", label=f"forecast starts @ t={seq_len}")
    plt.xlabel("walk index")
    plt.ylabel("wear (um)")
    plt.title("fold1 retrieval comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def write_summary_md(md_path: Path, overall_rows: list[dict], stage_rows: list[dict], best_blend_name: str) -> None:
    lines = [
        "# fold1 KNN Retrieval Summary",
        "",
        f"- best_blend_by_fullcurve_raw_mae: `{best_blend_name}`",
        "",
        "## Overall",
        "",
        "| mode | mse_full_raw | rmse_full_raw | mae_full_raw | valid_points |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['mode']} | {row['mse_full_raw']:.4f} | {row['rmse_full_raw']:.4f} | {row['mae_full_raw']:.4f} | {row['valid_points']} |"
        )

    lines.extend([
        "",
        "## Stage Metrics",
        "",
        "| mode | stage | num_points | wear_range_um | MAE | RMSE | mean_residual | underest_ratio |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for row in stage_rows:
        lines.append(
            f"| {row['mode']} | {row['stage']} | {row['num_points']} | {row['wear_min_um']:.4f}-{row['wear_max_um']:.4f} | "
            f"{row['mae_um']:.4f} | {row['rmse_um']:.4f} | {row['mean_residual_um']:.4f} | {row['underest_ratio']:.4f} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--runtime_root", type=Path, default=Path("dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_fold1_gpu0"))
    parser.add_argument("--checkpoint_path", type=Path, default=Path("checkpoints/forecast_PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2_dual_seed2026_e1000_bt96_gpu0_timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0/checkpoint.pth"))
    parser.add_argument("--results_subdir", type=str, default="20260324_KNNFold1")
    parser.add_argument("--distance", type=str, default="cosine")
    parser.add_argument("--ks", type=str, default="", help="deprecated, ignored; use knn_config.json")
    parser.add_argument("--betas", type=str, default="", help="deprecated, ignored; use knn_config.json")
    parser.add_argument("--knn_config", type=Path, default=DEFAULT_KNN_CONFIG)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    runtime_root = (project_root / args.runtime_root).resolve()
    checkpoint_path = (project_root / args.checkpoint_path).resolve()
    results_root = project_root / "results" / args.results_subdir
    results_root.mkdir(parents=True, exist_ok=True)
    config_path = args.knn_config if args.knn_config.is_absolute() else (project_root / args.knn_config)
    fixed_k, fixed_beta, fixed_late_q, knn_cfg = load_fixed_knn_config(config_path.resolve())
    if str(args.ks).strip() or str(args.betas).strip():
        print(f"[KNN][INFO] --ks/--betas are ignored; using fixed config from {config_path.resolve()}")
    print(f"[KNN][INFO] fixed k={fixed_k}, beta={fixed_beta}, late_q={fixed_late_q}")

    cfg = build_args(project_root, runtime_root, checkpoint_path, args.results_subdir)
    exp = Exp_Forecast(cfg)
    load_checkpoint(exp, checkpoint_path)

    # Build a physical-history library: only train split, only stride=1 windows.
    args_train = copy.deepcopy(cfg)
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
        raise NotImplementedError("Only cosine distance is supported in the first KNN validation.")

    knn_pred = cosine_knn_predict(train_repr, train_targets, test_repr, k=fixed_k)
    mode_knn = f"knn-only@k{fixed_k}"
    knn_pred_full, _, _ = reconstruct_full_curve(knn_pred, test_targets, test_data, seq_len=seq_len)
    knn_metrics = metrics_from_full_curve(knn_pred_full, true_raw_full, seq_len=seq_len)
    overall_rows.append({"mode": mode_knn, **knn_metrics})
    for row in stage_metrics(knn_pred_full, true_raw_full, stage_info):
        stage_rows.append({"mode": mode_knn, **row})
    pred_curves[mode_knn] = knn_pred_full

    blend_pred = (1.0 - fixed_beta) * head_pred + fixed_beta * knn_pred
    mode_blend = f"blend@k{fixed_k}_b{str(fixed_beta).replace('.', '')}"
    blend_pred_full, _, _ = reconstruct_full_curve(blend_pred, test_targets, test_data, seq_len=seq_len)
    blend_metrics = metrics_from_full_curve(blend_pred_full, true_raw_full, seq_len=seq_len)
    overall_rows.append({"mode": mode_blend, **blend_metrics})
    for row in stage_metrics(blend_pred_full, true_raw_full, stage_info):
        stage_rows.append({"mode": mode_blend, **row})
    pred_curves[mode_blend] = blend_pred_full

    overall_rows.sort(key=lambda x: x["mae_full_raw"])
    best_curve_keys = ["head-only", mode_knn, mode_blend]
    save_curve_plot(results_root / "wear_full_curve_knn_compare.png", true_raw_full, {k: pred_curves[k] for k in best_curve_keys}, seq_len)

    overall_path = results_root / "knn_fold1_overall_metrics.csv"
    stage_path = results_root / "knn_fold1_stage_metrics.csv"
    summary_path = results_root / "knn_fold1_summary.md"
    manifest_path = results_root / "knn_fold1_manifest.json"

    import csv
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

    write_summary_md(summary_path, overall_rows, stage_rows, mode_blend)
    manifest = {
        "runtime_root": str(runtime_root),
        "checkpoint_path": str(checkpoint_path),
        "distance": args.distance,
        "k": fixed_k,
        "beta": fixed_beta,
        "late_library_quantile": fixed_late_q,
        "knn_config_path": str(config_path.resolve()),
        "knn_config": knn_cfg,
        "selected_blend_name": mode_blend,
        "selected_blend_mae_full_raw": blend_metrics["mae_full_raw"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[KNN][OK] results_root={results_root}")
    print(f"[KNN][OK] selected_blend={mode_blend} mae_full_raw={blend_metrics['mae_full_raw']:.4f}")


if __name__ == "__main__":
    main()
