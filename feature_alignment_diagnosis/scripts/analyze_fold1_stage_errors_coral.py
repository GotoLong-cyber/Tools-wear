#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from exp.exp_forecast import Exp_Forecast
from feature_alignment_diagnosis.scripts.analyze_fold1_stage_errors import (
    compute_stage_metrics,
    plot_stage_curve,
    plot_stage_residual,
    write_summary,
)


def build_args(project_root: Path, results_subdir: str) -> SimpleNamespace:
    runtime_dir = project_root / "dataset/passlevel_tree_select/runtime_rms7_plus_feat4_plus_se1_a2_coral_fold1_gpu0"
    checkpoint_setting = (
        "forecast_PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2CORAL_dual_seed2026_e1000_bt96_gpu0_"
        "timer_xl_PHM_MergedMultivariateNpy_sl96_it96_ot16_lr0.0001_bt96_wd0_el8_dm1024_dff2048_nh8_cosTrue_test_0"
    )
    return SimpleNamespace(
        task_name="forecast",
        is_training=0,
        model_id="PHM_c1c4_to_c6_rms7_plus_feat4_plus_se1_A2CORAL_dual_seed2026_e1000_bt96_gpu0",
        model="timer_xl",
        seed=2026,
        data="PHM_MergedMultivariateNpy",
        root_path=str(runtime_dir),
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
        test_dir=checkpoint_setting,
        test_file_name="checkpoint.pth",
        output_attention=False,
        visualize=False,
        flash_attention=False,
        target_only=True,
        target_idx=-1,
        adaptation=True,
        pretrain_model_path=str(project_root / "checkpoint/checkpoint.pth"),
        subset_rand_ratio=1,
        freeze_backbone=True,
        lam_mono=0.01,
        lam_smooth=0.00001,
        lam_coral=0.001,
        unfreeze_last_n=1,
        keep_features_path="",
        train_runs="c1,c4",
        test_runs="c6",
        split_ratio=0.8,
        time_gap=0,
        wear_agg="max",
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
        train_epochs=1000,
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
        devices="0,1,2,3",
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
        domain_des="PHM tool wear forecasting.",
        llm_model="LLAMA",
        llm_layers=6,
        local_rank=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("feature_alignment_diagnosis/outputs/20260324_fold1_stage_error_coral"),
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default="20260324_A2PlusSE1_CORAL_fold1",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_args = build_args(project_root, args.results_subdir)
    setting = exp_args.test_dir
    exp = Exp_Forecast(exp_args)
    result = exp.test(setting, test=1)

    pred_full = result["pred_full"].astype(np.float32)
    true_raw_full = result["true_raw_full_um"].astype(np.float32)
    seq_len = int(exp_args.test_seq_len)

    df, stage_full = compute_stage_metrics(pred_full, true_raw_full, seq_len)
    df.to_csv(out_dir / "fold1_stage_error_metrics.csv", index=False)
    np.savez(
        out_dir / "fold1_stage_error_arrays.npz",
        pred_full=pred_full,
        true_raw_full_um=true_raw_full,
        stage_id=stage_full,
    )

    # plot_stage_curve(...Current-best...) / plot_stage_residual(...Current-best...)
    plot_stage_curve(out_dir / "fold1_stage_curve_colored.png", true_raw_full, pred_full, stage_full, seq_len)
    plot_stage_residual(out_dir / "fold1_stage_residuals.png", true_raw_full, pred_full, stage_full, seq_len)
    write_summary(out_dir / "fold1_stage_error_summary.md", df)

    manifest = {
        "checkpoint_setting": exp_args.test_dir,
        "results_subdir": args.results_subdir,
        "seq_len": seq_len,
        "metrics_csv": str(out_dir / "fold1_stage_error_metrics.csv"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] coral stage error analysis written to: {out_dir}")


if __name__ == "__main__":
    main()
