#!/usr/bin/env python3
"""
Stage-A stable feature filtering for TD28 pass-level data.

Design goals:
1) No leakage: feature filtering is fitted only on current fold train runs.
2) Remove high-risk features first:
   - high distribution shift across train runs
   - unstable wear relationship across train runs
   - high redundancy among remaining features
3) Export artifacts for the next stage (no model training in this script).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


EPS = 1e-12


def parse_runs(text: str) -> List[str]:
    runs = [x.strip() for x in text.split(",") if x.strip()]
    if not runs:
        raise ValueError("run list is empty")
    return runs


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    xv = x[mask]
    yv = y[mask]
    if np.std(xv) < EPS or np.std(yv) < EPS:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])


def is_sign_consistent(corrs: List[float], sign_eps: float) -> bool:
    signs = []
    for c in corrs:
        if c > sign_eps:
            signs.append(1)
        elif c < -sign_eps:
            signs.append(-1)
        else:
            signs.append(0)

    non_zero = [s for s in signs if s != 0]
    if not non_zero:
        return True
    return all(s == non_zero[0] for s in non_zero)


def load_npz(dataset_dir: Path, run: str, suffix: str) -> Dict[str, np.ndarray]:
    fp = dataset_dir / f"{run}_{suffix}.npz"
    if not fp.exists():
        raise FileNotFoundError(f"missing npz: {fp}")

    z = np.load(fp, allow_pickle=True)
    required = {"X", "y", "feature_names"}
    if not required.issubset(set(z.files)):
        raise KeyError(f"npz keys missing in {fp}, got={z.files}")

    x = z["X"].astype(np.float64)
    y = z["y"].astype(np.float64).reshape(-1)
    f = [str(v) for v in z["feature_names"].tolist()]
    p = z["pass_idx"].astype(np.int64) if "pass_idx" in z.files else np.arange(x.shape[0], dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got {x.shape} in {fp}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y length mismatch in {fp}, {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] != len(f):
        raise ValueError(f"X/features mismatch in {fp}, {x.shape[1]} vs {len(f)}")

    return {"X": x, "y": y, "feature_names": np.array(f, dtype=object), "pass_idx": p}


def calc_feature_metrics(
    train_data: Dict[str, Dict[str, np.ndarray]],
    feature_names: List[str],
    shift_sigma_thr: float,
    corr_std_thr: float,
    corr_sign_eps: float,
    score_shift_weight: float,
) -> List[dict]:
    merged_x = np.concatenate([train_data[r]["X"] for r in train_data], axis=0)

    metrics = []
    for j, feat in enumerate(feature_names):
        corrs = []
        means = []
        for run in train_data:
            xj = train_data[run]["X"][:, j]
            y = train_data[run]["y"]
            corrs.append(safe_pearson(xj, y))
            means.append(float(np.nanmean(xj)))

        xj_all = merged_x[:, j]
        std_train = float(np.nanstd(xj_all))
        shift_sigma = float((max(means) - min(means)) / max(std_train, EPS))
        corr_abs_mean = float(np.mean(np.abs(corrs)))
        corr_std = float(np.std(corrs))
        sign_consistent = is_sign_consistent(corrs, corr_sign_eps)

        flag_high_shift = shift_sigma > shift_sigma_thr
        flag_unstable_corr = (not sign_consistent) or (corr_std > corr_std_thr)

        score_pre = corr_abs_mean - corr_std - score_shift_weight * shift_sigma

        row = {
            "feature_id": j,
            "feature_name": feat,
            "corr_abs_mean": corr_abs_mean,
            "corr_std": corr_std,
            "sign_consistent": sign_consistent,
            "shift_sigma": shift_sigma,
            "std_train": std_train,
            "score_pre": score_pre,
            "flag_high_shift": flag_high_shift,
            "flag_unstable_corr": flag_unstable_corr,
            "flag_redundant": False,
            "redundant_to": "",
            "drop_reason": "",
            "keep_stageA": True,
        }

        for run, c in zip(train_data.keys(), corrs):
            row[f"corr_{run}"] = float(c)
        for run, m in zip(train_data.keys(), means):
            row[f"mean_{run}"] = float(m)

        metrics.append(row)

    return metrics


def redundancy_prune(
    metrics: List[dict],
    train_data: Dict[str, Dict[str, np.ndarray]],
    corr_redundancy_thr: float,
) -> Tuple[List[int], Dict[int, int], np.ndarray, List[int]]:
    merged_x = np.concatenate([train_data[r]["X"] for r in train_data], axis=0)

    prelim_keep = [
        m["feature_id"]
        for m in metrics
        if not m["flag_high_shift"] and not m["flag_unstable_corr"]
    ]

    if len(prelim_keep) <= 1:
        return prelim_keep, {}, np.array([[]], dtype=np.float64), prelim_keep

    pos_map = {fid: p for p, fid in enumerate(prelim_keep)}
    cmat = np.corrcoef(merged_x[:, prelim_keep].T)
    abs_cmat = np.abs(cmat)

    order = sorted(
        prelim_keep,
        key=lambda i: (
            -metrics[i]["score_pre"],
            -metrics[i]["corr_abs_mean"],
            metrics[i]["corr_std"],
            metrics[i]["feature_name"],
        ),
    )

    dropped_by: Dict[int, int] = {}
    final_keep: List[int] = []

    for fid in order:
        if fid in dropped_by:
            continue
        final_keep.append(fid)
        pi = pos_map[fid]
        for other in order:
            if other == fid or other in dropped_by or other in final_keep:
                continue
            pj = pos_map[other]
            if abs_cmat[pi, pj] >= corr_redundancy_thr:
                dropped_by[other] = fid

    return final_keep, dropped_by, abs_cmat, prelim_keep


def apply_drop_reasons(metrics: List[dict], final_keep_ids: List[int], dropped_by: Dict[int, int]) -> None:
    keep_set = set(final_keep_ids)
    for m in metrics:
        reasons = []
        if m["flag_high_shift"]:
            reasons.append("high_shift")
        if m["flag_unstable_corr"]:
            reasons.append("unstable_corr")
        if m["feature_id"] in dropped_by:
            rep = dropped_by[m["feature_id"]]
            m["flag_redundant"] = True
            m["redundant_to"] = metrics[rep]["feature_name"]
            reasons.append(f"redundant_to:{metrics[rep]['feature_name']}")

        m["keep_stageA"] = m["feature_id"] in keep_set
        m["drop_reason"] = "|".join(reasons)


def write_audit_csv(metrics: List[dict], train_runs: List[str], out_csv: Path) -> None:
    dynamic_cols = []
    for run in train_runs:
        dynamic_cols.append(f"corr_{run}")
    for run in train_runs:
        dynamic_cols.append(f"mean_{run}")

    fieldnames = [
        "feature_id",
        "feature_name",
        *dynamic_cols,
        "corr_abs_mean",
        "corr_std",
        "sign_consistent",
        "shift_sigma",
        "std_train",
        "score_pre",
        "flag_high_shift",
        "flag_unstable_corr",
        "flag_redundant",
        "redundant_to",
        "drop_reason",
        "keep_stageA",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in metrics:
            w.writerow(m)


def export_filtered_npz(
    all_runs_data: Dict[str, Dict[str, np.ndarray]],
    keep_ids: List[int],
    keep_names: List[str],
    suffix_out: str,
    data_out_dir: Path,
) -> None:
    data_out_dir.mkdir(parents=True, exist_ok=True)

    for run, d in all_runs_data.items():
        x_keep = d["X"][:, keep_ids].astype(np.float32)
        y = d["y"].astype(np.float32)
        pass_idx = d["pass_idx"].astype(np.int32)

        npz_out = data_out_dir / f"{run}_{suffix_out}.npz"
        np.savez(
            npz_out,
            X=x_keep,
            y=y,
            pass_idx=pass_idx,
            feature_names=np.array(keep_names, dtype=object),
        )

        csv_out = data_out_dir / f"{run}_{suffix_out}_view.csv"
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            header = ["pass_idx", *keep_names, "wear"]
            w.writerow(header)
            for i in range(x_keep.shape[0]):
                row = [int(pass_idx[i]), *x_keep[i].tolist(), float(y[i])]
                w.writerow(row)


def save_keep_lists(metrics: List[dict], keep_names: List[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    keep_fp = out_dir / "keep_features_stageA.txt"
    drop_fp = out_dir / "drop_features_stageA.txt"

    dropped = [m for m in metrics if not m["keep_stageA"]]

    keep_fp.write_text("\n".join(keep_names) + "\n", encoding="utf-8")
    with drop_fp.open("w", encoding="utf-8") as f:
        for m in dropped:
            f.write(f"{m['feature_name']}\t{m['drop_reason']}\n")


def build_summary(metrics: List[dict], keep_names: List[str], train_runs: List[str], args: argparse.Namespace) -> dict:
    n_all = len(metrics)
    n_keep = len(keep_names)

    return {
        "train_runs": train_runs,
        "n_features_all": n_all,
        "n_features_keep": n_keep,
        "n_drop": n_all - n_keep,
        "params": {
            "shift_sigma_thr": args.shift_sigma_thr,
            "corr_std_thr": args.corr_std_thr,
            "corr_sign_eps": args.corr_sign_eps,
            "corr_redundancy_thr": args.corr_redundancy_thr,
            "score_shift_weight": args.score_shift_weight,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-A stable feature filtering for td28")
    parser.add_argument("--dataset_dir", type=Path, required=True, help="path to base td28 npz dir")
    parser.add_argument("--train_runs", type=str, required=True, help="comma-separated train runs, e.g. c1,c4")
    parser.add_argument(
        "--export_runs",
        type=str,
        default="",
        help="comma-separated runs to export filtered npz/csv; default=train_runs",
    )
    parser.add_argument("--input_suffix", type=str, default="passlevel_td28", help="input npz suffix")
    parser.add_argument("--output_suffix", type=str, default="passlevel_td28_stageA", help="output npz suffix")

    parser.add_argument("--shift_sigma_thr", type=float, default=2.0)
    parser.add_argument("--corr_std_thr", type=float, default=0.35)
    parser.add_argument("--corr_sign_eps", type=float, default=0.05)
    parser.add_argument("--corr_redundancy_thr", type=float, default=0.98)
    parser.add_argument("--score_shift_weight", type=float, default=0.10)

    parser.add_argument("--out_root", type=Path, required=True, help="output root of this round")

    args = parser.parse_args()

    train_runs = parse_runs(args.train_runs)
    export_runs = parse_runs(args.export_runs) if args.export_runs.strip() else train_runs

    # Load train runs for fitting Stage-A filters.
    train_data: Dict[str, Dict[str, np.ndarray]] = {}
    for run in train_runs:
        train_data[run] = load_npz(args.dataset_dir, run, args.input_suffix)

    feat_ref = train_data[train_runs[0]]["feature_names"].tolist()
    for run in train_runs[1:]:
        feat_cur = train_data[run]["feature_names"].tolist()
        if feat_cur != feat_ref:
            raise ValueError(f"feature_names mismatch between {train_runs[0]} and {run}")

    metrics = calc_feature_metrics(
        train_data=train_data,
        feature_names=feat_ref,
        shift_sigma_thr=args.shift_sigma_thr,
        corr_std_thr=args.corr_std_thr,
        corr_sign_eps=args.corr_sign_eps,
        score_shift_weight=args.score_shift_weight,
    )

    final_keep, dropped_by, _, _ = redundancy_prune(
        metrics=metrics,
        train_data=train_data,
        corr_redundancy_thr=args.corr_redundancy_thr,
    )

    apply_drop_reasons(metrics, final_keep, dropped_by)
    keep_names = [metrics[i]["feature_name"] for i in final_keep]

    # Save audit files.
    results_dir = args.out_root / "results"
    data_dir = args.out_root / "data"
    write_audit_csv(metrics, train_runs, results_dir / "feature_audit_stageA.csv")
    save_keep_lists(metrics, keep_names, results_dir)

    summary = build_summary(metrics, keep_names, train_runs, args)
    (results_dir / "stageA_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Export filtered feature matrices for selected runs (no fitting on non-train runs).
    all_runs_data: Dict[str, Dict[str, np.ndarray]] = {}
    for run in export_runs:
        all_runs_data[run] = load_npz(args.dataset_dir, run, args.input_suffix)
        feat_cur = all_runs_data[run]["feature_names"].tolist()
        if feat_cur != feat_ref:
            raise ValueError(f"feature_names mismatch in export run {run}")

    export_filtered_npz(
        all_runs_data=all_runs_data,
        keep_ids=final_keep,
        keep_names=keep_names,
        suffix_out=args.output_suffix,
        data_out_dir=data_dir,
    )


if __name__ == "__main__":
    main()
