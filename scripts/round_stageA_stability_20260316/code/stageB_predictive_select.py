#!/usr/bin/env python3
"""
Stage-B predictive feature selection on top of Stage-A kept features.

Strict inductive rule:
- Fit and rank with train_runs only.
- test_run can be exported with selected feature columns, but must not be used in fitting/ranking.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def parse_runs(text: str) -> List[str]:
    runs = [x.strip() for x in text.split(",") if x.strip()]
    if not runs:
        raise ValueError("run list is empty")
    return runs


def parse_k_list(text: str) -> List[int]:
    ks = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not ks:
        raise ValueError("k_list is empty")
    if min(ks) <= 0:
        raise ValueError("k must be positive")
    return sorted(set(ks))


def load_npz(dataset_dir: Path, run: str, suffix: str) -> Dict[str, np.ndarray]:
    fp = dataset_dir / f"{run}_{suffix}.npz"
    if not fp.exists():
        raise FileNotFoundError(f"missing npz: {fp}")
    z = np.load(fp, allow_pickle=True)
    need = {"X", "y", "feature_names"}
    if not need.issubset(set(z.files)):
        raise KeyError(f"missing keys in {fp}, got={z.files}")

    x = z["X"].astype(np.float64)
    y = z["y"].astype(np.float64).reshape(-1)
    f = [str(v) for v in z["feature_names"].tolist()]
    p = z["pass_idx"].astype(np.int64) if "pass_idx" in z.files else np.arange(x.shape[0], dtype=np.int64)

    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got {x.shape}")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"X/y mismatch in {fp}")
    if x.shape[1] != len(f):
        raise ValueError(f"feature count mismatch in {fp}")

    return {"X": x, "y": y, "feature_names": np.array(f, dtype=object), "pass_idx": p}


def make_cv_splits(train_runs: List[str]) -> List[Tuple[str, List[str]]]:
    """
    Return list of (val_run, train_run_list).
    For 2 runs: c1->c4 and c4->c1
    For >2 runs: leave-one-run-out in train_runs set.
    """
    splits = []
    for val in train_runs:
        tr = [r for r in train_runs if r != val]
        if not tr:
            continue
        splits.append((val, tr))
    if not splits:
        raise ValueError("no valid CV splits built from train_runs")
    return splits


def rank_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def rank_asc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    return ranks


def evaluate_split(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    random_state: int,
) -> Dict[str, np.ndarray]:
    n_feat = train_x.shape[1]

    # Method-1: single-variable linear regression (lower MAE is better)
    lin_mae = np.zeros(n_feat, dtype=np.float64)
    for j in range(n_feat):
        lr = LinearRegression()
        lr.fit(train_x[:, [j]], train_y)
        pred = lr.predict(val_x[:, [j]])
        lin_mae[j] = mean_absolute_error(val_y, pred)
    lin_rank = rank_asc(lin_mae)

    # Method-2: tree importance (higher importance is better)
    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rf.fit(train_x, train_y)
    tree_imp = rf.feature_importances_.astype(np.float64)
    tree_rank = rank_desc(tree_imp)

    # Method-3: RFE rank (smaller is better)
    rfe_est = RandomForestRegressor(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    rfe = RFE(estimator=rfe_est, n_features_to_select=1, step=1)
    rfe.fit(train_x, train_y)
    rfe_rank = rfe.ranking_.astype(np.float64)

    return {
        "lin_mae": lin_mae,
        "lin_rank": lin_rank,
        "tree_imp": tree_imp,
        "tree_rank": tree_rank,
        "rfe_rank": rfe_rank,
    }


def aggregate_scores(
    feat_names: List[str],
    split_results: Dict[str, Dict[str, np.ndarray]],
    w_lin: float,
    w_tree: float,
    w_rfe: float,
    w_stability: float,
) -> List[dict]:
    split_names = list(split_results.keys())
    n_feat = len(feat_names)

    rows = []
    for j in range(n_feat):
        lin_mae_arr = np.array([split_results[s]["lin_mae"][j] for s in split_names], dtype=np.float64)
        lin_rank_arr = np.array([split_results[s]["lin_rank"][j] for s in split_names], dtype=np.float64)
        tree_imp_arr = np.array([split_results[s]["tree_imp"][j] for s in split_names], dtype=np.float64)
        tree_rank_arr = np.array([split_results[s]["tree_rank"][j] for s in split_names], dtype=np.float64)
        rfe_rank_arr = np.array([split_results[s]["rfe_rank"][j] for s in split_names], dtype=np.float64)

        rank_lin_mean = float(np.mean(lin_rank_arr))
        rank_tree_mean = float(np.mean(tree_rank_arr))
        rank_rfe_mean = float(np.mean(rfe_rank_arr))

        stability_penalty = float(np.mean([np.std(lin_rank_arr), np.std(tree_rank_arr), np.std(rfe_rank_arr)]))

        # lower is better
        composite_rank_score = (
            w_lin * rank_lin_mean
            + w_tree * rank_tree_mean
            + w_rfe * rank_rfe_mean
            + w_stability * stability_penalty
        )

        row = {
            "feature_id": j,
            "feature_name": feat_names[j],
            "lin_mae_mean": float(np.mean(lin_mae_arr)),
            "lin_mae_std": float(np.std(lin_mae_arr)),
            "tree_imp_mean": float(np.mean(tree_imp_arr)),
            "tree_imp_std": float(np.std(tree_imp_arr)),
            "rank_lin_mean": rank_lin_mean,
            "rank_tree_mean": rank_tree_mean,
            "rank_rfe_mean": rank_rfe_mean,
            "rank_lin_std": float(np.std(lin_rank_arr)),
            "rank_tree_std": float(np.std(tree_rank_arr)),
            "rank_rfe_std": float(np.std(rfe_rank_arr)),
            "stability_penalty": stability_penalty,
            "composite_rank_score": composite_rank_score,
        }

        for s in split_names:
            row[f"lin_mae__{s}"] = float(split_results[s]["lin_mae"][j])
            row[f"lin_rank__{s}"] = float(split_results[s]["lin_rank"][j])
            row[f"tree_imp__{s}"] = float(split_results[s]["tree_imp"][j])
            row[f"tree_rank__{s}"] = float(split_results[s]["tree_rank"][j])
            row[f"rfe_rank__{s}"] = float(split_results[s]["rfe_rank"][j])

        rows.append(row)

    rows.sort(key=lambda r: (r["composite_rank_score"], r["rank_lin_mean"], r["feature_name"]))
    for i, r in enumerate(rows, start=1):
        r["final_rank"] = i
    return rows


def write_rank_csv(rows: List[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def export_k_sets(
    all_runs_data: Dict[str, Dict[str, np.ndarray]],
    feat_names: List[str],
    ranked_rows: List[dict],
    ks: List[int],
    out_data_dir: Path,
    out_results_dir: Path,
    output_prefix: str,
) -> None:
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    for k in ks:
        selected_names = [r["feature_name"] for r in ranked_rows[:k]]
        selected_idx = [name_to_idx[n] for n in selected_names]

        keep_txt = out_results_dir / f"keep_features_stageB_k{k}.txt"
        keep_txt.write_text("\n".join(selected_names) + "\n", encoding="utf-8")

        for run, d in all_runs_data.items():
            xk = d["X"][:, selected_idx].astype(np.float32)
            y = d["y"].astype(np.float32)
            pass_idx = d["pass_idx"].astype(np.int32)

            npz_out = out_data_dir / f"{run}_{output_prefix}_k{k}.npz"
            np.savez(
                npz_out,
                X=xk,
                y=y,
                pass_idx=pass_idx,
                feature_names=np.array(selected_names, dtype=object),
            )

            csv_out = out_data_dir / f"{run}_{output_prefix}_k{k}_view.csv"
            with csv_out.open("w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["pass_idx", *selected_names, "wear"])
                for i in range(xk.shape[0]):
                    w.writerow([int(pass_idx[i]), *xk[i].tolist(), float(y[i])])


def main() -> None:
    p = argparse.ArgumentParser(description="Stage-B predictive selection with cross-tool stability")
    p.add_argument("--dataset_dir", type=Path, required=True)
    p.add_argument("--input_suffix", type=str, required=True)
    p.add_argument("--train_runs", type=str, required=True)
    p.add_argument("--export_runs", type=str, required=True)
    p.add_argument("--k_list", type=str, default="6,8,10")
    p.add_argument("--output_prefix", type=str, default="passlevel_td28_stageB_fold1")
    p.add_argument("--out_root", type=Path, required=True)
    p.add_argument("--random_state", type=int, default=2026)

    p.add_argument("--w_lin", type=float, default=0.5)
    p.add_argument("--w_tree", type=float, default=0.3)
    p.add_argument("--w_rfe", type=float, default=0.2)
    p.add_argument("--w_stability", type=float, default=0.3)

    args = p.parse_args()

    train_runs = parse_runs(args.train_runs)
    export_runs = parse_runs(args.export_runs)
    ks = parse_k_list(args.k_list)

    # Load train runs for fitting/ranking
    train_data: Dict[str, Dict[str, np.ndarray]] = {}
    for r in train_runs:
        train_data[r] = load_npz(args.dataset_dir, r, args.input_suffix)

    feat_names = train_data[train_runs[0]]["feature_names"].tolist()
    for r in train_runs[1:]:
        if train_data[r]["feature_names"].tolist() != feat_names:
            raise ValueError(f"feature mismatch between train runs: {r}")

    splits = make_cv_splits(train_runs)

    split_results: Dict[str, Dict[str, np.ndarray]] = {}
    for i, (val_run, tr_runs) in enumerate(splits):
        xtr = np.concatenate([train_data[r]["X"] for r in tr_runs], axis=0)
        ytr = np.concatenate([train_data[r]["y"] for r in tr_runs], axis=0)
        xval = train_data[val_run]["X"]
        yval = train_data[val_run]["y"]

        split_name = f"train_{'+'.join(tr_runs)}__val_{val_run}"
        split_results[split_name] = evaluate_split(
            train_x=xtr,
            train_y=ytr,
            val_x=xval,
            val_y=yval,
            random_state=args.random_state + i,
        )

    ranked_rows = aggregate_scores(
        feat_names=feat_names,
        split_results=split_results,
        w_lin=args.w_lin,
        w_tree=args.w_tree,
        w_rfe=args.w_rfe,
        w_stability=args.w_stability,
    )

    out_results = args.out_root / "results"
    out_data = args.out_root / "data"
    out_results.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    write_rank_csv(ranked_rows, out_results / "feature_rank_stageB.csv")

    # Load export runs (can include test_run, but no fitting uses it)
    all_runs_data: Dict[str, Dict[str, np.ndarray]] = {}
    for r in export_runs:
        all_runs_data[r] = load_npz(args.dataset_dir, r, args.input_suffix)
        if all_runs_data[r]["feature_names"].tolist() != feat_names:
            raise ValueError(f"feature mismatch in export run: {r}")

    export_k_sets(
        all_runs_data=all_runs_data,
        feat_names=feat_names,
        ranked_rows=ranked_rows,
        ks=ks,
        out_data_dir=out_data,
        out_results_dir=out_results,
        output_prefix=args.output_prefix,
    )

    summary = {
        "train_runs": train_runs,
        "export_runs": export_runs,
        "splits": [{"val_run": v, "train_runs": tr} for v, tr in splits],
        "n_features_in": len(feat_names),
        "k_list": ks,
        "weights": {
            "w_lin": args.w_lin,
            "w_tree": args.w_tree,
            "w_rfe": args.w_rfe,
            "w_stability": args.w_stability,
        },
        "strict_inductive_note": "ranking fitted by train_runs only; export_runs may include test_run for column export only",
        "top10": [r["feature_name"] for r in ranked_rows[:10]],
    }
    (out_results / "stageB_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
