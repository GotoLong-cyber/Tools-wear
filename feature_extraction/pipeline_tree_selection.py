import argparse
import csv
import json
import os
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _safe_arr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _rms(x: np.ndarray) -> float:
    x = _safe_arr(x)
    return float(np.sqrt(np.mean(x * x)))


def _read_wear_csv(path: str, wear_agg: str = "max") -> np.ndarray:
    df = pd.read_csv(path)
    if "cut" in df.columns:
        df = df.sort_values("cut").reset_index(drop=True)

    cols = [c for c in ["flute_1", "flute_2", "flute_3"] if c in df.columns]
    if wear_agg in df.columns:
        y = df[wear_agg].to_numpy(dtype=np.float32)
    elif wear_agg == "mean":
        y = df[cols].mean(axis=1).to_numpy(dtype=np.float32)
    else:
        y = df[cols].max(axis=1).to_numpy(dtype=np.float32)
    return y


def _infer_prefix(run_name: str) -> str:
    if not (run_name.startswith("c") and run_name[1:].isdigit()):
        raise ValueError(f"bad run name: {run_name}")
    return f"c_{run_name[1:]}"


def _extract_one_pass(raw: np.ndarray, feature_set: str) -> np.ndarray:
    x = _safe_arr(raw[:, :7])
    if feature_set == "avg7":
        return np.mean(x, axis=0).astype(np.float32)

    if feature_set == "td28":
        feats: List[float] = []
        for ch in range(7):
            s = x[:, ch]
            feats.extend(
                [
                    float(np.mean(s)),
                    float(np.std(s)),
                    _rms(s),
                    float(np.max(s) - np.min(s)),
                ]
            )
        return np.asarray(feats, dtype=np.float32)

    raise ValueError(f"unknown feature_set: {feature_set}")


def _build_run_features(dataset_root: str, run_name: str, feature_set: str, wear_agg: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    run_dir = os.path.join(dataset_root, run_name)
    wear_csv = os.path.join(run_dir, f"{run_name}_wear.csv")
    if not os.path.exists(wear_csv):
        raise FileNotFoundError(f"wear csv not found: {wear_csv}")

    y = _read_wear_csv(wear_csv, wear_agg=wear_agg)
    T = len(y)

    prefix = _infer_prefix(run_name)
    sample_csv = os.path.join(run_dir, f"{prefix}_001.csv")
    d = _extract_one_pass(pd.read_csv(sample_csv, header=None).to_numpy(dtype=np.float64), feature_set).shape[0]

    X = np.zeros((T, d), dtype=np.float32)
    for k in range(1, T + 1):
        fp = os.path.join(run_dir, f"{prefix}_{k:03d}.csv")
        if not os.path.exists(fp):
            raise FileNotFoundError(f"sensor file missing: {fp}")
        raw = pd.read_csv(fp, header=None).to_numpy(dtype=np.float64)
        X[k - 1] = _extract_one_pass(raw, feature_set)

    pass_idx = np.arange(T, dtype=np.int32)
    return X, y.astype(np.float32), pass_idx


def _train_coverage_end(T: int, seq_len: int, pred_len: int, split_ratio: float) -> int:
    n_windows = T - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(f"T={T} too short for seq_len={seq_len}, pred_len={pred_len}")
    n_train = max(int(np.floor(n_windows * split_ratio)), 1)
    last_start = n_train - 1
    return min(last_start + seq_len + pred_len, T)


def _parse_csv_list(s: str) -> List[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def _save_npz(path: str, X: np.ndarray, y: np.ndarray, pass_idx: np.ndarray, feature_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        y=y,
        pass_idx=pass_idx,
        feature_names=np.asarray(feature_names, dtype=object),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/dataset")
    parser.add_argument("--runs", type=str, default="c1,c4,c6")
    parser.add_argument("--train_runs", type=str, default="c1,c4")
    parser.add_argument("--feature_set", type=str, default="td28", choices=["td28", "avg7"])
    parser.add_argument("--wear_agg", type=str, default="max", choices=["max", "mean", "flute_1", "flute_2", "flute_3"])
    parser.add_argument("--seq_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=16)
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--rf_seeds", type=str, default="2026,2027,2028")
    parser.add_argument("--n_estimators", type=int, default=600)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="/home/lc24/Timer/FeatureExact3.10/FeatureTest/OpenLTM/dataset/passlevel_tree_select")
    args = parser.parse_args()

    runs = _parse_csv_list(args.runs)
    train_runs = _parse_csv_list(args.train_runs)
    rf_seeds = [int(x.strip()) for x in args.rf_seeds.split(",") if x.strip()]

    base_dir = os.path.join(args.out_dir, f"base_{args.feature_set}")
    selected_dir = os.path.join(args.out_dir, f"selected_{args.feature_set}_k{args.top_k}")
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(selected_dir, exist_ok=True)

    run_data = {}
    for run in runs:
        X, y, pass_idx = _build_run_features(args.dataset_root, run, args.feature_set, args.wear_agg)
        d = X.shape[1]
        feat_names = [f"Feat_{i + 1}" for i in range(d)]
        out_npz = os.path.join(base_dir, f"{run}_passlevel_{args.feature_set}.npz")
        _save_npz(out_npz, X, y, pass_idx, feat_names)
        run_data[run] = {"X": X, "y": y, "pass_idx": pass_idx, "feature_names": feat_names}
        print(f"[Base] {run}: X{X.shape}, y{y.shape} -> {out_npz}")

    xs, ys = [], []
    for run in train_runs:
        Xr = run_data[run]["X"]
        yr = run_data[run]["y"]
        end = _train_coverage_end(T=Xr.shape[0], seq_len=args.seq_len, pred_len=args.pred_len, split_ratio=args.split_ratio)
        xs.append(Xr[:end])
        ys.append(yr[:end])
        print(f"[TrainSlice] {run}: use 0:{end} / {Xr.shape[0]}")

    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    print(f"[TrainMatrix] X{X_train.shape}, y{y_train.shape}")

    imps = []
    for seed in rf_seeds:
        rf = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        imps.append(rf.feature_importances_)
        print(f"[RF] seed={seed} done")

    imp_mat = np.vstack(imps)
    imp_mean = imp_mat.mean(axis=0)
    imp_std = imp_mat.std(axis=0)

    d = X_train.shape[1]
    k = max(1, min(int(args.top_k), d))
    selected_idx = np.argsort(imp_mean)[::-1][:k]
    selected_names = [f"Feat_{int(i) + 1}" for i in selected_idx]

    keep_txt = os.path.join(selected_dir, f"keep_features_tree_{args.feature_set}_k{k}.txt")
    with open(keep_txt, "w", encoding="utf-8") as f:
        for name in selected_names:
            f.write(name + "\n")
    print(f"[Keep] {keep_txt}")

    imp_csv = os.path.join(selected_dir, f"tree_importance_{args.feature_set}_k{k}.csv")
    with open(imp_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature", "importance_mean", "importance_std"])
        for rank, idx in enumerate(selected_idx, start=1):
            w.writerow([rank, f"Feat_{int(idx) + 1}", float(imp_mean[idx]), float(imp_std[idx])])
    print(f"[Importance] {imp_csv}")

    for run in runs:
        Xr = run_data[run]["X"][:, selected_idx]
        yr = run_data[run]["y"]
        pr = run_data[run]["pass_idx"]
        out_npz = os.path.join(selected_dir, f"{run}_passlevel_{args.feature_set}_tree_k{k}.npz")
        _save_npz(out_npz, Xr, yr, pr, selected_names)
        print(f"[SelectedNPZ] {run}: X{Xr.shape}, y{yr.shape} -> {out_npz}")

    meta = {
        "dataset_root": args.dataset_root,
        "runs": runs,
        "train_runs": train_runs,
        "feature_set": args.feature_set,
        "wear_agg": args.wear_agg,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "split_ratio": args.split_ratio,
        "rf_seeds": rf_seeds,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "top_k": k,
        "keep_features_txt": keep_txt,
        "selected_feature_names": selected_names,
    }
    meta_json = os.path.join(selected_dir, "selection_meta.json")
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[Meta] {meta_json}")


if __name__ == "__main__":
    main()
