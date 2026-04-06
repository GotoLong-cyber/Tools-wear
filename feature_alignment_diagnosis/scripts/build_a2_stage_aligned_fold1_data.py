#!/usr/bin/env python3
"""
DEPRECATED: exploratory script only.
Uses full-run wear_mono[-1] for progress normalization (non-causal within-run future leakage).
Do NOT use for paper results. Official A2 is implemented in data_loader.py as train-time augmentation.

A2 (fold1) data builder: fixed-step baseline vs wear-stage aligned slicing.

This script builds a stage-aligned variant for train runs (c1/c4 by default),
while keeping test run (c6) untouched. It does not train model.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


def read_wear_max(csv_path: Path) -> np.ndarray:
    cuts = []
    vals = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        wear_cols = [c for c in cols if c != "cut"]
        if not wear_cols:
            raise ValueError(f"{csv_path}: no wear columns")
        for i, row in enumerate(rd):
            cut = float(row["cut"]) if "cut" in row and row["cut"] != "" else float(i + 1)
            cuts.append(cut)
            vals.append(max(float(row[c]) for c in wear_cols))
    if not vals:
        raise ValueError(f"{csv_path}: empty wear file")
    return np.asarray(vals, dtype=np.float32)


def write_wear_csv(csv_path: Path, wear: np.ndarray) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        # keep columns compatible with current loader wear_agg=max
        wr.writerow(["cut", "flute_1", "flute_2", "flute_3"])
        for i, v in enumerate(wear, start=1):
            wr.writerow([i, float(v), float(v), float(v)])


def progress_align(X: np.ndarray, wear: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Align series to uniform wear progress axis [0, 1].
    """
    T = len(wear)
    if T < 4:
        return X.copy(), wear.copy()

    wear_mono = np.maximum.accumulate(wear.astype(np.float64))
    w0 = float(wear_mono[0])
    wr = float(wear_mono[-1] - wear_mono[0])
    if wr <= 1e-12:
        return X.copy(), wear.copy()

    p = (wear_mono - w0) / wr
    # make strictly increasing for interpolation
    for i in range(1, T):
        if p[i] <= p[i - 1]:
            p[i] = min(1.0, p[i - 1] + 1e-6)
    if p[-1] <= p[0]:
        return X.copy(), wear.copy()

    grid = np.linspace(0.0, 1.0, T, dtype=np.float64)

    X_new = np.empty_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        X_new[:, j] = np.interp(grid, p, X[:, j]).astype(np.float32)
    y_new = np.interp(grid, p, wear_mono).astype(np.float32)
    return X_new, y_new


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--base_npz_dir",
        type=Path,
        default=Path("dataset/passlevel_tree_select/base_td28"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("dataset/passlevel_tree_select/selected_rms7_plus_feat4_a2_stagealign_fold1"),
    )
    parser.add_argument("--runs", nargs="+", default=["c1", "c4", "c6"])
    parser.add_argument("--train_runs", nargs="+", default=["c1", "c4"])
    parser.add_argument(
        "--feature_idx",
        nargs="+",
        type=int,
        default=[2, 6, 10, 14, 18, 22, 26, 3],
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    base_npz_dir = (root / args.base_npz_dir).resolve()
    out_dir = (root / args.output_dir).resolve()
    wear_out_dir = out_dir / "wear_csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    wear_out_dir.mkdir(parents=True, exist_ok=True)

    idx = np.asarray(args.feature_idx, dtype=np.int64)
    feat_names = [f"Feat_{i + 1}" for i in idx]

    audit = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "base_npz_dir": str(base_npz_dir),
        "output_dir": str(out_dir),
        "runs": args.runs,
        "train_runs": args.train_runs,
        "feature_idx": idx.tolist(),
        "feature_names": feat_names,
        "strategy": "wear_progress_alignment_for_train_runs_only",
        "note": "test run remains untouched; no test label used for fitting.",
        "records": [],
    }

    for run in args.runs:
        npz_path = base_npz_dir / f"{run}_passlevel_td28.npz"
        wear_csv_path = root / "dataset" / run / f"{run}_wear.csv"
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        if not wear_csv_path.exists():
            raise FileNotFoundError(wear_csv_path)

        z = np.load(npz_path, allow_pickle=True)
        X = z["X"].astype(np.float32)
        T = X.shape[0]
        y = read_wear_max(wear_csv_path)
        if len(y) != T:
            raise ValueError(f"{run}: wear len {len(y)} != X len {T}")

        X_sel = X[:, idx].astype(np.float32)
        if run in args.train_runs:
            X_new, y_new = progress_align(X_sel, y)
            aligned = True
        else:
            X_new, y_new = X_sel, y
            aligned = False

        pass_idx = (
            z["pass_idx"].astype(np.int32)
            if "pass_idx" in z.files and len(z["pass_idx"]) == T
            else np.arange(1, T + 1, dtype=np.int32)
        )

        np.savez(
            out_dir / f"{run}_passlevel_rms7_plus_feat4_a2stage.npz",
            X=X_new.astype(np.float32),
            y=y_new.astype(np.float32),
            pass_idx=pass_idx,
            feature_names=np.array(feat_names, dtype=object),
        )
        write_wear_csv(wear_out_dir / f"{run}_wear.csv", y_new.astype(np.float32))

        audit["records"].append(
            {
                "run": run,
                "aligned": aligned,
                "T": int(T),
                "wear_start": float(y[0]),
                "wear_end": float(y[-1]),
                "wear_new_start": float(y_new[0]),
                "wear_new_end": float(y_new[-1]),
            }
        )

    (out_dir / "keep_features_rms7_plus_feat4_a2stage.txt").write_text(
        "\n".join(feat_names) + "\n", encoding="utf-8"
    )
    with (out_dir / "a2_stagealign_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)

    print(f"[A2][OK] built at: {out_dir}")
    print(f"[A2][OK] wear csv at: {wear_out_dir}")
    print(f"[A2][OK] features: {feat_names}")


if __name__ == "__main__":
    main()
