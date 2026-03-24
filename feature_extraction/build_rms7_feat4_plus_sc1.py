#!/usr/bin/env python3
"""
Build pass-level features:
RMS7 + Feat_4 + normalized spectral centroid on channel-1.

This is the first "A2 + orthogonal physical information" candidate.
The added feature is frequency-domain and does not introduce a new
learned component or test-time leakage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BASE_KEEP_IDX = np.asarray([2, 6, 10, 14, 18, 22, 26, 3], dtype=np.int64)
BASE_KEEP_NAMES = [
    "Feat_3",
    "Feat_7",
    "Feat_11",
    "Feat_15",
    "Feat_19",
    "Feat_23",
    "Feat_27",
    "Feat_4",
]


def normalized_spectral_centroid(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    denom = float(spec.sum())
    if denom <= 1e-12 or spec.size <= 1:
        return 0.0
    freq_idx = np.arange(spec.size, dtype=np.float64)
    centroid = float((freq_idx * spec).sum() / denom)
    return centroid / float(spec.size - 1)


def infer_prefix(run_name: str) -> str:
    if not (run_name.startswith("c") and run_name[1:].isdigit()):
        raise ValueError(f"bad run name: {run_name}")
    return f"c_{run_name[1:]}"


def build_one_run(project_root: Path, run: str, out_dir: Path, channel_idx: int) -> None:
    base_npz = project_root / "dataset/passlevel_tree_select/base_td28" / f"{run}_passlevel_td28.npz"
    run_dir = project_root / "dataset" / run
    z = np.load(base_npz, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    pass_idx = z["pass_idx"].astype(np.int32)

    X_sel = X[:, BASE_KEEP_IDX].astype(np.float32)
    prefix = infer_prefix(run)
    sc = np.zeros((X_sel.shape[0], 1), dtype=np.float32)
    for i in range(X_sel.shape[0]):
        csv_path = run_dir / f"{prefix}_{i + 1:03d}.csv"
        raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, usecols=[channel_idx])
        sc[i, 0] = np.float32(normalized_spectral_centroid(raw))

    X_out = np.concatenate([X_sel, sc], axis=1)
    feat_names = np.asarray(BASE_KEEP_NAMES + [f"SPEC_CENTROID_CH{channel_idx + 1}"], dtype=object)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{run}_passlevel_rms7_plus_feat4_plus_sc1.npz",
        X=X_out,
        y=y,
        pass_idx=pass_idx,
        feature_names=feat_names,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=Path, required=True)
    p.add_argument(
        "--runs",
        nargs="+",
        default=["c1", "c4", "c6"],
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_sc1"),
    )
    p.add_argument("--channel_idx", type=int, default=0)
    args = p.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    for run in args.runs:
        build_one_run(project_root, run, out_dir, channel_idx=args.channel_idx)
        print(f"[OK] built {run} -> {out_dir / f'{run}_passlevel_rms7_plus_feat4_plus_sc1.npz'}")


if __name__ == "__main__":
    main()
