#!/usr/bin/env python3
"""
Build pass-level RMS7 + PTP7 features from base_td28.

Kept dimensions:
- RMS: Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27
- PTP: Feat_4, Feat_8, Feat_12, Feat_16, Feat_20, Feat_24, Feat_28
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


KEEP_IDX = np.asarray([2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23, 26, 27], dtype=np.int64)
KEEP_NAMES = [
    "Feat_3",
    "Feat_4",
    "Feat_7",
    "Feat_8",
    "Feat_11",
    "Feat_12",
    "Feat_15",
    "Feat_16",
    "Feat_19",
    "Feat_20",
    "Feat_23",
    "Feat_24",
    "Feat_27",
    "Feat_28",
]


def build_one_run(project_root: Path, run: str, out_dir: Path) -> None:
    base_npz = project_root / "dataset/passlevel_tree_select/base_td28" / f"{run}_passlevel_td28.npz"
    z = np.load(base_npz, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    pass_idx = z["pass_idx"].astype(np.int32)

    X_out = X[:, KEEP_IDX].astype(np.float32)
    feat_names = np.asarray(KEEP_NAMES, dtype=object)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{run}_passlevel_rms7_ptp7.npz",
        X=X_out,
        y=y,
        pass_idx=pass_idx,
        feature_names=feat_names,
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=Path, required=True)
    p.add_argument("--runs", nargs="+", default=["c1", "c4", "c6"])
    p.add_argument(
        "--out_dir",
        type=Path,
        default=Path("dataset/passlevel_tree_select/selected_rms7_ptp7"),
    )
    args = p.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    for run in args.runs:
        build_one_run(project_root, run, out_dir)
        print(f"[OK] built {run} -> {out_dir / f'{run}_passlevel_rms7_ptp7.npz'}")


if __name__ == "__main__":
    main()
