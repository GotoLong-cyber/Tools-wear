#!/usr/bin/env python3
"""
Build pass-level RMS7 + wavelet-energy-ratio-7 features.

Kept dimensions:
- RMS: Feat_3, Feat_7, Feat_11, Feat_15, Feat_19, Feat_23, Feat_27
- WAVELET_ENERGY_RATIO_CH{1..7}: one scalar per channel computed from db1 level-3 detail energy ratio
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pywt


RMS_KEEP_IDX = np.asarray([2, 6, 10, 14, 18, 22, 26], dtype=np.int64)
RMS_KEEP_NAMES = [
    "Feat_3",
    "Feat_7",
    "Feat_11",
    "Feat_15",
    "Feat_19",
    "Feat_23",
    "Feat_27",
]


def infer_prefix(run_name: str) -> str:
    if not (run_name.startswith("c") and run_name[1:].isdigit()):
        raise ValueError(f"bad run name: {run_name}")
    return f"c_{run_name[1:]}"


def safe_arr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def wavelet_energy_ratio(x: np.ndarray, wavelet: str = "db1", level: int = 3) -> float:
    x = safe_arr(x)
    x = x - np.mean(x)
    coeffs = pywt.wavedec(x, wavelet=wavelet, level=level, mode="symmetric")
    energies = np.asarray([float(np.sum(c * c)) for c in coeffs], dtype=np.float64)
    total = float(np.sum(energies))
    if total <= 1e-12:
        return 0.0
    detail = float(np.sum(energies[1:]))
    return detail / total


def build_one_run(project_root: Path, run: str, out_dir: Path) -> None:
    base_npz = project_root / "dataset/passlevel_tree_select/base_td28" / f"{run}_passlevel_td28.npz"
    run_dir = project_root / "dataset" / run
    z = np.load(base_npz, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.float32)
    pass_idx = z["pass_idx"].astype(np.int32)

    X_sel = X[:, RMS_KEEP_IDX].astype(np.float32)
    prefix = infer_prefix(run)
    wav = np.zeros((X_sel.shape[0], 7), dtype=np.float32)
    for i in range(X_sel.shape[0]):
        csv_path = run_dir / f"{prefix}_{i + 1:03d}.csv"
        raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, usecols=list(range(7)))
        for ch in range(7):
            wav[i, ch] = np.float32(wavelet_energy_ratio(raw[:, ch]))

    X_out = np.concatenate([X_sel, wav], axis=1)
    feat_names = np.asarray(
        RMS_KEEP_NAMES + [f"WAVELET_ENERGY_RATIO_CH{i}" for i in range(1, 8)],
        dtype=object,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{run}_passlevel_rms7_wav7.npz",
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
        default=Path("dataset/passlevel_tree_select/selected_rms7_wav7"),
    )
    args = p.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    for run in args.runs:
        build_one_run(project_root, run, out_dir)
        print(f"[OK] built {run} -> {out_dir / f'{run}_passlevel_rms7_wav7.npz'}")


if __name__ == "__main__":
    main()
