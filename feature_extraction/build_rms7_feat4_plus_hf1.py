#!/usr/bin/env python3
"""
Build pass-level features:
RMS7 + Feat_4 + HF_ENERGY_RATIO_CH1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


# BASE_KEEP_IDX provenance:
# derived from Stage-B train-only predictive selection lineage.
# See feature_extraction/feature_selection_manifest.json for full provenance.
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


def hf_energy_ratio(x: np.ndarray, split_ratio: float = 0.7) -> float:
    x = x.astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    if spec.size <= 2:
        return 0.0
    spec = spec[1:]
    cut = int(np.floor(spec.size * split_ratio))
    cut = min(max(cut, 1), spec.size - 1)
    total = float(spec.sum())
    if total <= 1e-12:
        return 0.0
    return float(spec[cut:].sum() / total)


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
    hf = np.zeros((X_sel.shape[0], 1), dtype=np.float32)
    for i in range(X_sel.shape[0]):
        csv_path = run_dir / f"{prefix}_{i + 1:03d}.csv"
        raw = np.loadtxt(csv_path, delimiter=",", dtype=np.float64, usecols=[channel_idx])
        hf[i, 0] = np.float32(hf_energy_ratio(raw))

    X_out = np.concatenate([X_sel, hf], axis=1)
    feat_names = np.asarray(BASE_KEEP_NAMES + [f"HF_ENERGY_RATIO_CH{channel_idx + 1}"], dtype=object)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{run}_passlevel_rms7_plus_feat4_plus_hf1.npz",
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
        default=Path("dataset/passlevel_tree_select/selected_rms7_plus_feat4_plus_hf1"),
    )
    p.add_argument("--channel_idx", type=int, default=0)
    args = p.parse_args()

    project_root = args.project_root.resolve()
    out_dir = (project_root / args.out_dir).resolve()
    for run in args.runs:
        build_one_run(project_root, run, out_dir, channel_idx=args.channel_idx)
        print(f"[OK] built {run} -> {out_dir / f'{run}_passlevel_rms7_plus_feat4_plus_hf1.npz'}")


if __name__ == "__main__":
    main()
