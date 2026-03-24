#!/usr/bin/env python3
"""
Lightweight diagnosis for HF_ENERGY_RATIO_CH1.

Goal:
- compare HF energy ratio against the previously rejected SC1 candidate
- check whether the candidate is more wear-related and less domain-biased
- decide whether it deserves full 3-fold GPU training
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = ("c1", "c4", "c6")


@dataclass
class RunData:
    run: str
    pass_idx: np.ndarray
    wear: np.ndarray
    sc1: np.ndarray
    hf_ratio: np.ndarray


def read_wear_max(csv_path: Path) -> np.ndarray:
    vals = []
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        wear_cols = [c for c in (rd.fieldnames or []) if c != "cut"]
        for row in rd:
            vals.append(max(float(row[c]) for c in wear_cols))
    return np.asarray(vals, dtype=np.float64)


def normalized_spectral_centroid(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    denom = float(spec.sum())
    if denom <= 1e-12 or spec.size <= 1:
        return 0.0
    idx = np.arange(spec.size, dtype=np.float64)
    return float((idx * spec).sum() / denom) / float(spec.size - 1)


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
    high = float(spec[cut:].sum())
    return high / total


def infer_prefix(run_name: str) -> str:
    return f"c_{run_name[1:]}"


def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    xr = rankdata(x)
    yr = rankdata(y)
    return pearson_corr(xr, yr)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((x * y).sum() / denom)


def binary_auc(values: np.ndarray, labels: np.ndarray) -> float:
    pos = values[labels == 1]
    neg = values[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    ranks = rankdata(values)
    pos_ranks = ranks[labels == 1].sum()
    n_pos = float(len(pos))
    n_neg = float(len(neg))
    auc = (pos_ranks - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return float(auc)


def build_run(project_root: Path, run: str, channel_idx: int) -> RunData:
    base_npz = project_root / "dataset/passlevel_tree_select/base_td28" / f"{run}_passlevel_td28.npz"
    z = np.load(base_npz, allow_pickle=True)
    pass_idx = z["pass_idx"].astype(np.int32)
    wear = read_wear_max(project_root / "dataset" / run / f"{run}_wear.csv")
    prefix = infer_prefix(run)
    run_dir = project_root / "dataset" / run
    sc1 = np.zeros_like(wear, dtype=np.float64)
    hf = np.zeros_like(wear, dtype=np.float64)
    for i in range(len(wear)):
        raw = np.loadtxt(run_dir / f"{prefix}_{i + 1:03d}.csv", delimiter=",", dtype=np.float64, usecols=[channel_idx])
        sc1[i] = normalized_spectral_centroid(raw)
        hf[i] = hf_energy_ratio(raw)
    return RunData(run=run, pass_idx=pass_idx, wear=wear, sc1=sc1, hf_ratio=hf)


def plot_feature_curves(runs: list[RunData], out_path: Path, attr: str, title: str) -> None:
    plt.figure(figsize=(10, 5))
    for rd in runs:
        vals = getattr(rd, attr)
        plt.plot(rd.pass_idx, vals, label=rd.run, linewidth=1.8)
    plt.xlabel("pass index")
    plt.ylabel(attr)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scatter(runs: list[RunData], out_path: Path, attr: str, title: str) -> None:
    plt.figure(figsize=(10, 5))
    for rd in runs:
        vals = getattr(rd, attr)
        plt.scatter(rd.wear, vals, s=10, alpha=0.6, label=rd.run)
    plt.xlabel("wear (um)")
    plt.ylabel(attr)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pair_hist(run_a: RunData, run_b: RunData, out_path: Path, attr: str, title: str) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(getattr(run_a, attr), bins=30, alpha=0.55, density=True, label=run_a.run)
    plt.hist(getattr(run_b, attr), bins=30, alpha=0.55, density=True, label=run_b.run)
    plt.xlabel(attr)
    plt.ylabel("density")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--project_root", type=Path, required=True)
    p.add_argument("--channel_idx", type=int, default=0)
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("feature_alignment_diagnosis/outputs/20260324_hf_ratio_ch1_diag"),
    )
    args = p.parse_args()

    root = args.project_root.resolve()
    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [build_run(root, run, args.channel_idx) for run in RUNS]
    by_run = {rd.run: rd for rd in runs}

    corr_rows = []
    for rd in runs:
        corr_rows.append(
            {
                "run": rd.run,
                "feature": "SPEC_CENTROID_CH1",
                "pearson_wear": pearson_corr(rd.sc1, rd.wear),
                "spearman_wear": spearman_corr(rd.sc1, rd.wear),
            }
        )
        corr_rows.append(
            {
                "run": rd.run,
                "feature": "HF_ENERGY_RATIO_CH1",
                "pearson_wear": pearson_corr(rd.hf_ratio, rd.wear),
                "spearman_wear": spearman_corr(rd.hf_ratio, rd.wear),
            }
        )

    pair_rows = []
    pairs = [("c1", "c4"), ("c1", "c6"), ("c4", "c6")]
    for a, b in pairs:
        rd_a = by_run[a]
        rd_b = by_run[b]
        labels = np.concatenate([np.zeros_like(rd_a.wear, dtype=np.int32), np.ones_like(rd_b.wear, dtype=np.int32)])
        sc_vals = np.concatenate([rd_a.sc1, rd_b.sc1])
        hf_vals = np.concatenate([rd_a.hf_ratio, rd_b.hf_ratio])
        pair_rows.append({"pair": f"{a}_vs_{b}", "feature": "SPEC_CENTROID_CH1", "domain_auc": binary_auc(sc_vals, labels)})
        pair_rows.append({"pair": f"{a}_vs_{b}", "feature": "HF_ENERGY_RATIO_CH1", "domain_auc": binary_auc(hf_vals, labels)})

    plot_feature_curves(runs, out_dir / "sc1_curves.png", "sc1", "SPEC_CENTROID_CH1 across runs")
    plot_feature_curves(runs, out_dir / "hf_ratio_curves.png", "hf_ratio", "HF_ENERGY_RATIO_CH1 across runs")
    plot_scatter(runs, out_dir / "sc1_vs_wear.png", "sc1", "SPEC_CENTROID_CH1 vs wear")
    plot_scatter(runs, out_dir / "hf_ratio_vs_wear.png", "hf_ratio", "HF_ENERGY_RATIO_CH1 vs wear")
    plot_pair_hist(by_run["c1"], by_run["c4"], out_dir / "pair_hist_sc1_c1_c4.png", "sc1", "SC1 domain overlap: c1 vs c4")
    plot_pair_hist(by_run["c1"], by_run["c4"], out_dir / "pair_hist_hf_c1_c4.png", "hf_ratio", "HF ratio domain overlap: c1 vs c4")

    corr_csv = out_dir / "candidate_corr_summary.csv"
    with corr_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["run", "feature", "pearson_wear", "spearman_wear"])
        wr.writeheader()
        wr.writerows(corr_rows)

    pair_csv = out_dir / "candidate_domain_auc.csv"
    with pair_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["pair", "feature", "domain_auc"])
        wr.writeheader()
        wr.writerows(pair_rows)

    hf_corr = np.mean([r["spearman_wear"] for r in corr_rows if r["feature"] == "HF_ENERGY_RATIO_CH1"])
    sc_corr = np.mean([r["spearman_wear"] for r in corr_rows if r["feature"] == "SPEC_CENTROID_CH1"])
    hf_corr_abs = np.mean([abs(r["spearman_wear"]) for r in corr_rows if r["feature"] == "HF_ENERGY_RATIO_CH1"])
    sc_corr_abs = np.mean([abs(r["spearman_wear"]) for r in corr_rows if r["feature"] == "SPEC_CENTROID_CH1"])
    hf_auc = np.mean([abs(r["domain_auc"] - 0.5) for r in pair_rows if r["feature"] == "HF_ENERGY_RATIO_CH1"])
    sc_auc = np.mean([abs(r["domain_auc"] - 0.5) for r in pair_rows if r["feature"] == "SPEC_CENTROID_CH1"])

    if hf_corr_abs > sc_corr_abs and hf_auc < sc_auc:
        verdict = "HF_ENERGY_RATIO_CH1 is a better next candidate than SC1 and deserves 3-fold training."
    else:
        verdict = "HF_ENERGY_RATIO_CH1 does not yet show a clear advantage over SC1; do not start 3-fold training yet."

    summary_md = out_dir / "hf_ratio_candidate_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# HF_ENERGY_RATIO_CH1 轻量诊断",
                "",
                f"- 时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "- 目标: 在正式三折训练前，比较 HF_ENERGY_RATIO_CH1 与已失败候选 SC1 的稳健性。",
                "",
                "## 判断准则",
                "- wear 相关性更强",
                "- 训练域间分离程度更低（domain AUC 更接近 0.5）",
                "",
                "## 汇总结论",
                f"- 平均 Spearman wear 相关: HF={hf_corr:.4f}, SC1={sc_corr:.4f}",
                f"- 平均 |Spearman| wear 相关: HF={hf_corr_abs:.4f}, SC1={sc_corr_abs:.4f}",
                f"- 平均训练域分离偏离 |AUC-0.5|: HF={hf_auc:.4f}, SC1={sc_auc:.4f}",
                f"- Verdict: {verdict}",
                "",
                "## 输出文件",
                f"- `{corr_csv.name}`",
                f"- `{pair_csv.name}`",
                "- `sc1_curves.png`",
                "- `hf_ratio_curves.png`",
                "- `sc1_vs_wear.png`",
                "- `hf_ratio_vs_wear.png`",
                "- `pair_hist_sc1_c1_c4.png`",
                "- `pair_hist_hf_c1_c4.png`",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "output_dir": str(out_dir),
        "channel_idx": args.channel_idx,
        "verdict": verdict,
        "mean_spearman_hf": hf_corr,
        "mean_spearman_sc1": sc_corr,
        "mean_abs_spearman_hf": hf_corr_abs,
        "mean_abs_spearman_sc1": sc_corr_abs,
        "mean_domain_bias_hf": hf_auc,
        "mean_domain_bias_sc1": sc_auc,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] diagnosis written to {out_dir}")
    print(f"[OK] verdict: {verdict}")


if __name__ == "__main__":
    main()
