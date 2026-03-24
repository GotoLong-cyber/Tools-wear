#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def load_case_arrays(selected_dir: Path, suffix: str):
    packs = {}
    for run in ["c1", "c4", "c6"]:
        p = selected_dir / f"{run}_passlevel_{suffix}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing npz: {p}")
        z = np.load(p, allow_pickle=True)
        packs[run] = {
            "X": z["X"].astype(np.float32),
            "y": z["y"].astype(np.float32),
            "feature_names": [str(x) for x in z["feature_names"].tolist()],
        }
    return packs


def rbf_mmd2(x: np.ndarray, y: np.ndarray) -> float:
    xy = np.vstack([x, y])
    sq_d = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    # median heuristic
    tri = sq_d[np.triu_indices_from(sq_d, 1)]
    med = np.median(tri[tri > 0]) if np.any(tri > 0) else 1.0
    gamma = 1.0 / (2.0 * med + 1e-12)

    def k(a, b):
        d = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * d)

    kxx = k(x, x)
    kyy = k(y, y)
    kxy = k(x, y)

    m = x.shape[0]
    n = y.shape[0]
    mmd2 = (kxx.sum() - np.trace(kxx)) / (m * (m - 1) + 1e-12) + \
           (kyy.sum() - np.trace(kyy)) / (n * (n - 1) + 1e-12) - \
           2.0 * kxy.mean()
    return float(mmd2)


def coral_distance(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    cx = (xc.T @ xc) / max(1, x.shape[0] - 1)
    cy = (yc.T @ yc) / max(1, y.shape[0] - 1)
    return float(np.sum((cx - cy) ** 2))


def get_domain_auc_single_features(x_std: np.ndarray, domain_binary: np.ndarray, feat_names: list[str]):
    rows = []
    for i, fn in enumerate(feat_names):
        vals = x_std[:, i]
        auc = roc_auc_score(domain_binary, vals)
        auc = max(auc, 1.0 - auc)  # symmetric
        rows.append((fn, float(auc)))
    rows.sort(key=lambda t: t[1], reverse=True)
    return rows


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def plot_pca_by_domain(pc: np.ndarray, domains: np.ndarray, out: Path):
    ensure_dir(out.parent)
    cmap = {"c1": "#1f77b4", "c4": "#2ca02c", "c6": "#d62728"}
    plt.figure(figsize=(7, 5), dpi=150)
    for d in ["c1", "c4", "c6"]:
        m = domains == d
        plt.scatter(pc[m, 0], pc[m, 1], s=12, alpha=0.6, c=cmap[d], label=d)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA by Domain")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_pca_by_wear_bin(pc: np.ndarray, wear: np.ndarray, q33: float, q67: float, out: Path):
    ensure_dir(out.parent)
    bins = np.where(wear <= q33, "low", np.where(wear <= q67, "mid", "high"))
    cmap = {"low": "#1f77b4", "mid": "#ff7f0e", "high": "#d62728"}
    plt.figure(figsize=(7, 5), dpi=150)
    for b in ["low", "mid", "high"]:
        m = bins == b
        plt.scatter(pc[m, 0], pc[m, 1], s=12, alpha=0.6, c=cmap[b], label=b)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA by Wear Bin")
    plt.legend(frameon=False)
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_pca_by_wear_continuous(pc: np.ndarray, wear: np.ndarray, out: Path):
    ensure_dir(out.parent)
    plt.figure(figsize=(7, 5), dpi=150)
    sc = plt.scatter(pc[:, 0], pc[:, 1], c=wear, s=12, cmap="viridis", alpha=0.75)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA by Continuous Wear")
    cb = plt.colorbar(sc)
    cb.set_label("wear")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def plot_pca_wear_panels(pc: np.ndarray, wear: np.ndarray, domains: np.ndarray, out: Path):
    ensure_dir(out.parent)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=150, sharex=True, sharey=True)
    for ax, d in zip(axes, ["c1", "c4", "c6"]):
        m = domains == d
        sc = ax.scatter(pc[m, 0], pc[m, 1], c=wear[m], s=12, cmap="viridis", alpha=0.75)
        ax.set_title(d)
        ax.grid(alpha=0.25, linestyle="--")
    axes[0].set_ylabel("PC2")
    for ax in axes:
        ax.set_xlabel("PC1")
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("wear")
    fig.suptitle("PCA by Continuous Wear (Domain Panels)")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_box_feature(values_by_run: dict[str, np.ndarray], feat_name: str, out: Path):
    ensure_dir(out.parent)
    data = [values_by_run[r] for r in ["c1", "c4", "c6"]]
    plt.figure(figsize=(6, 4), dpi=150)
    plt.boxplot(data, labels=["c1", "c4", "c6"], showfliers=False)
    plt.title(f"Boxplot of {feat_name} by Domain")
    plt.ylabel("feature value")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def write_summary_row(summary_csv: Path, row: dict):
    fields = [
        "case",
        "time",
        "n_features",
        "mmd_rbf_train_vs_test",
        "coral_train_vs_test",
        "overall_domain_auc_train_vs_test",
        "pca_explained_ratio_sum",
    ]

    existing = []
    if summary_csv.exists():
        with summary_csv.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for rr in r:
                if rr.get("case") != row["case"]:
                    existing.append(rr)

    existing.append({k: row[k] for k in fields})

    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(existing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--selected_dir", type=Path, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    parser.add_argument("--focus_feature", type=str, default="")
    args = parser.parse_args()

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    packs = load_case_arrays(args.selected_dir, args.suffix)
    feat_names = packs["c1"]["feature_names"]

    x_train = np.vstack([packs["c1"]["X"], packs["c4"]["X"]])
    y_train = np.concatenate([packs["c1"]["y"], packs["c4"]["y"]])
    x_test = packs["c6"]["X"]
    y_test = packs["c6"]["y"]

    scaler = StandardScaler().fit(x_train)
    x_all_std = np.vstack([
        scaler.transform(packs["c1"]["X"]),
        scaler.transform(packs["c4"]["X"]),
        scaler.transform(packs["c6"]["X"]),
    ])
    wear_all = np.concatenate([packs["c1"]["y"], packs["c4"]["y"], packs["c6"]["y"]])
    domains = np.array(["c1"] * packs["c1"]["X"].shape[0] + ["c4"] * packs["c4"]["X"].shape[0] + ["c6"] * packs["c6"]["X"].shape[0])

    pca = PCA(n_components=2, random_state=2026)
    pc = pca.fit_transform(x_all_std)

    domain_binary = np.concatenate([np.zeros(x_train.shape[0]), np.ones(x_test.shape[0])])
    x_train_test_std = np.vstack([scaler.transform(x_train), scaler.transform(x_test)])

    clf = LogisticRegression(max_iter=1000, random_state=2026)
    clf.fit(x_train_test_std, domain_binary)
    probs = clf.predict_proba(x_train_test_std)[:, 1]
    overall_auc = float(roc_auc_score(domain_binary, probs))

    single_auc = get_domain_auc_single_features(x_train_test_std, domain_binary, feat_names)

    out_case_dir = args.output_root / args.case
    ensure_dir(out_case_dir)

    # write single-feature auc
    with (out_case_dir / "single_feature_domain_auc.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "domain_auc_train_vs_test"])
        for fn, auc in single_auc:
            w.writerow([fn, f"{auc:.10f}"])

    # plots
    q33 = float(np.quantile(y_train, 0.33))
    q67 = float(np.quantile(y_train, 0.67))
    plot_pca_by_domain(pc, domains, out_case_dir / "pca_by_domain.png")
    plot_pca_by_wear_bin(pc, wear_all, q33, q67, out_case_dir / "pca_by_wear_bin.png")
    plot_pca_by_wear_continuous(pc, wear_all, out_case_dir / "pca_by_wear_continuous.png")
    plot_pca_wear_panels(pc, wear_all, domains, out_case_dir / "pca_by_wear_continuous_domain_panels.png")

    # boxplots: top3 + focus feature if provided
    chosen = [fn for fn, _ in single_auc[:3]]
    if args.focus_feature and args.focus_feature in feat_names and args.focus_feature not in chosen:
        chosen.append(args.focus_feature)

    for fn in chosen:
        idx = feat_names.index(fn)
        values_by_run = {r: packs[r]["X"][:, idx] for r in ["c1", "c4", "c6"]}
        plot_box_feature(values_by_run, fn, out_case_dir / f"box_{fn}.png")

    metrics = {
        "case": args.case,
        "time": now_str,
        "n_features": int(len(feat_names)),
        "train_points": int(x_train.shape[0]),
        "test_points": int(x_test.shape[0]),
        "pca_explained_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "mmd_rbf_train_vs_test": rbf_mmd2(scaler.transform(x_train), scaler.transform(x_test)),
        "coral_train_vs_test": coral_distance(x_train, x_test),
        "overall_domain_auc_train_vs_test": overall_auc,
        "top3_single_feature_domain_auc": [
            {"feature": fn, "domain_auc_train_vs_test": auc}
            for fn, auc in single_auc[:3]
        ],
        "wear_bin_thresholds_train_q33_q67": [q33, q67],
    }

    with (out_case_dir / "alignment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # summary + readme
    write_summary_row(args.output_root / "summary_metrics.csv", metrics)

    readme = args.output_root / "README.txt"
    lines = []
    if readme.exists():
        lines = readme.read_text(encoding="utf-8").splitlines()
    if not lines:
        lines = ["Fold1 对齐诊断产物", f"生成时间: {now_str}"]
    case_line_idx = None
    for i, line in enumerate(lines):
        if line.startswith("case:"):
            case_line_idx = i
            break
    if case_line_idx is None:
        lines.append(f"case: {args.case}")
    else:
        cases = [x.strip() for x in lines[case_line_idx].split(":", 1)[1].split(",") if x.strip()]
        if args.case not in cases:
            cases.append(args.case)
        lines[case_line_idx] = "case: " + ", ".join(cases)
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] generated case: {args.case}")
    print(f"[OK] output: {out_case_dir}")


if __name__ == "__main__":
    main()
