import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _safe_arr(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _trimmed(x: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    x = _safe_arr(x)
    if ratio <= 0:
        return x
    lo = np.quantile(x, ratio)
    hi = np.quantile(x, 1.0 - ratio)
    sel = x[(x >= lo) & (x <= hi)]
    return sel if sel.size > 0 else x


def _safe_std(x: np.ndarray) -> float:
    s = float(np.std(x))
    return s if s > 1e-12 else 1e-12


def _rms(x: np.ndarray) -> float:
    x = _safe_arr(x)
    return float(np.sqrt(np.mean(x * x)))


def _skew(x: np.ndarray) -> float:
    x = _safe_arr(x)
    m = float(np.mean(x))
    s = _safe_std(x)
    z = (x - m) / s
    return float(np.mean(z ** 3))


def _kurtosis_excess(x: np.ndarray) -> float:
    x = _safe_arr(x)
    m = float(np.mean(x))
    s = _safe_std(x)
    z = (x - m) / s
    return float(np.mean(z ** 4) - 3.0)


def _spectral_entropy(psd: np.ndarray) -> float:
    psd = _safe_arr(psd)
    psd_sum = float(np.sum(psd) + 1e-12)
    p = psd / psd_sum
    return float(-np.sum(p * np.log(p + 1e-12)))


def _one_channel_19_feats(sig: np.ndarray, fs: float) -> np.ndarray:
    x = _safe_arr(sig)
    x = x - np.mean(x)

    mean = float(np.mean(x))
    std = float(np.std(x))
    r = _rms(x)
    vmin = float(np.min(x))
    vmax = float(np.max(x))
    ptp = float(vmax - vmin)
    skw = _skew(x)
    krt = _kurtosis_excess(x)
    crest = float(vmax / (r + 1e-12))

    n = len(x)
    xf = np.fft.rfft(x)
    mag = np.abs(xf)
    psd = (mag ** 2) / (n + 1e-12)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd_sum = float(np.sum(psd) + 1e-12)

    centroid = float(np.sum(freqs * psd) / psd_sum)
    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    peak_power = float(psd[peak_idx])
    sent = _spectral_entropy(psd)
    cumsum = np.cumsum(psd) / psd_sum
    rolloff85 = float(freqs[np.searchsorted(cumsum, 0.85)])

    bands = [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000), (20000, fs / 2)]
    band_powers = []
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        band_powers.append(float(np.sum(psd[m]) / psd_sum))

    feats = np.array(
        [
            mean,
            std,
            r,
            vmin,
            vmax,
            ptp,
            skw,
            krt,
            crest,
            centroid,
            peak_freq,
            peak_power,
            sent,
            rolloff85,
            band_powers[0],
            band_powers[1],
            band_powers[2],
            band_powers[3],
            band_powers[4],
        ],
        dtype=np.float32,
    )
    return feats


def _read_wear_csv(wear_csv: str, wear_agg: str = "max") -> np.ndarray:
    df = pd.read_csv(wear_csv)
    if "cut" in df.columns:
        df = df.sort_values("cut").reset_index(drop=True)

    if wear_agg in df.columns:
        y = df[wear_agg].to_numpy(dtype=np.float32)
    else:
        num_cols = [c for c in df.columns if c != "cut"]
        mat = df[num_cols].to_numpy(dtype=np.float32)
        if wear_agg == "mean":
            y = mat.mean(axis=1)
        else:
            y = mat.max(axis=1)
    return y.astype(np.float32)


def _build_run_133_features(
    dataset_root: str,
    run_name: str,
    fs: float = 50000.0,
    trim_ratio: float = 0.01,
    wear_agg: str = "max",
    n_ch: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    run_dir = os.path.join(dataset_root, run_name)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir not found: {run_dir}")

    wear_csv = os.path.join(run_dir, f"{run_name}_wear.csv")
    if not os.path.exists(wear_csv):
        raise FileNotFoundError(f"wear csv not found: {wear_csv}")
    y = _read_wear_csv(wear_csv, wear_agg=wear_agg)

    if not (run_name.startswith("c") and run_name[1:].isdigit()):
        raise ValueError(f"bad run name: {run_name}")
    prefix = f"c_{run_name[1:]}"

    total = len(y)
    X = np.zeros((total, n_ch * 19), dtype=np.float32)
    for k in range(1, total + 1):
        csv_path = os.path.join(run_dir, f"{prefix}_{k:03d}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sensor csv not found: {csv_path}")
        raw = pd.read_csv(csv_path, header=None).to_numpy(dtype=np.float64)
        raw = _safe_arr(raw)
        if raw.ndim != 2 or raw.shape[1] < n_ch:
            raise ValueError(f"{csv_path} bad shape: {raw.shape}")

        feats = []
        for ch in range(n_ch):
            sig = _trimmed(raw[:, ch], ratio=trim_ratio)
            feats.append(_one_channel_19_feats(sig, fs=fs))
        X[k - 1] = np.concatenate(feats, axis=0)

    return X, y


def _train_coverage_end(T: int, seq_len: int, pred_len: int, split_ratio: float) -> int:
    n_windows = T - seq_len - pred_len + 1
    if n_windows <= 0:
        raise ValueError(f"T={T} too short for seq_len={seq_len}, pred_len={pred_len}")
    n_train = max(int(np.floor(n_windows * split_ratio)), 1)
    last_start = n_train - 1
    return min(last_start + seq_len + pred_len, T)


def build_train_matrix_from_raw(
    dataset_root: str,
    train_runs: Sequence[str],
    seq_len: int = 96,
    pred_len: int = 16,
    split_ratio: float = 0.8,
    fs: float = 50000.0,
    trim_ratio: float = 0.01,
    wear_agg: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for run in train_runs:
        X_run, y_run = _build_run_133_features(
            dataset_root=dataset_root,
            run_name=run,
            fs=fs,
            trim_ratio=trim_ratio,
            wear_agg=wear_agg,
        )
        end = _train_coverage_end(
            T=X_run.shape[0], seq_len=seq_len, pred_len=pred_len, split_ratio=split_ratio
        )
        xs.append(X_run[:end])
        ys.append(y_run[:end])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def select_features_l1_from_raw(
    dataset_root: str,
    train_runs: Sequence[str] = ("c1", "c4"),
    seq_len: int = 96,
    pred_len: int = 16,
    split_ratio: float = 0.8,
    fs: float = 50000.0,
    trim_ratio: float = 0.01,
    wear_agg: str = "max",
    cv: int = 5,
    random_state: int = 2026,
    coef_threshold: float = 1e-8,
) -> Dict[str, object]:
    X_train, y_train = build_train_matrix_from_raw(
        dataset_root=dataset_root,
        train_runs=train_runs,
        seq_len=seq_len,
        pred_len=pred_len,
        split_ratio=split_ratio,
        fs=fs,
        trim_ratio=trim_ratio,
        wear_agg=wear_agg,
    )

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lasso", LassoCV(cv=cv, random_state=random_state, max_iter=20000, n_alphas=100)),
        ]
    )
    model.fit(X_train, y_train)
    coef = np.abs(model.named_steps["lasso"].coef_)
    selected_idx = np.where(coef > coef_threshold)[0]
    if selected_idx.size == 0:
        selected_idx = np.array([int(np.argmax(coef))], dtype=np.int64)

    return {
        "method": "l1_lasso",
        "selected_idx_0based": selected_idx,
        "selected_feat_names_1based": [f"Feat_{int(i) + 1}" for i in selected_idx],
        "best_alpha": float(model.named_steps["lasso"].alpha_),
        "coef_abs": coef,
    }


def save_keep_features_txt(selected_idx_0based: Sequence[int], output_txt: str) -> None:
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for idx in np.unique(np.asarray(selected_idx_0based, dtype=np.int64)):
            f.write(f"Feat_{int(idx) + 1}\n")
