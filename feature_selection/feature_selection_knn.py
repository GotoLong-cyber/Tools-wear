from typing import Dict, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_selection_l1 import build_train_matrix_from_raw


def select_features_knn_from_raw(
    dataset_root: str,
    train_runs: Sequence[str] = ("c1", "c4"),
    seq_len: int = 96,
    pred_len: int = 16,
    split_ratio: float = 0.8,
    fs: float = 50000.0,
    trim_ratio: float = 0.01,
    wear_agg: str = "max",
    random_state: int = 2026,
    prefilter_top_k: int = 40,
    select_k: int = 20,
    n_neighbors: int = 7,
    cv: int = 5,
    direction: str = "forward",
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

    # Pre-filter to reduce SFS search space.
    et = ExtraTreesRegressor(n_estimators=500, random_state=random_state, n_jobs=-1)
    et.fit(X_train, y_train)
    imp = et.feature_importances_
    p = max(1, min(int(prefilter_top_k), X_train.shape[1]))
    pre_idx = np.argsort(imp)[::-1][:p]

    X_pre = X_train[:, pre_idx]
    k = max(1, min(int(select_k), X_pre.shape[1]))

    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")),
        ]
    )
    sfs = SequentialFeatureSelector(
        estimator=knn,
        n_features_to_select=k,
        direction=direction,
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=-1,
    )
    sfs.fit(X_pre, y_train)
    mask = sfs.get_support()
    selected_idx = pre_idx[np.where(mask)[0]]

    return {
        "method": "knn_wrapper_sfs",
        "selected_idx_0based": selected_idx,
        "selected_feat_names_1based": [f"Feat_{int(i) + 1}" for i in selected_idx],
        "prefilter_importances": imp,
    }
