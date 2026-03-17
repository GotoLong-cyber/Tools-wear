from typing import Dict, Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from feature_selection_l1 import build_train_matrix_from_raw


def select_features_tree_from_raw(
    dataset_root: str,
    train_runs: Sequence[str] = ("c1", "c4"),
    seq_len: int = 96,
    pred_len: int = 16,
    split_ratio: float = 0.8,
    fs: float = 50000.0,
    trim_ratio: float = 0.01,
    wear_agg: str = "max",
    random_state: int = 2026,
    n_estimators: int = 600,
    max_depth: int = 8,
    top_k: int = 60,
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

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    imp = rf.feature_importances_

    k = max(1, min(int(top_k), X_train.shape[1]))
    selected_idx = np.argsort(imp)[::-1][:k]

    return {
        "method": "tree_random_forest",
        "selected_idx_0based": selected_idx,
        "selected_feat_names_1based": [f"Feat_{int(i) + 1}" for i in selected_idx],
        "feature_importances": imp,
    }
