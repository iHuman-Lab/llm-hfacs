from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_dataset(path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    stratify = y if y.nunique() > 1 else None
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def build_dataset_XY(
    df: pd.DataFrame,
    feature_cols: List[str],
    y: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series]:
    cols = [c for c in feature_cols if c in df.columns]
    X = df[cols].notna().astype(int)
    return X, y


def print_eval_metrics(
    y_test: pd.Series,
    y_pred,
    classes: Optional[List[str]] = None,
) -> None:
    print("\nTest accuracy:", accuracy_score(y_test, y_pred))
    if classes:
        print(
            "\nConfusion matrix (rows=true, cols=pred):\n",
            confusion_matrix(y_test, y_pred, labels=classes),
        )
        print("\nClassification report:\n")
        print(classification_report(y_test, y_pred, labels=classes, digits=4))
    else:
        print(
            "\nConfusion matrix (rows=true, cols=pred):\n",
            confusion_matrix(y_test, y_pred),
        )
        print("\nClassification report:\n")
        print(classification_report(y_test, y_pred, digits=4))


def save_predictions(
    y_test: pd.Series,
    y_pred,
    out_dir: str,
    filename: str,
) -> None:
    pred_path = os.path.join(out_dir, filename)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(
        pred_path, index=False
    )
    print(f"\nSaved: {pred_path}")


def get_hfacs_feature_cols(config: dict) -> List[str]:
    _targets = {"Error", "Violation"}
    return [
        col
        for cat, subcats in config["hfacs_categories"].items()
        if cat not in _targets
        for col in subcats
    ]


def filter_tied_rows(
    df: pd.DataFrame,
    y3: pd.Series,
) -> Tuple:
    dropped = (y3 == -1).sum()
    if dropped:
        print(f"\nDropped full-tie rows: {dropped}")
    mask = y3 != -1
    return df.loc[mask].copy(), y3.loc[mask]


def print_class_distribution(y3: pd.Series) -> None:
    labels = {0: "Neither", 1: "Error", 2: "Violation"}
    print("\nClass distribution after scoring and grouping:")
    for group, count in y3.value_counts().sort_index().items():
        print(f"  {labels.get(group, str(group))} ({group}): {count}")


def make_three_class_target(
    df: pd.DataFrame,
    error_weights: Dict[str, float],
    viol_weights: Dict[str, float],
) -> pd.Series:
    """Build 3-class target: 0=Neither, 1=Error, 2=Violation, -1=drop (full tie).

    Rules:
    - Only Violation active            → 2
    - Only Error active                → 1
    - Neither active                   → 0
    - Both active, viol_count > err_count → 2
    - Both active, err_count > viol_count → 1
    - Both active, counts equal, viol_wsum > err_wsum → 2
    - Both active, counts equal, err_wsum > viol_wsum → 1
    - Both active, counts equal, weights equal → -1 (drop)
    """

    def _numeric(cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.DataFrame(0, index=df.index, columns=cols[:1] or ["_empty"])
        return df[present].apply(pd.to_numeric, errors="coerce").fillna(0)

    def _wsum(cols_weights):
        total = pd.Series(0.0, index=df.index)
        for c, w in cols_weights.items():
            if c in df.columns:
                total += (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0).astype(
                    float
                ) * w
        return total

    err_active = _numeric(list(error_weights)) > 0
    vio_active = _numeric(list(viol_weights)) > 0

    err_flag = err_active.any(axis=1).astype(int)
    vio_flag = vio_active.any(axis=1).astype(int)
    err_count = err_active.sum(axis=1)
    vio_count = vio_active.sum(axis=1)
    err_wsum = _wsum(error_weights)
    vio_wsum = _wsum(viol_weights)

    y3 = pd.Series(-2, index=df.index, dtype=int)

    # Simple cases
    y3[(err_flag == 0) & (vio_flag == 0)] = 0
    y3[(err_flag == 1) & (vio_flag == 0)] = 1
    y3[(err_flag == 0) & (vio_flag == 1)] = 2

    # Both active — resolve by count then weight
    both = (err_flag == 1) & (vio_flag == 1)
    y3[both & (vio_count > err_count)] = 2
    y3[both & (err_count > vio_count)] = 1
    remaining = both & (err_count == vio_count)
    y3[remaining & (vio_wsum > err_wsum)] = 2
    y3[remaining & (err_wsum > vio_wsum)] = 1
    y3[remaining & (err_wsum == vio_wsum)] = -1  # full tie → drop

    return y3


def make_three_class_target_from_config(
    df: pd.DataFrame,
    config: dict,
) -> pd.Series:
    error_weights = config["hfacs_categories"]["Error"]
    viol_weights = config["hfacs_categories"]["Violation"]
    return make_three_class_target(df, error_weights, viol_weights)


def make_four_class_target(df: pd.DataFrame) -> pd.Series:
    """Build 4-class target from AE100/AE200 ground-truth columns.

    Returns string labels: 'AE100 only', 'AE200 only', 'Both', 'Neither'.
    """
    ae100 = df["AE100"].notna()
    ae200 = df["AE200"].notna()
    y = pd.Series("Neither", index=df.index, dtype=object)
    y[ae100 & ~ae200] = "AE100 only"
    y[~ae100 & ae200] = "AE200 only"
    y[ae100 & ae200] = "Both"
    return y
