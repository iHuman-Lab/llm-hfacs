from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from features.utils import (
    ensure_dir,
    get_hfacs_feature_cols,
    load_dataset,
    print_eval_metrics,
    save_predictions,
)


def make_four_class_target_from_config(df: pd.DataFrame, config: dict) -> pd.Series:
    error_cols = list(config["hfacs_categories"]["Error"])
    viol_cols = list(config["hfacs_categories"]["Violation"])

    def _active(cols):
        present = [c for c in cols if c in df.columns]
        if not present:
            return pd.Series(False, index=df.index)
        return (df[present].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(
            axis=1
        )

    has_error = _active(error_cols)
    has_viol = _active(viol_cols)

    y = pd.Series(0, index=df.index, dtype=int)
    y[has_error & ~has_viol] = 1  # Error only
    y[~has_error & has_viol] = 2  # Violation only
    y[has_error & has_viol] = 3  # Both
    return y


def random_forest(config):
    out_dir = config["paths"]["rf_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["rf"]
    random_state = config["models"]["random_state"]

    df = load_dataset(config["paths"]["rf_classify_input"])

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y4 = make_four_class_target_from_config(df, config)

    label_names = {0: "Neither", 1: "Error", 2: "Violation", 3: "Both"}
    print("\nClass distribution (all data):")
    for cls, name in label_names.items():
        print(f"  {name}: {(y4 == cls).sum()}")

    X = df.loc[:, feature_cols].astype(float)

    train_dfs, test_dfs = [], []
    for cls in y4.unique():
        idx = y4[y4 == cls].index
        sample = X.loc[idx].sample(n=125 + 50, random_state=random_state)
        train_dfs.append(sample.iloc[:125])
        test_dfs.append(sample.iloc[125:])
    X_train = pd.concat(train_dfs)
    X_test = pd.concat(test_dfs)
    y_train = y4.loc[X_train.index]
    y_test = y4.loc[X_test.index]

    print(f"\nTrain: {len(X_train)} samples (125 per class)")
    print(f"Test:  {len(X_test)} samples (50 per class)")

    param_dist = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "min_samples_leaf": cfg["min_samples_leaf"],
        "max_features": cfg["max_features"],
    }

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = RandomizedSearchCV(
        RandomForestClassifier(random_state=random_state, n_jobs=-1),
        param_dist,
        n_iter=cfg["n_iter"],
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print("\nClassification report (0=Neither, 1=Error, 2=Violation, 3=Both):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "rf_classify_predictions.csv")
