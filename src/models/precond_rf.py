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


def precond_rf(config):
    out_dir = config["paths"]["rf_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["rf"]
    random_state = config["models"]["random_state"]
    train_per_class = cfg.get("train_per_class", 125)
    test_per_class = cfg.get("test_per_class", 50)
    use_llm = cfg.get("use_llm_features", False)

    df = load_dataset(config["paths"]["rf_classify_input"])

    hfacs_cols = get_hfacs_feature_cols(config)
    bare_cols = [c for c in hfacs_cols if c in df.columns]
    llm_cols = ["llm_" + c for c in hfacs_cols if "llm_" + c in df.columns]

    if not bare_cols:
        raise ValueError("No ground truth HFACS columns found in dataset.")
    if use_llm and not llm_cols:
        raise ValueError("use_llm_features=true but no llm_* columns found in dataset.")

    feature_cols = llm_cols if use_llm else bare_cols
    mode_label = "LLM-extracted" if use_llm else "ground truth"
    y = df["category"]

    print(
        f"\n[RF-Preconditions] Features: {mode_label} ({len(feature_cols)} columns) | Labels: ground truth"
    )
    print("\nClass distribution (all data):")
    for cls, count in y.value_counts().sort_index().items():
        print(f"  {cls}: {count}")

    train_idx, test_idx = [], []
    for cls in sorted(y.unique()):
        idx = y[y == cls].index.tolist()
        sampled = pd.Series(idx).sample(
            n=train_per_class + test_per_class, random_state=random_state
        )
        train_idx.extend(sampled.iloc[:train_per_class].tolist())
        test_idx.extend(sampled.iloc[train_per_class:].tolist())

    X_train = df.loc[train_idx, feature_cols].astype(float)
    y_train = y.loc[train_idx]
    X_test = df.loc[test_idx, feature_cols].astype(float)
    y_test = y.loc[test_idx]

    print(f"\nTrain: {len(X_train)} samples ({train_per_class} per class)")
    print(f"Test:  {len(X_test)} samples ({test_per_class} per class)")

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
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "rf_preconditions_predictions.csv")
