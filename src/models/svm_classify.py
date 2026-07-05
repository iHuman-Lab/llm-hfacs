from __future__ import annotations

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_and_report
from features.utils import (
    ensure_dir,
    get_hfacs_feature_cols,
    load_dataset,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)
from models.random_forest import make_four_class_target_from_config


def svm_classify(config):
    out_dir = config["paths"]["svm_classify_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["svm_classify"]
    random_state = config["models"]["random_state"]

    df = load_dataset(config["paths"]["rf_classify_input"])

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y4 = make_four_class_target_from_config(df, config)

    label_names = {0: "Neither", 1: "Error", 2: "Violation", 3: "Both"}
    print("\nClass distribution (all data):")
    for cls, name in label_names.items():
        print(f"  {name}: {(y4 == cls).sum()}")

    X = df.loc[:, feature_cols].astype(float)
    X, y4 = balance_and_report(X, y4, "full")
    X_train, X_test, y_train, y_test = stratified_split(
        X, y4, test_size=cfg["test_size"], random_state=random_state
    )

    print("\nClass distribution in train split:")
    for cls, count in y_train.value_counts().sort_index().items():
        print(f"  Class {cls} ({label_names[cls]}): {count} cases")
    print("\nClass distribution in test split:")
    for cls, count in y_test.value_counts().sort_index().items():
        print(f"  Class {cls} ({label_names[cls]}): {count} cases")

    # Two sub-grids: linear kernel (no gamma) and rbf kernel (with gamma).
    # class_weight='balanced' compensates for imbalance without discarding samples.
    param_grid = [
        {
            "svc__C": cfg["C"],
            "svc__kernel": ["linear"],
        },
        {
            "svc__C": cfg["C"],
            "svc__kernel": ["rbf"],
            "svc__gamma": cfg["gamma"],
        },
    ]

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(random_state=random_state, probability=False)),
        ]
    )

    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train, y_train)

    print(f"\nBest params: {grid.best_params_}")
    print(f"Best CV f1_macro: {grid.best_score_:.4f}")

    y_pred = grid.predict(X_test)
    print("\nClassification report (0=Neither, 1=Error, 2=Violation, 3=Both):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "svm_classify_predictions.csv")
