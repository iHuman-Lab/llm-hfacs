from __future__ import annotations

import numpy as np
import pymc as pm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features.balancing import balance_and_report
from features.utils import (
    build_dataset_XY,
    ensure_dir,
    make_four_class_target,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)

CLASSES = ["AE100 only", "AE200 only", "Both", "Neither"]


def ghfacs_svm(config, df, *, use_random_state=True):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_svm"]
    random_state = config["models"]["random_state"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    if use_random_state:
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=config["models"]["test_size"], random_state=random_state
        )
    else:
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["models"]["test_size"], stratify=stratify
        )

    assert set(y_train.unique()) == set(y.unique()), (
        f"Missing classes in train split: {set(y.unique()) - set(y_train.unique())}"
    )

    X_train, y_train = balance_and_report(X_train, y_train)
    X_test, y_test = balance_and_report(X_test, y_test)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_svm_predictions.csv")


def ghfacs_rf(config, df, *, use_random_state=True):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_rf"]
    random_state = config["models"]["random_state"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    if use_random_state:
        X_train, X_test, y_train, y_test = stratified_split(
            X, y, test_size=config["models"]["test_size"], random_state=random_state
        )
    else:
        stratify = y if y.nunique() > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["models"]["test_size"], stratify=stratify
        )

    X_train, y_train = balance_and_report(X_train, y_train)
    X_test, y_test = balance_and_report(X_test, y_test)

    param_grid = {
        "n_estimators": cfg["n_estimators"],
        "max_depth": cfg["max_depth"],
        "min_samples_split": cfg["min_samples_split"],
        "max_features": cfg["max_features"],
    }

    rs_kwarg = {"random_state": random_state} if use_random_state else {}
    n_splits = min(cfg["cv_splits"], int(y_train.value_counts().min()))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, **rs_kwarg)
    grid = GridSearchCV(
        RandomForestClassifier(**rs_kwarg),
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
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_rf_predictions.csv")


ghfacs_svm_balancing = ghfacs_svm
ghfacs_rf_balancing = ghfacs_rf

def ghfacs_svm_no_rs(config, df):
    ghfacs_svm(config, df, use_random_state=False)

def ghfacs_rf_no_rs(config, df):
    ghfacs_rf(config, df, use_random_state=False)


def ghfacs_svm_nonbalance(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_svm_nonbalance"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"], random_state=config["models"]["random_state"]
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(C=cfg["C"], kernel=cfg["kernel"], gamma=cfg["gamma"], probability=False)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_svm_nonbalance_predictions.csv")


def ghfacs_bayes(config, df):
    out_dir = config["paths"]["ghfacs_svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["ghfacs_bayes"]
    random_state = config["models"]["random_state"]

    X, y = build_dataset_XY(df, config["ghfacs"]["precondition_cols"], make_four_class_target(df))
    X_train, X_test, y_train, y_test = stratified_split(
        X, y, test_size=config["models"]["test_size"], random_state=random_state
    )
    X_train, y_train = balance_and_report(X_train, y_train)
    X_test, y_test = balance_and_report(X_test, y_test)

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    y_train_idx = y_train.map(class_to_idx).values.astype(int)

    X_train_np = X_train.values.astype(float)
    X_test_np = X_test.values.astype(float)
    n_features = X_train_np.shape[1]
    n_classes = len(CLASSES)

    with pm.Model():
        intercepts = pm.Normal("intercepts", mu=0, sigma=1, shape=(n_classes,))
        betas = pm.Normal("betas", mu=0, sigma=1, shape=(n_classes, n_features))

        theta = intercepts + pm.math.dot(X_train_np, betas.T)
        p = pm.Deterministic("p", pm.math.softmax(theta, axis=-1))

        pm.Categorical("y_obs", p=p, observed=y_train_idx)

        trace = pm.sample(
            cfg["n_samples"], tune=cfg["tune"], cores=cfg["cores"],
            random_seed=random_state, progressbar=True,
        )

    betas_samples = trace.posterior["betas"].values.reshape(-1, n_classes, n_features)
    intercepts_samples = trace.posterior["intercepts"].values.reshape(-1, n_classes)

    theta_test = (
        np.einsum("nf,skf->snk", X_test_np, betas_samples)
        + intercepts_samples[:, np.newaxis, :]
    )
    pred_samples = theta_test.argmax(axis=-1)

    y_pred_idx = np.array([
        np.bincount(pred_samples[:, i], minlength=n_classes).argmax()
        for i in range(pred_samples.shape[1])
    ])
    y_pred = [CLASSES[i] for i in y_pred_idx]

    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "ghfacs_bayes_predictions.csv")
