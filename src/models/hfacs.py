from __future__ import annotations

import numpy as np
import pymc as pm
from features.balancing import balance_and_report
from features.utils import (
    ensure_dir,
    filter_tied_rows,
    get_hfacs_feature_cols,
    load_dataset,
    make_three_class_target_from_config,
    print_eval_metrics,
    save_predictions,
    stratified_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

CLASSES = ["Neither", "Error", "Violation"]


def bayesian(config):
    out_dir = config["paths"]["svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["bayes"]
    random_state = config["models"]["random_state"]

    df = load_dataset(config["paths"]["processed_csv"])

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y3 = make_three_class_target_from_config(df, config)
    df, y3 = filter_tied_rows(df, y3)

    X = df.loc[:, feature_cols].astype(float)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y3, test_size=cfg["test_size"], random_state=random_state
    )
    X_train, y_train = balance_and_report(X_train, y_train, "train")
    X_test, y_test = balance_and_report(X_test, y_test, "test")

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
            cfg["n_samples"],
            tune=cfg["tune"],
            cores=cfg["cores"],
            random_seed=random_state,
            progressbar=True,
        )

    betas_samples = trace.posterior["betas"].values.reshape(-1, n_classes, n_features)
    intercepts_samples = trace.posterior["intercepts"].values.reshape(-1, n_classes)

    theta_test = (
        np.einsum("nf,skf->snk", X_test_np, betas_samples)
        + intercepts_samples[:, np.newaxis, :]
    )
    pred_samples = theta_test.argmax(axis=-1)

    y_pred_idx = np.array(
        [
            np.bincount(pred_samples[:, i], minlength=n_classes).argmax()
            for i in range(pred_samples.shape[1])
        ]
    )
    y_pred = [CLASSES[i] for i in y_pred_idx]

    print("\nClassification report (0=Neither, 1=Error, 2=Violation):")
    print_eval_metrics(y_test, y_pred, CLASSES)
    save_predictions(y_test, y_pred, out_dir, "bayesian_predictions.csv")


def svm(config, df):
    out_dir = config["paths"]["svm_output_dir"]
    ensure_dir(out_dir)
    cfg = config["models"]["svm"]

    feature_cols = [c for c in get_hfacs_feature_cols(config) if c in df.columns]
    y3 = make_three_class_target_from_config(df, config)
    df, y3 = filter_tied_rows(df, y3)

    X = df.loc[:, feature_cols].astype(float)
    X_train, X_test, y_train, y_test = stratified_split(
        X, y3, test_size=cfg["test_size"], random_state=config["models"]["random_state"]
    )
    X_train, y_train = balance_and_report(X_train, y_train, "train")
    X_test, y_test = balance_and_report(X_test, y_test, "test")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "svc",
                SVC(
                    C=cfg["C"],
                    kernel=cfg["kernel"],
                    gamma=cfg["gamma"],
                    probability=False,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    print("\nClassification report (0=Neither, 1=Error, 2=Violation):")
    print_eval_metrics(y_test, y_pred)
    save_predictions(y_test, y_pred, out_dir, "svm_predictions.csv")
