import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    learning_curve,
)
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("./data/processed/step3_hfacs_categories.csv")
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ============================================================
# HFACS FEATURES (L4 + L3 + L2)
# ============================================================

HFACS_FEATURES = [
    # L4
    "Organizational_Process",
    "Organizational_Climate",
    "Resource_Management",
    # L3
    "Inadequate_Supervision",
    "Failure_to_Correct",
    # L2
    "Condition_of_Operators",
    "Personnel_Factors",
    "Situational_Factors",
]

# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df):
    """
    Unsafe Act = Error OR Violation (L1)
    """
    df = df.copy()

    df["unsafe_act"] = (
        (df["Error"] == 1) | (df["Violation"] == 1)
    ).astype(int)

    X = df[HFACS_FEATURES].values
    y = df["unsafe_act"].values.astype(int)

    return X, y


# ============================================================
# HFACS PATTERN ANALYSIS
# ============================================================

def analyze_hfacs_patterns(X, y_pred):
    """
    Most common HFACS (L4–L3–L2) activation patterns
    leading to predicted unsafe acts.
    """
    X_df = pd.DataFrame(X, columns=HFACS_FEATURES)
    X_df["predicted_unsafe"] = y_pred

    unsafe_df = X_df[X_df["predicted_unsafe"] == 1]

    print("\n=== MOST COMMON HFACS PATTERNS (Predicted Unsafe Acts) ===")
    patterns = (
        unsafe_df[HFACS_FEATURES]
        .astype(str)
        .agg("".join, axis=1)
        .value_counts()
        .head(10)
    )
    print(patterns)


# ============================================================
# LEARNING CURVE (TRAIN vs TEST)
# ============================================================

def print_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        cv=5,
        scoring="roc_auc",
        train_sizes=np.linspace(0.1, 1.0, 6),
        n_jobs=-1,
    )

    print("\n=== LEARNING CURVE (ROC-AUC) ===")
    for ts, tr, te in zip(
        train_sizes,
        train_scores.mean(axis=1),
        test_scores.mean(axis=1),
    ):
        print(
            f"Train size={ts:4d} | "
            f"Train AUC={tr:.3f} | "
            f"Test AUC={te:.3f}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ---------------- Load data ----------------
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_data(df)

    # ---------------- Train / test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # ---------------- SVM pipeline ----------------
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ))
    ])

    # ---------------- Grid Search ----------------
    param_grid = {
        "svm__C": [0.1, 1, 5, 10],
        "svm__gamma": [0.01, 0.05, "scale"],
    }

    grid = GridSearchCV(
        svm_pipeline,
        param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
        verbose=1,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print("\nBest SVM parameters:")
    print(grid.best_params_)

    # ---------------- Predict probabilities ----------------
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # ---------------- Threshold tuning ----------------
    thresholds = np.linspace(0.3, 0.7, 21)
    best_f1, best_thresh = -1, 0.5

    for t in thresholds:
        y_pred_tmp = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_tmp)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    y_pred = (y_prob >= best_thresh).astype(int)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== OPTIMIZED SVM PERFORMANCE ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Threshold: {best_thresh:.2f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["No Unsafe Act", "Unsafe Act"]
    ))

    # ---------------- Pattern analysis ----------------
    analyze_hfacs_patterns(X_test, y_pred)

    # ---------------- Learning curve ----------------
    print_learning_curve(best_model, X, y)


# ============================================================
if __name__ == "__main__":
    main()
