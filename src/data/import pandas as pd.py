import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
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
TEST_SIZE = 0.30
RANDOM_STATE = 42
THRESHOLD = 0.40   # tuned threshold (not default 0.5)

# ============================================================
# HFACS FEATURES (MATCH YOUR DATA)
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
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Binary classification:
    Error = 1
    Violation = 0
    """

    def resolve_l1(row):
        if row["Error"] == 1:
            return 1
        if row["Violation"] == 1:
            return 0
        return np.nan

    df = df.copy()
    df["L1_label"] = df.apply(resolve_l1, axis=1)
    df = df.dropna(subset=["L1_label"])

    X = df[HFACS_FEATURES].values
    y = df["L1_label"].astype(int).values

    return X, y


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

    # =====================================================
    # SVM PIPELINE (IMPROVED)
    # =====================================================

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",   # IMPORTANT
            random_state=RANDOM_STATE,
        ))
    ])

    # ---------------- Hyperparameter tuning ----------------
    param_grid = {
        "svm__C": [0.1, 1, 10, 50],
        "svm__gamma": [0.01, 0.1, "scale"],
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
    y_pred = (y_prob >= THRESHOLD).astype(int)

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print("\n=== IMPROVED SVM PERFORMANCE (TEST SET) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"ROC-AUC  : {auc:.3f}")
    print(f"Threshold: {THRESHOLD}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["Violation", "Error"]
    ))


# ============================================================
if __name__ == "__main__":
    main()
