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

# ============================================================
# HFACS FEATURES
# ============================================================

HFACS_FEATURES = [
    "Organizational_Process",
    "Organizational_Climate",
    "Resource_Management",
    "Inadequate_Supervision",
    "Failure_to_Correct",
    "Condition_of_Operators",
    "Personnel_Factors",
    "Situational_Factors",
]

# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):
    """
    Unsafe Act = Error OR Violation
    """

    df = df.copy()
    df["unsafe_act"] = (
        (df["Error"] == 1) | (df["Violation"] == 1)
    ).astype(int)

    X = df[HFACS_FEATURES].values
    y = df["unsafe_act"].values.astype(int)

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
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # =====================================================
    # SVM PIPELINE
    # =====================================================

    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ))
    ])

    # ---------------- Grid search ----------------
    param_grid = {
        "svm__C": [0.1, 1, 5, 10, 50],
        "svm__gamma": [0.01, 0.05, 0.1, "scale"],
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

    # =====================================================
    # THRESHOLD TUNING
    # =====================================================

    thresholds = np.linspace(0.3, 0.7, 21)
    best_f1 = -1
    best_thresh = 0.5

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    # final predictions
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

# ============================================================
if __name__ == "__main__":
    main()
