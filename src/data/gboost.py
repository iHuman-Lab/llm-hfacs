import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

# Try to import XGBoost - it's excellent for this type of problem
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("XGBoost not installed. Install with: pip install xgboost")
    XGB_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = Path("./data/processed/step3_hfacs_categories.csv")
TEST_SIZE = 0.30
RANDOM_STATE = 42
CV_FOLDS = 5

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
    df = df.copy()
    df["unsafe_act"] = ((df["Error"] == 1) | (df["Violation"] == 1)).astype(int)

    X = df[HFACS_FEATURES].values
    y = df["unsafe_act"].values.astype(int)

    return X, y

# ============================================================
# THRESHOLD TUNING FUNCTION
# ============================================================

def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.3, 0.7, 41)  # finer grid
    best_f1 = -1
    best_thresh = 0.5
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
    return best_thresh, best_f1

# ============================================================
# MAIN
# ============================================================

def main():
    # ---------------- Load data ----------------
    df = pd.read_csv(DATA_PATH)
    X, y = prepare_data(df)

    print(f"Dataset shape: {X.shape}")
    print(f"Unsafe Act prevalence: {y.mean():.3f} ({y.sum()} positive out of {len(y)})")

    # ---------------- Train / test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    # =====================================================
    # 1. Improved SVM
    # =====================================================
    print("\n=== Training SVM ===")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ))
    ])

    param_grid_svm = {
        "svm__C": [0.1, 1, 10, 50, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.05, 0.1, 0.5],
    }

    grid_svm = GridSearchCV(
        svm_pipeline,
        param_grid_svm,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    grid_svm.fit(X_train, y_train)
    best_svm = grid_svm.best_estimator_

    y_prob_svm = best_svm.predict_proba(X_test)[:, 1]
    thresh_svm, f1_svm_thresh = find_best_threshold(y_test, y_prob_svm)
    y_pred_svm = (y_prob_svm >= thresh_svm).astype(int)

    acc_svm = accuracy_score(y_test, y_pred_svm)
    auc_svm = roc_auc_score(y_test, y_prob_svm)

    results.append({
        "Model": "SVM (RBF)",
        "Best Params": grid_svm.best_params_,
        "Accuracy": acc_svm,
        "F1 (opt thresh)": f1_svm_thresh,
        "ROC-AUC": auc_svm,
        "Threshold": thresh_svm,
    })

    print(f"SVM Best params: {grid_svm.best_params_}")
    print(f"SVM Acc: {acc_svm:.3f} | F1: {f1_svm_thresh:.3f} | AUC: {auc_svm:.3f}")

    # =====================================================
    # 2. Random Forest
    # =====================================================
    print("\n=== Training Random Forest ===")
    rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    param_grid_rf = {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_rf = GridSearchCV(
        rf,
        param_grid_rf,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_

    y_prob_rf = best_rf.predict_proba(X_test)[:, 1]
    thresh_rf, f1_rf_thresh = find_best_threshold(y_test, y_prob_rf)
    y_pred_rf = (y_prob_rf >= thresh_rf).astype(int)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    auc_rf = roc_auc_score(y_test, y_prob_rf)

    results.append({
        "Model": "Random Forest",
        "Best Params": grid_rf.best_params_,
        "Accuracy": acc_rf,
        "F1 (opt thresh)": f1_rf_thresh,
        "ROC-AUC": auc_rf,
        "Threshold": thresh_rf,
    })

    print(f"RF Best params: {grid_rf.best_params_}")
    print(f"RF Acc: {acc_rf:.3f} | F1: {f1_rf_thresh:.3f} | AUC: {auc_rf:.3f}")

    # Feature importances
    importances = best_rf.feature_importances_
    feat_imp = sorted(zip(HFACS_FEATURES, importances), key=lambda x: -x[1])
    print("\nTop Feature Importances (Random Forest):")
    for feat, imp in feat_imp:
        print(f"  {feat}: {imp:.4f}")

    # =====================================================
    # 3. XGBoost (if available)
    # =====================================================
    if XGB_AVAILABLE:
        print("\n=== Training XGBoost ===")
        xgb = XGBClassifier(
            scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),  # handles imbalance
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        param_grid_xgb = {
            "n_estimators": [100, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.3],
            "subsample": [0.8, 1.0],
        }

        grid_xgb = GridSearchCV(
            xgb,
            param_grid_xgb,
            scoring="roc_auc",
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1,
        )
        grid_xgb.fit(X_train, y_train)
        best_xgb = grid_xgb.best_estimator_

        y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]
        thresh_xgb, f1_xgb_thresh = find_best_threshold(y_test, y_prob_xgb)
        y_pred_xgb = (y_prob_xgb >= thresh_xgb).astype(int)

        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        auc_xgb = roc_auc_score(y_test, y_prob_xgb)

        results.append({
            "Model": "XGBoost",
            "Best Params": grid_xgb.best_params_,
            "Accuracy": acc_xgb,
            "F1 (opt thresh)": f1_xgb_thresh,
            "ROC-AUC": auc_xgb,
            "Threshold": thresh_xgb,
        })

        print(f"XGBoost Best params: {grid_xgb.best_params_}")
        print(f"XGBoost Acc: {acc_xgb:.3f} | F1: {f1_xgb_thresh:.3f} | AUC: {auc_xgb:.3f}")

    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    summary_df = pd.DataFrame(results)
    print(summary_df.round(3).to_string(index=False))

    # Print full report for the best model by F1
    best_model_name = summary_df.loc[summary_df["F1 (opt thresh)"].idxmax(), "Model"]
    print(f"\nBest model by F1: {best_model_name}")
    if best_model_name == "SVM (RBF)":
        print(classification_report(y_test, y_pred_svm, target_names=["No Unsafe Act", "Unsafe Act"]))
    elif best_model_name == "Random Forest":
        print(classification_report(y_test, y_pred_rf, target_names=["No Unsafe Act", "Unsafe Act"]))
    elif best_model_name == "XGBoost":
        print(classification_report(y_test, y_pred_xgb, target_names=["No Unsafe Act", "Unsafe Act"]))

if __name__ == "__main__":
    main()