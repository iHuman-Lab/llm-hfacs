from __future__ import annotations

import pandas as pd
import yaml

from features.metrics import sklearn_metrics
from features.utils import get_hfacs_feature_cols, print_eval_metrics

GROUND_TRUTH_PATH = "data/processed/rf/hfacs_category_features_50_per_class.csv"
PRED_PATH = "data/processed/rf/llm_predicted_subfactor_features_50_per_class.csv"


def compare_llm_subfactor_predictions(config: dict) -> None:
    feature_cols = get_hfacs_feature_cols(config)

    gt = pd.read_csv(GROUND_TRUTH_PATH)
    pred = pd.read_csv(PRED_PATH)
    merged = gt.merge(pred, on="original_index", suffixes=("_true", "_pred"))

    rows = []
    for col in feature_cols:
        metrics = sklearn_metrics(merged[f"{col}_true"], merged[f"{col}_pred"])
        rows.append({"factor": col, **metrics})
    report = pd.DataFrame(rows).set_index("factor")
    print("\nPer-factor metrics (true=ground truth, pred=LLM):\n")
    print(report.to_string())
    print("\nMacro-average:\n")
    print(report.astype(float).mean().to_string())

    q1_true = merged["category_true"].isin(["Error", "Both"]).astype(int)
    q1_pred = (merged["llm_Q1_Error"]).astype(int)
    q2_true = merged["category_true"].isin(["Violation", "Both"]).astype(int)
    q2_pred = (merged["llm_Q2_Violation"]).astype(int)

    print("\nQ1_Error:")
    print(sklearn_metrics(q1_true, q1_pred))
    print("\nQ2_Violation:")
    print(sklearn_metrics(q2_true, q2_pred))

    print("\nFinal_Class:")
    print_eval_metrics(
        merged["category_true"],
        merged["llm_final_class"],
        classes=["Neither", "Error", "Violation", "Both"],
    )


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    compare_llm_subfactor_predictions(config)
