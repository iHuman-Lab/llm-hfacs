from __future__ import annotations

from pathlib import Path

import pandas as pd

from features.utils import ensure_dir

# Downloaded from Drive after running lora_inference_llama3_extract_classify.ipynb
INPUT_PATH = "data/processed/rf/rf_llm_features_50_per_class.csv"
OUTPUT_PATH = "data/processed/rf/llm_predicted_hfacs_features_50_per_class.csv"

FACTOR_COLUMNS = [
    "Situational_Factors",
    "Personnel_Factors",
    "Condition_of_Operators",
    "Inadequate_Supervision",
    "Failed_to_Correct_Problem",
    "Planned_Inappropriate_Operations",
    "Supervisory_Violation",
    "Organizational_Process",
    "Q1_Error",
    "Q2_Violation",
]


def convert_llm_predictions_to_rf_features() -> None:
    df = pd.read_csv(INPUT_PATH)

    out = df[["original_index", "category"]].copy()
    for col in FACTOR_COLUMNS:
        out[col] = (df[col] == "Yes").astype(int)
    out["llm_final_class"] = df["Final_Class"]

    out_path = Path(OUTPUT_PATH)
    ensure_dir(str(out_path.parent))
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to: {out_path}")


if __name__ == "__main__":
    convert_llm_predictions_to_rf_features()
