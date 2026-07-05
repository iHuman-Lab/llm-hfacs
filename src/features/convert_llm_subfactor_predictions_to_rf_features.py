from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from features.utils import ensure_dir, get_hfacs_feature_cols

# Downloaded from Drive after running lora_inference_llama3_subfactors_extract_classify.ipynb
INPUT_PATH = "data/processed/rf/rf_llm_subfactor_features_50_per_class.csv"
OUTPUT_PATH = "data/processed/rf/llm_predicted_subfactor_features_50_per_class.csv"


def convert_llm_subfactor_predictions_to_rf_features(config: dict) -> None:
    feature_cols = get_hfacs_feature_cols(config)
    df = pd.read_csv(INPUT_PATH)

    out = df[["original_index", "category"]].copy()
    for col in feature_cols:
        out[col] = (df[col] == "Yes").astype(int)
    out["llm_Q1_Error"] = (df["Q1_Error"] == "Yes").astype(int)
    out["llm_Q2_Violation"] = (df["Q2_Violation"] == "Yes").astype(int)
    out["llm_final_class"] = df["Final_Class"]

    out_path = Path(OUTPUT_PATH)
    ensure_dir(str(out_path.parent))
    out.to_csv(out_path, index=False)
    print(f"Saved {len(out)} rows to: {out_path}")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    convert_llm_subfactor_predictions_to_rf_features(config)
