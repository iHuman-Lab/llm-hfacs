from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from features.utils import ensure_dir, load_dataset, resolve_feature_cols

TRAIN_PER_CLASS = 125  # 125 * 4 classes = 500
TEST_PER_CLASS = 50  # 50 * 4 classes = 200
INPUT_CSV = "data/intermediate/balanced_processed_output_undersampled.csv"
OUTPUT_DIR = "data/processed/lora_subfactors"
PROMPT_PATH = "src/llm/prompts/ASRS/ASRS_extract_and_classify_lora_subfactors.yaml"


def _build_label(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_cols = resolve_feature_cols(config, df)
    features = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    labels = pd.DataFrame(index=df.index)
    for col in feature_cols:
        labels[col] = (features[col] > 0).map({True: "Yes", False: "No"})

    final_class = df["category"]
    labels["Q1_Error"] = final_class.isin(["Error", "Both"]).map(
        {True: "Yes", False: "No"}
    )
    labels["Q2_Violation"] = final_class.isin(["Violation", "Both"]).map(
        {True: "Yes", False: "No"}
    )
    labels["Final_Class"] = final_class
    return labels


def _render_prompt(template: str, narrative: str) -> str:
    return template.replace("{narrative}", narrative)


def _build_messages(prm: dict, narrative: str, label: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": prm["system_prompt"]},
            {
                "role": "user",
                "content": _render_prompt(prm["user_prompt_template"], narrative),
            },
            {"role": "assistant", "content": json.dumps(label)},
        ]
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} rows to: {path}")


def build_lora_dataset(config: dict) -> None:
    narr_col = config["llm"]["narrative_column"]
    random_state = config["models"]["random_state"]

    df = load_dataset(INPUT_CSV)
    labels = _build_label(df, config)
    prm = yaml.safe_load(Path(PROMPT_PATH).read_text(encoding="utf-8"))

    out_dir = Path(OUTPUT_DIR)
    ensure_dir(str(out_dir))

    train_rows, test_rows = [], []
    for cls, group in df.groupby("category"):
        n = TRAIN_PER_CLASS + TEST_PER_CLASS
        sample = group.sample(n=n, random_state=random_state)
        train_idx = sample.index[:TRAIN_PER_CLASS]
        test_idx = sample.index[TRAIN_PER_CLASS:]

        for idx in train_idx:
            narrative = str(df.loc[idx, narr_col]).strip()
            train_rows.append(
                _build_messages(prm, narrative, labels.loc[idx].to_dict())
            )
        for idx in test_idx:
            narrative = str(df.loc[idx, narr_col]).strip()
            row = _build_messages(prm, narrative, labels.loc[idx].to_dict())
            row["original_index"] = int(idx)
            test_rows.append(row)

    _write_jsonl(train_rows, out_dir / "lora_train.jsonl")
    _write_jsonl(test_rows, out_dir / "lora_test.jsonl")


if __name__ == "__main__":
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    build_lora_dataset(config)
