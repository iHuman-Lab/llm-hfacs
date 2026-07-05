"""Evaluate GPT predictions on the same test rows used in Qwen LoRA.

Control from configs/config.yaml:
  llm:
    eval_prompts: [classify_direct, ASRS_extract_and_classify]
  lora_qwen3:
    train_per_class: 125
    test_per_class: 50
"""

from __future__ import annotations

import pandas as pd
import yaml
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

CLASS_COL = "category"


def eval_gpt_on_test_set(config: dict) -> None:
    lora_cfg = config["lora_qwen3"]
    train_per_class = lora_cfg["train_per_class"]
    test_per_class = lora_cfg["test_per_class"]
    random_state = config.get("models", {}).get("random_state", 42)
    prompts = config["llm"].get("eval_prompts", [])

    base = pd.read_csv(config["paths"]["ghfacs_data_dir"] + "/balanced_700.csv")

    test_indices = []
    for _, group in base.groupby(CLASS_COL):
        sample = group.sample(
            n=train_per_class + test_per_class, random_state=random_state
        )
        test_indices.extend(sample.index[train_per_class:].tolist())

    y_true = base.loc[test_indices, CLASS_COL].reset_index(drop=True)
    print(f"Test set: {len(y_true)} samples ({test_per_class} per class)")

    for prompt in prompts:
        filename = f"balanced_700_LLM_Output_{prompt}.csv"
        path = Path(config["paths"]["llm_output_dir"]) / filename
        if not path.exists():
            print(f"\n[{prompt}] File not found — skipping: {path}")
            continue

        gpt = pd.read_csv(path)
        if "Final_Class" not in gpt.columns:
            print(
                f"\n[{prompt}] No Final_Class column — predictions missing or errored."
            )
            continue

        y_pred = gpt.loc[test_indices, "Final_Class"].reset_index(drop=True)
        print(f"\n{'='*60}")
        print(f"GPT — {prompt} | {len(y_true)} test samples")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    config = yaml.safe_load(Path("configs/config.yaml").read_text())
    eval_gpt_on_test_set(config)
