from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from features.utils import ensure_dir, load_dataset


def _build_all_labels(df: pd.DataFrame, config: dict, task: str) -> pd.DataFrame:
    skip = {"Error", "Violation"}

    if task == "extract_preconditions":
        labels = pd.DataFrame(index=df.index)
        for cat, cols in config["hfacs_categories"].items():
            if cat in skip:
                continue
            for col in cols:
                if col in df.columns:
                    labels[col] = (
                        pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
                    ).astype(int)
                else:
                    labels[col] = 0
        return labels

    final_class = df["category"]
    q1 = final_class.isin(["Error", "Both"]).map({True: "Yes", False: "No"})
    q2 = final_class.isin(["Violation", "Both"]).map({True: "Yes", False: "No"})

    if task == "classify_direct":
        return pd.DataFrame(
            {"Q1_Error": q1, "Q2_Violation": q2, "Final_Class": final_class},
            index=df.index,
        )

    # extract_classify: 8 HFACS top-level categories + Q1/Q2/Final_Class
    labels = pd.DataFrame(index=df.index)
    for cat, cols in config["hfacs_categories"].items():
        if cat in skip:
            continue
        valid = [c for c in cols if c in df.columns]
        present = (
            (df[valid].apply(pd.to_numeric, errors="coerce").fillna(0) > 0).any(axis=1)
            if valid
            else pd.Series(False, index=df.index)
        )
        labels[cat] = present.map({True: "Yes", False: "No"})
    labels["Q1_Error"] = q1
    labels["Q2_Violation"] = q2
    labels["Final_Class"] = final_class
    return labels


def _build_think_trace(label: dict) -> str:
    if "Final_Class" not in label:
        # extract_preconditions: all binary items
        lines = ["Checking each HFACS precondition factor:"]
        for key, val in label.items():
            lines.append(f"- {key}: {val}")
        return "\n".join(lines)

    # classify_direct or extract_classify
    factor_keys = [
        k for k in label if k not in ("Q1_Error", "Q2_Violation", "Final_Class")
    ]
    lines = []
    if factor_keys:
        lines.append("Checking each HFACS contributing-factor category:")
        for key in factor_keys:
            lines.append(f"- {key}: {label[key]}")
        lines.append("")
    lines.append("Classifying the unsafe act:")
    lines.append(f"- Q1_Error: {label['Q1_Error']}")
    lines.append(f"- Q2_Violation: {label['Q2_Violation']}")
    lines.append("")
    lines.append(
        f"Since Q1_Error={label['Q1_Error']} and Q2_Violation={label['Q2_Violation']}, "
        f"Final_Class={label['Final_Class']}."
    )
    return "\n".join(lines)


def _build_messages(prm: dict, narrative: str, label: dict) -> dict:
    trace = _build_think_trace(label)
    answer = f"<think>\n{trace}\n</think>\n\n{json.dumps(label)}"
    return {
        "messages": [
            {"role": "system", "content": prm["system_prompt"]},
            {
                "role": "user",
                "content": prm["user_prompt_template"].replace(
                    "{narrative}", narrative
                ),
            },
            {"role": "assistant", "content": answer},
        ]
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} rows → {path}")


def _push_to_hub(train_rows: list[dict], test_rows: list[dict], cfg: dict) -> None:
    dataset_id = cfg.get("hf_dataset_id", "")
    if not dataset_id:
        return
    from datasets import Dataset, DatasetDict

    ds = DatasetDict(
        {
            "train": Dataset.from_list(train_rows),
            "test": Dataset.from_list(test_rows),
        }
    )
    ds.push_to_hub(dataset_id, token=cfg.get("hf_token") or None)
    print(f"Pushed to HuggingFace Hub: {dataset_id}")


def build_lora_dataset(config: dict) -> None:
    cfg = config["lora_qwen3"]
    task = cfg["task"]
    narr_col = cfg["narrative_column"]
    random_state = config.get("models", {}).get("random_state", 42)

    df = load_dataset(cfg["labeled_data"])
    labels = _build_all_labels(df, config, task)
    prm = yaml.safe_load(Path(cfg["prompt_yaml"]).read_text(encoding="utf-8"))

    out_dir = Path(cfg["output_dir"])
    ensure_dir(str(out_dir))

    train_rows, test_rows = [], []
    for _, group in df.groupby(cfg["class_column"]):
        sample = group.sample(
            n=cfg["train_per_class"] + cfg["test_per_class"], random_state=random_state
        )
        for idx in sample.index[: cfg["train_per_class"]]:
            row = _build_messages(
                prm, str(df.loc[idx, narr_col]).strip(), labels.loc[idx].to_dict()
            )
            row["original_index"] = int(idx)
            row["split"] = "train"
            train_rows.append(row)
        for idx in sample.index[cfg["train_per_class"] :]:
            row = _build_messages(
                prm, str(df.loc[idx, narr_col]).strip(), labels.loc[idx].to_dict()
            )
            row["original_index"] = int(idx)
            row["split"] = "test"
            test_rows.append(row)

    _write_jsonl(train_rows, out_dir / "lora_train.jsonl")
    _write_jsonl(test_rows, out_dir / "lora_test.jsonl")
    _push_to_hub(train_rows, test_rows, cfg)


if __name__ == "__main__":
    cfg = yaml.safe_load(
        Path("configs/finetune_qwen3_config.yaml").read_text(encoding="utf-8")
    )
    needs_hfacs = cfg["lora_qwen3"]["task"] in (
        "extract_classify",
        "extract_preconditions",
    )
    if needs_hfacs:
        main_cfg = yaml.safe_load(
            Path("configs/config.yaml").read_text(encoding="utf-8")
        )
        cfg.update(main_cfg)
    build_lora_dataset(cfg)
