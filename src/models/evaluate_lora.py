
import argparse
import csv
from pathlib import Path
import json

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

DEFAULT_MODEL = None
DEFAULT_TEST_DATA = None

# Use the same IO prompts as the training script.
DEFAULT_PROMPTS = Path("prompts/io.yaml")

# Load the LoRA adapter saved by train_lora.py.
DEFAULT_ADAPTER = Path("outputs/hfacs_lora/final_adapter")

# Save evaluation results separately from training outputs.
DEFAULT_OUTPUT = Path("outputs/hfacs_evaluation")

DEFAULT_MAX_LENGTH = 2048

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--test-data", type=Path, required=True)

    return parser.parse_args()

# Use the same Input-Output (IO) prompts as training.
def load_prompt_templates(path):
    with path.open("r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)

    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"Expected a nonempty YAML mapping in: {path}")

    for factor, template in templates.items():
        if (
            not isinstance(factor, str)
            or not isinstance(template, str)
            or "{context}" not in template
        ):
            raise ValueError(
                f"Invalid prompt or missing {{context}} for factor: {factor!r}"
            )

    return templates

# Temporary test CSV format: report, factor, label.
# The test data must be separate from the training data.
def load_test_rows(path, templates):
    if not path.is_file():
        raise FileNotFoundError(f"Missing test CSV: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        required = {"report", "factor", "label"}

        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")

        rows = list(reader)

    if not rows:
        raise ValueError("Test CSV is empty")

    for line, row in enumerate(rows, start=2):
        if not row["report"] or not row["report"].strip():
            raise ValueError(f"Empty report on CSV line {line}")

        if not row["factor"] or row["factor"].strip() not in templates:
            raise ValueError(f"Unknown HFACS factor on CSV line {line}")

        if not row["label"] or row["label"].strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Expected YES/NO on CSV line {line}")

    return rows

def load_model(model_path, adapter_path):
    # Load the tokenizer used during training.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer lacks pad_token and eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Load the original base model.
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
    )

    # Load the trained LoRA adapter.
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
    )

    model = model.to("cuda")
    model.eval()

    return model, tokenizer

def predict_label(model, tokenizer, report, factor, templates):
    # Build the same IO prompt used during training.
    prompt_text = templates[factor].replace("{context}", report)

    chat_kwargs = (
        {"enable_thinking": False}
        if "enable_thinking" in str(tokenizer.chat_template)
        else {}
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )

    # Convert the prompt into token IDs.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(model.device)

    if inputs["input_ids"].shape[1] >= DEFAULT_MAX_LENGTH:
        raise ValueError("Report exceeds the maximum input length.")

    # Generate the model's answer.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens.
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip().upper()

    # Accept only YES or NO as valid predictions.
    if answer in {"YES", "NO"}:
        return answer

    return "INVALID"

def evaluate(model, tokenizer, templates, rows):
    y_true = []
    y_pred = []
    results = []

    for index, row in enumerate(rows, start=1):
        report = row["report"]
        factor = row["factor"].strip()
        true_label = row["label"].strip().upper()

        # Predict without providing the expert label.
        prediction = predict_label(
            model,
            tokenizer,
            report,
            factor,
            templates,
        )

        y_true.append(true_label)
        y_pred.append(prediction)

        results.append({
            "report_id": row.get("report_id") or f"Report_{index:03d}",
            "factor": factor,
            "true_label": true_label,
            "prediction": prediction,
            "correct": prediction == true_label,
        })




    # Invalid predictions count as incorrect.
    accuracy = accuracy_score(y_true, y_pred)

    # Compute precision, recall, and F1 for YES and NO.
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["NO", "YES"],
        average=None,
        zero_division=0,
    )

    macro_f1 = float(f1.mean())

    # Include INVALID in the confusion matrix if it appears.
    prediction_labels = ["NO", "YES", "INVALID"]
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=prediction_labels,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": macro_f1,
        "invalid_predictions": y_pred.count("INVALID"),
        "precision_no": float(precision[0]),
        "recall_no": float(recall[0]),
        "f1_no": float(f1[0]),
        "precision_yes": float(precision[1]),
        "recall_yes": float(recall[1]),
        "f1_yes": float(f1[1]),
        "confusion_matrix": cm.tolist(),
    }

    return metrics, results


def evaluate_per_factor(results, templates):
    per_factor_metrics = {}

    for factor in templates:
        factor_rows = [
            row for row in results
            if row["factor"] == factor
        ]

        if not factor_rows:
            continue

        y_true = [row["true_label"] for row in factor_rows]
        y_pred = [row["prediction"] for row in factor_rows]

        accuracy = accuracy_score(y_true, y_pred)

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=["NO", "YES"],
            average=None,
            zero_division=0,
        )

        cm = confusion_matrix(
            y_true,
            y_pred,
            labels=["NO", "YES", "INVALID"],
        )

        per_factor_metrics[factor] = {
            "samples": len(factor_rows),
            "accuracy": float(accuracy),
            "macro_f1": float(f1.mean()),
            "invalid_predictions": y_pred.count("INVALID"),
            "precision_no": float(precision[0]),
            "recall_no": float(recall[0]),
            "f1_no": float(f1[0]),
            "precision_yes": float(precision[1]),
            "recall_yes": float(recall[1]),
            "f1_yes": float(f1[1]),
            "confusion_matrix": cm.tolist(),
        }

    return per_factor_metrics

def save_results(metrics, results, per_factor_metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save overall evaluation metrics.
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions for each test sample.
    predictions_path = output_dir / "predictions.csv"
    with predictions_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "report_id",
                "factor",
                "true_label",
                "prediction",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Evaluation metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")

    # Save evaluation metrics for each HFACS factor.
    per_factor_path = output_dir / "per_factor_metrics.csv"

    fieldnames = [
        "factor",
        "samples",
        "accuracy",
        "macro_f1",
        "invalid_predictions",
        "precision_no",
        "recall_no",
        "f1_no",
        "precision_yes",
        "recall_yes",
        "f1_yes",
    ]

    with per_factor_path.open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for factor, values in per_factor_metrics.items():
            writer.writerow({
                "factor": factor,
                **{key: values[key] for key in fieldnames[1:]},
            })

    print(f"Per-factor metrics saved to: {per_factor_path}")

def main():
    config = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    templates = load_prompt_templates(DEFAULT_PROMPTS)
    rows = load_test_rows(config.test_data, templates)

    model, tokenizer = load_model(
        config.model,
        config.adapter,
    )

    # Overall evaluation.
    metrics, results = evaluate(
        model,
        tokenizer,
        templates,
        rows,
    )

    # Per-factor evaluation.
    per_factor_metrics = evaluate_per_factor(
        results,
        templates,
    )

    # Save all results.
    save_results(
        metrics,
        results,
        per_factor_metrics,
        DEFAULT_OUTPUT,
    )

    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Overall Macro F1: {metrics['macro_f1']:.4f}")

    for factor, values in per_factor_metrics.items():
        print(
            f"{factor}: "
            f"Accuracy={values['accuracy']:.4f}, "
            f"Macro F1={values['macro_f1']:.4f}"
        )


if __name__ == "__main__":
    main()
