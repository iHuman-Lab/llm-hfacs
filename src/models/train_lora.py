import argparse
import csv
import os
import re
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL = os.environ.get("HFACS_MODEL_PATH", "Qwen/Qwen3-0.6B")
DEFAULT_DATA = Path("data/processed/hfacs_train.csv")
DEFAULT_PROMPTS = Path("prompts/io.yaml")
DEFAULT_OUTPUT = Path("outputs/hfacs_lora")
DEFAULT_MAX_LENGTH = 2048  


def parse_args():
    parser = argparse.ArgumentParser(description="Train a LoRA adapter for HFACS YES/NO classification")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model ID or local model directory / 模型名称或路径")
    parser.add_argument("--train-data", type=Path, default=DEFAULT_DATA, help="Expert-labeled CSV / 专家标注数据")
    parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPTS, help="HFACS IO YAML / 提示词文件")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Adapter/checkpoint output / 输出目录")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH, help="Max tokens per sample / 每条样本最大 token 数")
    parser.add_argument("--predict-data", type=Path, default=None, help="Optional CSV of reports to classify after training; columns: report, optional report_id")
    parser.add_argument("--predictions-output", type=Path, default=Path("outputs/hfacs_lora/predictions.csv"), help="CSV containing one YES/NO column per HFACS factor")
    parser.add_argument("--check-prompts", action="store_true", help="Check YAML only; no model/GPU/data needed / 只检查 Prompt")
    parser.add_argument("--check-data", action="store_true", help="Check YAML and CSV only; no model/GPU needed / 只检查数据")
    return parser.parse_args()


def load_prompt_templates(path):
    with path.open("r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"Expected a nonempty YAML mapping in: {path}")
    for factor, template in templates.items():
        if not isinstance(factor, str) or not isinstance(template, str) or "{context}" not in template:
            raise ValueError(f"Invalid prompt or missing {{context}} for factor: {factor!r}")
    return templates



def load_training_rows(path, templates):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing labeled CSV: {path}. Expected columns: report,factor,label. "
            "Real training requires expert labels."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"report", "factor", "label"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns: {sorted(required)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Training CSV is empty")
    for line, row in enumerate(rows, start=2):
        if not row["report"] or not row["report"].strip():
            raise ValueError(f"Empty report on CSV line {line}")
        if not row["factor"] or row["factor"].strip() not in templates:
            raise ValueError(f"Unknown/disabled HFACS factor on CSV line {line}: {row['factor']!r}")
        if not row["label"] or row["label"].strip().lower() not in {"yes", "no"}:
            raise ValueError(f"Expected yes/no on CSV line {line}: {row['label']!r}")
    return rows


def answer_end_token_id(tokenizer):
    """Qwen chat uses <|im_end|>; fall back to the tokenizer EOS for other models.

    Qwen 聊天回复优先使用 <|im_end|>，其他模型退回 EOS。
    """
    vocab = tokenizer.get_vocab()
    if "<|im_end|>" in vocab:
        return vocab["<|im_end|>"]
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer has neither <|im_end|> nor eos_token_id")
    return tokenizer.eos_token_id


class HFACSDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer, templates, max_length):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if tokenizer.chat_template is None:
            raise ValueError("This implementation requires a model tokenizer with a chat template")
        self.samples = []
        end_id = answer_end_token_id(tokenizer)

        for index, row in enumerate(rows, start=1):
            factor = row["factor"].strip()
            answer = row["label"].strip().upper()
            prompt_text = templates[factor].replace("{context}", row["report"])


            chat_kwargs = {"enable_thinking": False} if "enable_thinking" in str(tokenizer.chat_template) else {}
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            )
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if not answer_ids:
                raise ValueError(f"Empty answer tokens for sample {index}")

            total_length = len(prompt_ids) + len(answer_ids) + 1
            if total_length > max_length:
         
                raise ValueError(
                    f"Sample {index} ({factor}) has {total_length} tokens > {max_length}. "
                    "Long-report handling is not implemented; adjust the plan after inspecting real reports."
                )

            input_ids = prompt_ids + answer_ids + [end_id]
            labels = [-100] * len(prompt_ids) + answer_ids + [end_id]
            self.samples.append({
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]



def load_prediction_reports(path):
    """Provisional prediction CSV: report required, report_id optional.

    预测数据格式暂定；它与带有专家标签的训练数据分开。
    """
    if not path.is_file():
        raise FileNotFoundError(f"Missing prediction CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "report" not in reader.fieldnames:
            raise ValueError("Prediction CSV requires a 'report' column")
        reports = []
        for index, row in enumerate(reader, start=1):
            report = (row.get("report") or "").strip()
            if not report:
                raise ValueError(f"Empty prediction report on data row {index}")
            reports.append({
                "report_id": (row.get("report_id") or "").strip() or f"Report_{index:03d}",
                "report": report,
            })
    if not reports:
        raise ValueError("Prediction CSV is empty")
    return reports


def save_predictions(model, tokenizer, templates, reports, output_path, max_length):
    """One row per report; one YES/NO column per active IO factor.

    仅保存模型预测，不计算评价指标；标签不来自训练数据。
    """
    model.eval()
    device = next(model.parameters()).device
    factors = list(templates)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["report_id", *factors])
        writer.writeheader()
        for index, item in enumerate(reports, start=1):
            result = {"report_id": item["report_id"]}
            for factor in factors:
                prompt_text = templates[factor].replace("{context}", item["report"])
                chat_kwargs = {"enable_thinking": False} if "enable_thinking" in str(tokenizer.chat_template) else {}
                prompt = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_text}],
                    tokenize=False, add_generation_prompt=True, **chat_kwargs,
                )
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                if inputs["input_ids"].shape[1] + 5 > max_length:
                    raise ValueError(
                        f"Prediction report {index}, factor {factor}: prompt exceeds max_length={max_length}. "
                        "Long-report handling has not been implemented."
                    )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=answer_end_token_id(tokenizer),
                        use_cache=True,
                    )
                answer_ids = generated[0, inputs["input_ids"].shape[1]:]
                answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip().upper()
                # Do not silently turn invalid outputs into NO.
                if not re.fullmatch(r"(?:YES|NO)[.!]?", answer):
                    raise ValueError(
                        f"Invalid prediction for report {item['report_id']}, factor {factor}: {answer!r}. "
                        "Expected YES or NO."
                    )
                result[factor] = answer.rstrip(".!")
            writer.writerow(result)
            print(f"Predicted report {index}/{len(reports)}: {item['report_id']}", flush=True)
    print(f"HFACS YES/NO predictions saved to: {output_path}", flush=True)


def main():
    config = parse_args()
    templates = load_prompt_templates(config.prompt_file)
    print(f"Loaded {len(templates)} active HFACS IO prompts.", flush=True)
    if config.check_prompts:
        print("Prompt check passed / Prompt 检查通过")
        return

    rows = load_training_rows(config.train_data, templates)
    print(f"Loaded {len(rows)} labeled training rows.", flush=True)
    if config.check_data:
        print("Data check passed / 数据检查通过 (training not started)")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Training requires a configured GPU.")
    print(f"Training GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"Base model: {config.model}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer lacks pad_token and eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    dataset = HFACSDataset(rows, tokenizer, templates, max_length=config.max_length)
    print(f"Prepared {len(dataset)} tokenized samples.", flush=True)

    model = AutoModelForCausalLM.from_pretrained(config.model, torch_dtype=torch.float16)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        warmup_ratio=0.05,  
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
        gradient_checkpointing=True,  
        optim="adamw_torch",
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
        ),
    )
    trainer.train()

    final_path = config.output_dir / "final_adapter"
    final_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"LoRA adapter saved to: {final_path}", flush=True)
    print("Training finished. No validation metrics were computed / 尚未计算验证指标", flush=True)


    if config.predict_data is not None:
        reports = load_prediction_reports(config.predict_data)
        save_predictions(model, tokenizer, templates, reports, config.predictions_output, config.max_length)


if __name__ == "__main__":
    main()
