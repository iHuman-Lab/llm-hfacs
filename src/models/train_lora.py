import argparse
import csv
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

# Base model: to be selected based on future experiments.
DEFAULT_MODEL = None

# Expert-labeled HFACS training data: to be provided by the advisor.
DEFAULT_DATA = None
DEFAULT_PROMPTS = Path("prompts/io.yaml")
DEFAULT_OUTPUT = Path("outputs/hfacs_lora")
DEFAULT_MAX_LENGTH = 2048  


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", type=Path, required=True)

    return parser.parse_args()

# Currently using the simplest Input-Output (IO) prompting strategy.
# The prompting strategy may be adjusted based on future experiments.
def load_prompt_templates(path):
    with path.open("r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)
    if not isinstance(templates, dict) or not templates:
        raise ValueError(f"Expected a nonempty YAML mapping in: {path}")
    for factor, template in templates.items():
        if not isinstance(factor, str) or not isinstance(template, str) or "{context}" not in template:
            raise ValueError(f"Invalid prompt or missing {{context}} for factor: {factor!r}")
    return templates


# Temporary CSV format: report, factor, label.
# Will be adapted once the actual HFACS training data is available.
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

# Use an end-of-response token so the model learns to stop after YES or NO.
def answer_end_token_id(tokenizer):
  # Currently designed for Qwen chat models using <|im_end|> to end a response.
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





def main():
    config = parse_args()

    # Load the current HFACS IO prompts.
    templates = load_prompt_templates(DEFAULT_PROMPTS)
    print(f"Loaded {len(templates)} active HFACS IO prompts.")

    # Load expert-labeled training data.
    rows = load_training_rows(config.train_data, templates)
    print(f"Loaded {len(rows)} labeled training rows.")

    # Check GPU availability.
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available. Training requires a configured GPU.")

    print(f"Training GPU: {torch.cuda.get_device_name(0)}")
    print(f"Base model: {config.model}")

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(config.model)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer lacks pad_token and eos_token")
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare training dataset.
    dataset = HFACSDataset(
        rows,
        tokenizer,
        templates,
        max_length=DEFAULT_MAX_LENGTH,
    )
    print(f"Prepared {len(dataset)} tokenized samples.")

    # Load the base model.
    model = AutoModelForCausalLM.from_pretrained(
        config.model,
        torch_dtype=torch.float16,
    )

    model.config.use_cache = False

    # Configure LoRA.
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    # Training configuration.
    args = TrainingArguments(
        output_dir=str(DEFAULT_OUTPUT),
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

    # Train the LoRA adapter.
    trainer.train()

    # Save the trained LoRA adapter for later evaluation.
    final_path = DEFAULT_OUTPUT / "final_adapter"
    final_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    print(f"LoRA adapter saved to: {final_path}")


if __name__ == "__main__":
    main()
