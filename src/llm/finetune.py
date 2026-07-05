"""
LoRA fine-tuning pipeline for HFACS classification.
Run on Google Colab (GPU required):
  pip install peft transformers trl accelerate bitsandbytes datasets huggingface_hub
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_training_data(config: dict) -> pd.DataFrame:
    ft = config["finetune"]
    path = Path(ft["labeled_data"])
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)

    narr_col = ft["narrative_column"]
    class_col = ft["class_column"]
    df = df.dropna(subset=[narr_col, class_col]).reset_index(drop=True)

    print(f"[data] Training samples: {len(df)}")
    print(f"[data] Class distribution:\n{df[class_col].value_counts().to_string()}")
    return df


def load_prompts(config: dict) -> tuple[str, str | None]:
    """Return (system_prompt, user_template). Reads from prompt YAML if configured."""
    ft = config["finetune"]
    yaml_path = ft.get("prompt_yaml")
    if yaml_path:
        prompts = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
        system_prompt = prompts.get("system_prompt", "").strip()
        user_template = prompts.get("user_prompt_template", "").strip() or None
        if system_prompt:
            return system_prompt, user_template

    # fallback: generic prompt built from classes list
    classes_str = ", ".join(ft["classes"])
    system_prompt = (
        "You are an expert in aviation safety and the HFACS framework. "
        f"Classify the narrative into exactly one of: {classes_str}. "
        "Respond with only the category name, nothing else."
    )
    return system_prompt, None


def format_sample(
    narrative: str, label: str, system_prompt: str, user_template: str | None
) -> str:
    if user_template:
        user_msg = user_template.replace("{narrative}", narrative).replace(
            "{context}", narrative
        )
    else:
        user_msg = f"Classify this accident narrative:\n\n{narrative}"
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{label}"
        "<|eot_id|>"
    )


def build_hf_dataset(
    df: pd.DataFrame, config: dict, system_prompt: str, user_template: str | None
):
    from datasets import Dataset

    ft = config["finetune"]
    narr_col = ft["narrative_column"]
    class_col = ft["class_column"]

    df = df.copy()
    df["text"] = [
        format_sample(
            str(r[narr_col]).strip(),
            str(r[class_col]).strip(),
            system_prompt,
            user_template,
        )
        for _, r in df.iterrows()
    ]
    return Dataset.from_pandas(df[["text"]].reset_index(drop=True))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model_with_lora(config: dict):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login

    ft = config["finetune"]
    lora_cfg = ft["lora"]
    hf_token = ft["hf_token"]

    login(token=hf_token)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    base_model_id = ft["base_model"]
    print(f"[model] Loading {base_model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        target_modules=lora_cfg["target_modules"],
        lora_dropout=float(lora_cfg["dropout"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train(model, tokenizer, train_dataset, config: dict):
    import torch
    from trl import SFTTrainer, SFTConfig

    ft = config["finetune"]
    tr = ft["training"]

    tokenizer.model_max_length = int(tr["max_seq_len"])

    # only compute loss on the label token — mask everything up to and including
    # the assistant header so the model only learns to predict the class word
    response_template_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n", add_special_tokens=False
    )

    class _ResponseOnlyCollator:
        def __call__(self, features):
            max_len = max(len(f["input_ids"]) for f in features)
            input_ids, masks = [], []
            for f in features:
                pad = max_len - len(f["input_ids"])
                input_ids.append(f["input_ids"] + [tokenizer.pad_token_id] * pad)
                masks.append(
                    f.get("attention_mask", [1] * len(f["input_ids"])) + [0] * pad
                )
            input_ids = torch.tensor(input_ids)
            masks = torch.tensor(masks)
            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            tpl_len = len(response_template_ids)
            for i in range(labels.shape[0]):
                seq = input_ids[i].tolist()
                for j in range(len(seq) - tpl_len, -1, -1):
                    if seq[j : j + tpl_len] == response_template_ids:
                        labels[i, : j + tpl_len] = -100
                        break
            return {"input_ids": input_ids, "attention_mask": masks, "labels": labels}

    args = SFTConfig(
        output_dir=ft.get("checkpoint_dir", "/content/checkpoints"),
        num_train_epochs=int(tr["epochs"]),
        per_device_train_batch_size=int(tr["batch_size"]),
        gradient_accumulation_steps=int(tr["grad_accum"]),
        learning_rate=float(tr["learning_rate"]),
        fp16=False,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=int(tr["max_seq_len"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        data_collator=_ResponseOnlyCollator(),
        args=args,
    )

    print("[train] Starting training...")
    trainer.train()
    print("[train] Done.")
    return trainer


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_adapter(model, tokenizer, config: dict) -> None:
    import os

    save_path = config["finetune"]["adapter_save_path"]
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[save] Adapter saved to: {save_path}")


def load_adapter(config: dict):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from huggingface_hub import login

    ft = config["finetune"]
    login(token=ft["hf_token"])

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        ft["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        token=ft["hf_token"],
    )
    tokenizer = AutoTokenizer.from_pretrained(ft["adapter_save_path"])
    model = PeftModel.from_pretrained(base, ft["adapter_save_path"])
    model.eval()
    print(f"[load] Adapter loaded from: {ft['adapter_save_path']}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _make_prompt(narrative: str, system_prompt: str, user_template: str | None) -> str:
    user_msg = (
        user_template.replace("{narrative}", narrative).replace("{context}", narrative)
        if user_template
        else f"Classify this accident narrative:\n\n{narrative}"
    )
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{system_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def _parse_output(generated: str, classes: list) -> str:
    import json

    try:
        data = json.loads(generated)
        return data.get("Final_Class", "").strip()
    except Exception:
        for cls in classes:
            if cls.lower() in generated.lower():
                return cls
        return generated[:50]


def predict_one(
    narrative: str,
    model,
    tokenizer,
    system_prompt: str,
    user_template: str | None,
    config: dict,
) -> str:
    import torch

    max_seq_len = int(config["finetune"]["training"]["max_seq_len"])
    classes = config["finetune"]["classes"]
    prompt = _make_prompt(narrative, system_prompt, user_template)
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_seq_len
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()
    return _parse_output(generated, classes)


def run_inference(
    model,
    tokenizer,
    system_prompt: str,
    user_template: str | None,
    config: dict,
    batch_size: int = 8,
) -> pd.DataFrame:
    import torch
    from tqdm import tqdm

    # enable KV cache (disabled during training by gradient checkpointing)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    tokenizer.padding_side = "left"

    ft = config["finetune"]
    new_path = Path(ft["new_data_path"])
    out_path = Path(ft["inference_output"])
    max_seq_len = int(ft["training"]["max_seq_len"])
    classes = ft["classes"]
    label_lookup = {c.lower(): c for c in classes}

    df = (
        pd.read_csv(new_path, low_memory=False)
        if new_path.suffix.lower() == ".csv"
        else pd.read_excel(new_path)
    )
    narr_col = ft["narrative_column"]
    narratives = df[narr_col].astype(str).tolist()
    prompts = [
        _make_prompt(n.strip(), system_prompt, user_template) for n in narratives
    ]

    predictions = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="[inference]"):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        for out in outputs:
            input_len = inputs["input_ids"].shape[1]
            generated = tokenizer.decode(
                out[input_len:], skip_special_tokens=True
            ).strip()
            raw = _parse_output(generated, classes)
            predictions.append(label_lookup.get(raw.lower(), raw))

    df["predicted_hfacs_class"] = predictions
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[inference] Saved {len(df)} rows to: {out_path}")
    return df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_from_csv(results_path: str, config: dict) -> None:
    ft = config["finetune"]
    classes = ft["classes"]
    df = pd.read_csv(results_path, low_memory=False)
    y_true = df[ft["class_column"]].str.strip()
    y_pred = df["predicted_hfacs_class"].str.strip()
    valid = y_true.isin(classes) & y_pred.isin(classes)
    ev_true, ev_pred = y_true[valid], y_pred[valid]
    print("\n=== Evaluation ===")
    print(
        f"Total: {len(df)}  |  Evaluated: {valid.sum()}  |  Excluded: {(~valid).sum()}"
    )
    print(f"\nAccuracy: {accuracy_score(ev_true, ev_pred):.4f}")
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(ev_true, ev_pred, labels=classes)
    print(
        pd.DataFrame(
            cm,
            index=[f"True: {c}" for c in classes],
            columns=[f"Pred: {c}" for c in classes],
        ).to_string()
    )
    print("\n--- Classification Report ---")
    print(classification_report(ev_true, ev_pred, labels=classes, zero_division=0))


def evaluate(
    model,
    tokenizer,
    val_df: pd.DataFrame,
    system_prompt: str,
    user_template: str | None,
    config: dict,
) -> None:
    from tqdm import tqdm

    ft = config["finetune"]
    narr_col = ft["narrative_column"]
    class_col = ft["class_column"]
    classes = ft["classes"]
    label_lookup = {c.lower(): c for c in classes}

    # use already-predicted column if present — avoids re-running inference
    val_df = val_df.copy()
    if "predicted_hfacs_class" in val_df.columns:
        val_df["y_pred"] = val_df["predicted_hfacs_class"].str.strip()
    else:
        val_df["y_pred"] = [
            label_lookup.get(
                predict_one(
                    str(r[narr_col]).strip(),
                    model,
                    tokenizer,
                    system_prompt,
                    user_template,
                    config,
                ),
                None,
            )
            for _, r in tqdm(val_df.iterrows(), total=len(val_df), desc="[eval]")
        ]
    val_df["y_true"] = val_df[class_col].str.strip().str.lower().map(label_lookup)

    valid = val_df["y_pred"].isin(classes) & val_df["y_true"].isin(classes)
    invalid = val_df[~valid]
    ev = val_df[valid]

    print("\n=== Evaluation ===")
    print(f"Total: {len(val_df)}  |  Evaluated: {len(ev)}  |  Excluded: {len(invalid)}")
    if not invalid.empty:
        print(invalid[[narr_col, "y_pred"]].head())
    print(f"\nAccuracy: {accuracy_score(ev['y_true'], ev['y_pred']):.4f}")
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(ev["y_true"], ev["y_pred"], labels=classes)
    print(
        pd.DataFrame(
            cm,
            index=[f"True: {c}" for c in classes],
            columns=[f"Pred: {c}" for c in classes],
        ).to_string()
    )
    print("\n--- Classification Report ---")
    print(
        classification_report(
            ev["y_true"], ev["y_pred"], labels=classes, zero_division=0
        )
    )


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_finetune_pipeline(config: dict) -> None:
    system_prompt, user_template = load_prompts(config)

    train_df = load_training_data(config)
    train_dataset = build_hf_dataset(train_df, config, system_prompt, user_template)

    model, tokenizer = load_model_with_lora(config)
    train(model, tokenizer, train_dataset, config)
    save_adapter(model, tokenizer, config)

    if config["finetune"].get("new_data_path"):
        run_inference(model, tokenizer, system_prompt, user_template, config)
