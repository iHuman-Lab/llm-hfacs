from pathlib import Path

import yaml
from eda import (
    build_eda,
    default_input_from_config,
    print_report,
    read_table,
    save_tables,
)
from features.utils import load_dataset
from llm.classify import run_llm_classification as run_classify
from llm.finetune import run_finetune_pipeline as run_finetune
from llm.extract_preconditions import run_extract_preconditions
from models.hfacs import svm as run_svm
from models.random_forest import random_forest as run_rf
from models.precond_rf import precond_rf as run_rf_preconditions
from models.precondition_rf_lora import (
    precondition_rf_lora as run_rf_lora_preconditions,
)
from models.precondition_rf_gpt import precondition_rf_gpt as run_rf_gpt_preconditions
from models.precondition_rf_gpt_reverse import (
    precondition_rf_gpt_reverse as run_rf_gpt_reverse_preconditions,
)
from models.svm_classify import svm_classify as run_svm_classify

from utils import skip_run

from data.dataset import (
    build_balanced_dataset,
    build_processed_dataset,
    undersample_ae100,
)

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


with skip_run("skip", "load_raw_dataset") as check, check():
    processed_csv_path = Path(config["paths"]["processed_csv"])
    build_processed_dataset(config)
    print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("skip", "undersampled_dataset") as check, check():
    df = undersample_ae100(config)

with skip_run("skip", "balanced_dataset") as check, check():
    build_balanced_dataset(config)

with skip_run("skip", "eda") as check, check():
    eda_output_dir = Path(config["paths"]["eda_output_dir"])
    df_eda = read_table(default_input_from_config(config))
    tables = build_eda(df_eda)
    print_report(tables, len(df_eda))
    save_tables(tables, eda_output_dir)
    print(f"\n[INFO] Saved EDA tables to: {eda_output_dir}")

with skip_run("skip", "svm") as check, check():
    df = load_dataset(config["paths"]["processed_csv"])
    run_svm(config, df)

with skip_run("skip", "extract_preconditions") as check, check():
    run_extract_preconditions(config)

with skip_run("skip", "classify") as check, check():  # mode: evaluate in config.yaml
    run_classify(config)

with skip_run("skip", "rf") as check, check():
    run_rf(config)

with skip_run("skip", "rf_preconditions") as check, check():
    run_rf_preconditions(config)

with skip_run("skip", "svm_classify") as check, check():
    run_svm_classify(config)

with skip_run("skip", "finetune") as check, check():
    run_finetune(config)

with skip_run("skip", "rf_lora_preconditions") as check, check():
    run_rf_lora_preconditions(config)

with skip_run("skip", "rf_gpt_preconditions") as check, check():
    run_rf_gpt_preconditions(config)

with skip_run("run", "rf_gpt_reverse_preconditions") as check, check():
    run_rf_gpt_reverse_preconditions(config)
