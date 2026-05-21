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
from llm.classify import run as run_classify
from models.ghfacs import (
    ghfacs_rf,
    ghfacs_rf_balancing,
    ghfacs_rf_no_rs,
    ghfacs_svm,
    ghfacs_svm_balancing,
    ghfacs_svm_no_rs,
)
from models.hfacs import random_forest as run_rf
from models.hfacs import svm as run_svm
from utils import skip_run

from data.dataset import build_processed_dataset, undersample_ae100

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)


with skip_run("skip", "load_raw_dataset") as check, check():
    processed_csv_path = Path(config["paths"]["processed_csv"])
    build_processed_dataset(config)
    print(f"[INFO] Saved processed dataset: {processed_csv_path}")

with skip_run("skip", "undersampled_dataset") as check, check():
    df = undersample_ae100(config)

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

with skip_run("run", "rf") as check, check():
    run_rf(config)

with skip_run("skip", "ghfacs_svm") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_svm(config, df)

with skip_run("skip", "ghfacs_rf") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_rf(config, df)

with skip_run("skip", "ghfacs_rf_no_rs") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_rf_no_rs(config, df)

with skip_run("skip", "ghfacs_svm_balancing") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_svm_balancing(config, df)

with skip_run("skip", "ghfacs_rf_balancing") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_rf_balancing(config, df)

with skip_run("skip", "ghfacs_svm_no_rs") as check, check():
    df = load_dataset(config["paths"]["undersampled_csv"])
    ghfacs_svm_no_rs(config, df)

with skip_run("skip", "classify") as check, check():
    run_classify(config)
