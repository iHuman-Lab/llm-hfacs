import yaml
import pandas as pd
from data.seperator import extract_factors
from features.conditional_prob import (
    conditional_probabilities,
    conditional_probabilities_subcategory,
    build_subcategory_columns,
)
from models.prediction import compute_full_chain
from utils import skip_run

# The configuration file
with open("./configs/config_subcategories.yaml") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


with skip_run("skip", "extract_factors") as check, check():
    extract_factors(config)


with skip_run("skip", "compute_probability") as check, check():
    df = pd.read_csv(config["output_path"]).fillna(0)
    conditional_probabilities(df, "level_4", "level_3", config)
    conditional_probabilities(df, "level_3", "level_2", config)
    conditional_probabilities(df, "level_2", "level_1", config)


with skip_run("skip", "compute_final_probability") as check, check():
    result = compute_full_chain(levels=[4, 3, 2, 1])
    print(result)


with skip_run("skip", "build_subcategory_columns") as check, check():
    config_path = "./configs/config_subcategories.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["output_path"]).fillna(0)

    L4_cols = build_subcategory_columns(df, config, "Level_4")
    L3_cols = build_subcategory_columns(df, config, "Level_3")
    L2_cols = build_subcategory_columns(df, config, "Level_2")
    L1_cols = build_subcategory_columns(df, config, "Level_1")

    print(df.columns.tolist())


with skip_run("skip", "conditional_probabilities") as check, check():
    L4_to_L3 = conditional_probabilities_subcategory(df, L4_cols, L3_cols, "L4", "L3")
    L3_to_L2 = conditional_probabilities_subcategory(df, L3_cols, L2_cols, "L3", "L2")
    L2_to_L1 = conditional_probabilities_subcategory(df, L2_cols, L1_cols, "L2", "L1")

    L4_to_L3.to_csv("./data/processed/prob_L3_given_L4.csv", index=False)
    L3_to_L2.to_csv("./data/processed/prob_L2_given_L3.csv", index=False)
    L2_to_L1.to_csv("./data/processed/prob_L1_given_L2.csv", index=False)

    print("Conditional probabilities saved.")


with skip_run("skip", "compute_full_chain") as check, check():
    result = compute_full_chain()
    result.to_csv("./data/processed/L1_given_L4_fullchain.csv", index=True)
