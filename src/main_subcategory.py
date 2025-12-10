
import pandas as pd
import yaml

from utils import skip_run
from features.conditional_prob_subcategories import (
    build_subcategory_columns,
    compute_conditional_probabilities
)
from models.prediction_subcategories import compute_full_chain


# ============================================================
# LOAD CONFIG + DATA
# ============================================================
config_path = "./configs/config_subcategories.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["output_path"]).fillna(0)


# ============================================================
# 1. BUILD SUBCATEGORY COLUMNS
# ============================================================
with skip_run("run", "build_subcategory_columns") as check:
    if check():
        print("[INFO] Building Level 4...")
        L4_cols = build_subcategory_columns(df, config, "Level_4")

        print("[INFO] Building Level 3...")
        L3_cols = build_subcategory_columns(df, config, "Level_3")

        print("[INFO] Building Level 2...")
        L2_cols = build_subcategory_columns(df, config, "Level_2")

        print("[INFO] Building Level 1...")
        L1_cols = build_subcategory_columns(df, config, "Level_1")

        print("[OK] Subcategories created.")
        print(df.columns.tolist())


# ============================================================
# 2. CONDITIONAL PROBABILITIES
# ============================================================
with skip_run("run", "compute_conditional_probabilities") as check:
    if check():
        print("[INFO] Computing conditional probabilities...")

        L4_to_L3 = compute_conditional_probabilities(df, L4_cols, L3_cols, "L4", "L3")
        L3_to_L2 = compute_conditional_probabilities(df, L3_cols, L2_cols, "L3", "L2")
        L2_to_L1 = compute_conditional_probabilities(df, L2_cols, L1_cols, "L2", "L1")

        L4_to_L3.to_csv("./data/processed/aL3_given_L4.csv", index=False)
        L3_to_L2.to_csv("./data/processed/aL2_given_L3.csv", index=False)
        L2_to_L1.to_csv("./data/processed/aL1_given_L2.csv", index=False)

        print("[OK] Conditional probabilities saved.")


# ============================================================
# 3. FULL CHAIN L4 → L1
# ============================================================
with skip_run("run", "compute_full_chain") as check:
    if check():
        print("[INFO] Computing full chain L4 → L1...")

        result = compute_full_chain()
        result.to_csv("./data/processed/aL1_given_L4_fullchain.csv", index=True)

        print("[OK] Full chain saved.")
