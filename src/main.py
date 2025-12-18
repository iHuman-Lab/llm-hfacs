import yaml
import pandas as pd
from utils import skip_run

# ============================================================
# IMPORT ALL FUNCTIONS YOU DEFINED
# ============================================================

# preprocessing + HFACS creation
from data.dataset import (
    load_raw_dataset,
    extract_factor_columns,
    create_hfacs_categories,
    save_outputs,
)


from features.hfacs_order_probability import compute_all_full_hfacs_chains, compute_hfacs_ordered_probabilities, conditional_probabilities_hfacs, HFACS_ORDER, compute_combined_hfacs_matrix

# HFACS chain function
#from models.prediction import compute_full_chain


# ============================================================
# LOAD CONFIG
# ============================================================

with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]


# ============================================================
#  LOAD RAW DATA
# ============================================================

with skip_run("run", "load_raw_dataset") as check:
    if check():
        print("[INFO] Loading raw dataset...")
        df = load_raw_dataset(
            paths["raw_data"],
            save_cleaned=True,
        )
        print("[INFO] Raw shape:", df.shape)


# ============================================================
# EXTRACT FACTOR COLUMNS
# ============================================================

with skip_run("run", "extract_factor_columns") as check:
    if check():
        print("[INFO] Extracting factor columns...")
        df = extract_factor_columns(
            df,
            source_columns,
            save_step=True,
        )
        print("[INFO] After factor expansion:", df.shape)


# ============================================================
#  CREATE HFACS CATEGORY COLUMNS
# ============================================================

with skip_run("run", "create_hfacs_categories") as check:
    if check():
        print("[INFO] Creating HFACS categories...")
        df = create_hfacs_categories(
            df,
            hfacs_map,
            save_step=True,
        )
        print("[INFO] After HFACS categories:", df.shape)


# ============================================================
# SAVE FINAL DATASET
# ============================================================

with skip_run("run", "save_outputs") as check:
    if check():
        print("[INFO] Saving final dataset...")
        save_outputs(
            df,
            paths["processed_csv"],
            paths["processed_excel"],
        )
        print("[INFO] Saved CSV & Excel")



# ============================================================
#  COMPUTE HFACS-ORDERED CONDITIONAL PROBABILITIES
# ============================================================

with skip_run("run", "hfacs_ordered_probabilities") as check:
    if check():
        print("[INFO] Computing HFACS-ordered conditional probabilities...")
        compute_hfacs_ordered_probabilities(
            df,
            hfacs_order=HFACS_ORDER,
            output_dir="./data/processed",
        )
        print("[INFO] HFACS ordered probability tables saved.")


# ============================================================
# COMPUTE ALL HFACS FULL CHAINS (L4 → L1)
# ============================================================

with skip_run("run", "hfacs_full_chains") as check:
    if check():
        print("[INFO] Computing ALL HFACS full chains (Error + Violation)...")

        all_chains_df = compute_all_full_hfacs_chains(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
        )

        all_chains_df.to_csv(
            "./data/processed/HFACS_all_L4_to_L1_chains.csv",
            index=False,
        )

        print("[INFO] HFACS full-chain computation complete.")
        print(all_chains_df.head())


# ============================================================
#  COMPUTE COMBINED HFACS MATRIX (L4 → L1, AGGREGATE)
# ============================================================

with skip_run("run", "hfacs_combined_matrix") as check:
    if check():
        print("[INFO] Computing combined HFACS matrix (L4 → L1)...")

        combined_df = compute_combined_hfacs_matrix(
            hfacs_order=HFACS_ORDER,
            processed_dir="./data/processed",
            filename="HFACS_L4_to_L1_combined.csv",
        )

        print("[INFO] Combined HFACS matrix saved.")
        print(combined_df)
