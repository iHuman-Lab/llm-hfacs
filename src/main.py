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
from features.conditional_prob import (
    conditional_probabilities,
    compute_all_hfacs_probabilities,
)


from features.hfacs_order_probability import (
    compute_hfacs_ordered_probabilities,
    compute_full_hfacs_chain,
)

# HFACS chain function
from models.prediction import compute_full_chain


# ============================================================
# 1. LOAD CONFIG
# ============================================================

with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]


# ============================================================
# 2. LOAD RAW DATA
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
# 3. EXTRACT FACTOR COLUMNS
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
# 4. CREATE HFACS CATEGORY COLUMNS
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
# 5. SAVE FINAL DATASET
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
# 7. COMPUTE ALL HFACS CONDITIONAL PROBABILITIES
# ============================================================

with skip_run("run", "hfacs_probabilities") as check:
    if check():
        print("[INFO] Computing HFACS conditional probabilities...")
        compute_all_hfacs_probabilities(
            df,
            hfacs_map,
            output_dir="./data/processed",
        )
        print("[INFO] HFACS probability tables saved.")


# ============================================================
# 8. COMPUTE HFACS FULL CHAIN (USING SAVED PROBABILITIES)
# ============================================================

with skip_run("run", "hfacs_full_chain") as check:
    if check():
        print("[INFO] Computing HFACS chained probabilities...")

        # Example chain — YOU choose the order
        chain = [
            "Organizational_Process",
            "Inadequate_Supervision",
            "Error",
        ]

        chain_df = compute_full_chain(
            chain=chain,
            processed_dir="./data/processed",
        )

        chain_df.to_csv(
            "./data/processed/HFACS_chain_Organizational_to_Error.csv"
        )

        print("[INFO] HFACS chain result:")
        print(chain_df)



with skip_run("run", "hfacs_probabilities") as check:
    if check():
        print("[INFO] Computing HFACS conditional probabilities...")
        compute_hfacs_ordered_probabilities(
            df,
            output_dir="./data/processed",
        )
        print("[INFO] HFACS probability tables saved.")

