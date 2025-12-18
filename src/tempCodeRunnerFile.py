import yaml
import pandas as pd

from data.dataset import (
    load_raw_dataset,
    extract_factor_columns,
    create_hfacs_categories,
    save_outputs
)

from features.conditional_prob import (
    conditional_probabilities,
    conditional_probabilities_subcategory
)

from models.prediction import compute_full_chain

from utils import skip_run




# ============================================================
# LOAD CONFIG
# ============================================================
with open("./configs/config.yaml") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
category_map = config["hfacs_categories"]


# ============================================================
with skip_run("run", "load_raw_data") as check:
    if check():
        print("[INFO] Loading raw ASRS dataset...")
        df = load_raw_dataset(paths["raw_data"])


# ============================================================
# 2. EXTRACT FACTOR COLUMNS
# ============================================================
with skip_run("skip", "extract_factor_columns") as check:
    if check():
        print("[INFO] Extracting semicolon-separated factors...")
        df = extract_factor_columns(df, source_columns)
        print("[INFO] After factor expansion:", df.shape)


# ============================================================
# 3. CREATE HFACS CATEGORY FLAGS
# ============================================================
with skip_run("skip", "create_hfacs_categories") as check:
    if check():
        print("[INFO] Creating HFACS category columns...")
        df = create_hfacs_categories(df, category_map)
        print("[INFO] HFACS categories added:", df.shape)


# ============================================================
# 4. SAVE FINAL OUTPUTS
# ============================================================
with skip_run("skip", "save_outputs") as check:
    if check():
        print("[INFO] Saving final processed dataset...")
        save_outputs(
            df,
            paths["processed_csv"],
            paths["processed_excel"]
        )
        print("[INFO] Saved to:")
        print("   CSV:", paths["processed_csv"])
        print("   XLSX:", paths["processed_excel"])


print("\n[COMPLETE] Processing pipeline finished.")