import yaml
import pandas as pd

from data.dataset import (
    load_raw_dataset,
    extract_factor_columns,
    create_hfacs_categories,
    save_outputs
)

from utils import skip_run


# ============================================================
# 1. LOAD CONFIG
# ============================================================
with open("./configs/config.yaml") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
category_map = config["hfacs_categories"]


# ============================================================
# 2. LOAD RAW DATA
# ============================================================
with skip_run("run", "load_raw_data") as check:
    if check():
        print("[INFO] Loading raw ASRS dataset...")
        df = load_raw_dataset(
            paths["raw_data"],
            save_cleaned=True  # ensures step1 file is saved
        )
        print("[INFO] Loaded dataset shape:", df.shape)


# ============================================================
# 3. EXTRACT FACTORS (creates step2 file)
# ============================================================
with skip_run("run", "extract_factor_columns") as check:
    if check():
        print("[INFO] Extracting semicolon-separated factors...")
        df = extract_factor_columns(
            df,
            source_columns,
            save_step=True  # saves step2_factor_expanded.csv
        )
        print("[INFO] After factor expansion:", df.shape)


# ============================================================
# 4. OPTIONAL HFACS CATEGORIES (creates step3 file)
# ============================================================
with skip_run("skip", "create_hfacs_categories") as check:
    if check():
        print("[INFO] Creating HFACS category columns...")
        df = create_hfacs_categories(
            df,
            category_map,
            save_step=True  # saves step3_hfacs_categories.csv
        )
        print("[INFO] HFACS categories added:", df.shape)


# ============================================================
# 5. SAVE FINAL OUTPUT
# ============================================================
with skip_run("run", "save_outputs") as check:
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
