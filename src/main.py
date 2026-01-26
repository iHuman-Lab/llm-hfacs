import yaml

from data.dataset import (
    create_hfacs_categories,
    extract_factor_columns,
    load_raw_dataset,
    save_outputs,
)
from features.hfacs_order_probability import (
    HFACS_ORDER,
    compute_all_full_hfacs_chains,
    compute_combined_hfacs_matrix,
    compute_hfacs_ordered_probabilities,
)
from utils import skip_run

# HFACS chain function
# from models.prediction import compute_full_chain


with open("./configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

paths = config["paths"]
source_columns = config["source_columns"]
hfacs_map = config["hfacs_categories"]


with skip_run("skip", "load_raw_dataset") as check, check():
    print("[INFO] Loading raw dataset...")
    df = load_raw_dataset(
        paths["raw_data"],
        save_cleaned=True,
    )
    print("[INFO] Raw shape:", df.shape)


with skip_run("skip", "create_hfacs_category_data") as check, check():
    print("[INFO] Extracting factor columns...")
    df = extract_factor_columns(
        df,
        source_columns,
        save_step=True,
    )
    print("[INFO] Creating HFACS categories...")
    df = create_hfacs_categories(
        df,
        hfacs_map,
        save_step=True,
    )
    save_outputs(
        df,
        paths["processed_csv"],
        paths["processed_excel"],
    )
    print("[INFO] Saved CSV & Excel")


with skip_run("skip", "hfacs_ordered_probabilities") as check, check():
    print("[INFO] Computing HFACS-ordered conditional probabilities...")
    compute_hfacs_ordered_probabilities(
        df,
        hfacs_order=HFACS_ORDER,
        output_dir="./data/processed",
    )
    print("[INFO] HFACS ordered probability tables saved.")


with skip_run("skip", "hfacs_full_chains") as check, check():
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


with skip_run("skip", "hfacs_combined_matrix") as check, check():
    print("[INFO] Computing combined HFACS matrix (L4 → L1)...")

    combined_df = compute_combined_hfacs_matrix(
        hfacs_order=HFACS_ORDER,
        processed_dir="./data/processed",
        filename="HFACS_L4_to_L1_combined.csv",
    )

    print("[INFO] Combined HFACS matrix saved.")
    print(combined_df)
