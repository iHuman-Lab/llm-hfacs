import os
import pandas as pd
from tqdm import tqdm

from utils import skip_run

# -----------------------------------------------------------
# Import HFACS Bayesian functions
# -----------------------------------------------------------
from function_final_beysian import (
    compute_L2_to_L1,
    compute_L3_to_L2,
    compute_L4_to_L3,
    compute_full_chain,
)

# -----------------------------------------------------------
# Import your separator (extract_factors)
# -----------------------------------------------------------
from seperator import extract_factors


# ============================================================
# FILE PATHS
# ============================================================

RAW_INPUT = r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Elahe-oveisi-osu-resaerch\casualty\ASRS_all_factors_raw.xls"

# This is the file that Bayesian functions will use
PROCESSED_FILE = r"ASRS_all_factors.xlsx"


# ============================================================
# STEP 0 — Extract factors → Save as ASRS_all_factors.xlsx
# ============================================================

with skip_run("run", "extract_factors") as check, check():
    print("\n=== STEP 0: Extracting factors & creating ASRS_all_factors.xlsx ===")
    info = extract_factors(RAW_INPUT, PROCESSED_FILE)
    print("Created factor columns:")
    print(info["columns"])
    print("Saved:", info["output_file"])


# ============================================================
# STEP 1 — Compute P(L1 | L2)
# ============================================================

with skip_run("skip", "compute_L2_to_L1") as check, check():
    print("\n=== STEP 1: Computing P(L1 | L2) ===")
    df21 = compute_L2_to_L1(PROCESSED_FILE)
    print(df21.head())
    print("Saved: function_L2_to_L1_probabilities.csv")


# ============================================================
# STEP 2 — Compute P(L2 | L3)
# ============================================================

with skip_run("skip", "compute_L3_to_L2") as check, check():
    print("\n=== STEP 2: Computing P(L2 | L3) ===")
    df32 = compute_L3_to_L2(PROCESSED_FILE)
    print(df32.head())
    print("Saved: function_L3_to_L2_probabilities.csv")


# ============================================================
# STEP 3 — Compute P(L3 | L4)
# ============================================================
with skip_run("skip", "compute_L4_to_L3") as check, check():
    print("\n=== STEP 3: Computing P(L3 | L4) ===")
    df43 = compute_L4_to_L3(PROCESSED_FILE)
    print(df43.head())
    print("Saved: function_L4_to_L3_probabilities.csv")

# ============================================================
# STEP 4 — FULL BAYESIAN CHAIN: P(L1 | L4)
# ============================================================

with skip_run("skip", "compute_full_chain") as check, check():
    print("\n=== STEP 4: Computing FULL CHAIN P(L1 | L4) ===")
    full_df = compute_full_chain()
    print(full_df.head())
    print("Saved: function_L4_to_L1_full_chain_probabilities.csv")
