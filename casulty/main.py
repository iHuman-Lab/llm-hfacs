import os
import pandas as pd
from tqdm import tqdm
 
from utils import skip_run
 
# Import your four HFACS Bayesian functions
from function_final_beysian import (
    compute_L2_to_L1,
    compute_L3_to_L2,
    compute_L4_to_L3,
    compute_full_chain,
)
 
# ============================================================
# HFACS BAYESIAN PROCESS PIPELINE
# ============================================================
 
INPUT_FILE = r"C:\llm-hfacs\data\ASRS_all_factors.xlsx"
 

# Compute P(L3 | L4)
with skip_run("skip", "compute_L4_to_L3") as check, check():
    print("\n=== STEP 3: Computing P(L3 | L4) ===")
    df43 = compute_L4_to_L3(INPUT_FILE)
    print(df43.head())
    print("Saved: function_L4_to_L3_probabilities.csv")
 

# Compute P(L2 | L3)
with skip_run("skip", "compute_L3_to_L2") as check, check():
    print("\n=== STEP 2: Computing P(L2 | L3) ===")
    df32 = compute_L3_to_L2(INPUT_FILE)
    print(df32.head())
    print("Saved: function_L3_to_L2_probabilities.csv")
 
 

# Compute P(L1 | L2)
with skip_run("skip", "compute_L2_to_L1") as check, check():
    print("\n=== STEP 1: Computing P(L1 | L2) ===")
    df21 = compute_L2_to_L1(INPUT_FILE)
    print(df21.head())
    print("Saved: function_L2_to_L1_probabilities.csv")
 
 

# Compute P(L2 | L3)
with skip_run("skip", "compute_L3_to_L2") as check, check():
    print("\n=== STEP 2: Computing P(L2 | L3) ===")
    df32 = compute_L3_to_L2(INPUT_FILE)
    print(df32.head())
    print("Saved: function_L3_to_L2_probabilities.csv")
 
 

# Compute Full Chain: P(L1 | L4) = A * B * C
with skip_run("run", "compute_full_chain") as check, check():
    print("\n=== STEP 4: Computing FULL CHAIN P(L1 | L4) ===")
    full_df = compute_full_chain()
    print(full_df.head())
    print("Saved: function_L4_to_L1_full_chain_probabilities.csv")
 