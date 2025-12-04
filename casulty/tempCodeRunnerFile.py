
    df43 = compute_L4_to_L3(PROCESSED_FILE)
    print(df43.head())
    print("Saved: function_L4_to_L3_probabilities.csv")


# ============================================================
# STEP 4 — FULL BAYESIAN CHAIN: P(L1 | L4)
# ============================================================

with skip_run("skip", "compute_full_chain") as check