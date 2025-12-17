# ============================================================
# OPTIMIZED FULL BAYESIAN HFACS MODEL (PyMC) — WINDOWS SAFE
# ============================================================

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import multiprocessing as mp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================
    DATA_PATH = r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Elahe-oveisi-osu-resaerch\casualty\llm-hfacs\data\processed\step3_hfacs_categories.csv"

    HFACS_COLUMNS = [
        "Organizational_Process", "Organizational_Climate", "Resource_Management",
        "Inadequate_Supervision", "Failure_to_Correct",
        "Condition_of_Operators", "Personnel_Factors", "Situational_Factors",
        "Error", "Violation",
    ]

    try:
        df = pd.read_csv(DATA_PATH)
        df = df[HFACS_COLUMNS].dropna().astype(int)
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return

    print(f"[INFO] HFACS dataset ready: {df.shape}")

    # Define Column Groups
    HFACS_L4 = ["Organizational_Process", "Organizational_Climate", "Resource_Management"]
    HFACS_L3 = ["Inadequate_Supervision", "Failure_to_Correct"]
    HFACS_L2 = ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"]
    HFACS_L1 = ["Error", "Violation"]

    # ============================================================
    # 2. BUILD OPTIMIZED MODEL
    # ============================================================
    with pm.Model() as hfacs_model:
        # L4 → L3
        beta_L3 = pm.Normal("beta_L3", mu=0, sigma=1, shape=(len(HFACS_L4), len(HFACS_L3)))
        intercept_L3 = pm.Normal("intercept_L3", mu=0, sigma=1, shape=len(HFACS_L3))
        logits_L3 = intercept_L3 + pm.math.dot(df[HFACS_L4].values, beta_L3)
        pm.Bernoulli("L3_obs", logit_p=logits_L3, observed=df[HFACS_L3].values)

        # L3 → L2
        beta_L2 = pm.Normal("beta_L2", mu=0, sigma=1, shape=(len(HFACS_L3), len(HFACS_L2)))
        intercept_L2 = pm.Normal("intercept_L2", mu=0, sigma=1, shape=len(HFACS_L2))
        logits_L2 = intercept_L2 + pm.math.dot(df[HFACS_L3].values, beta_L2)
        pm.Bernoulli("L2_obs", logit_p=logits_L2, observed=df[HFACS_L2].values)

        # L2 → L1
        beta_L1 = pm.Normal("beta_L1", mu=0, sigma=1, shape=(len(HFACS_L2), len(HFACS_L1)))
        intercept_L1 = pm.Normal("intercept_L1", mu=0, sigma=1, shape=len(HFACS_L1))
        logits_L1 = intercept_L1 + pm.math.dot(df[HFACS_L2].values, beta_L1)
        pm.Bernoulli("L1_obs", logit_p=logits_L1, observed=df[HFACS_L1].values)

        # ============================================================
        # 3. SAMPLE POSTERIOR
        # ============================================================
        print("[INFO] Starting sampling... this may take 15-30 minutes.")
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=1, 
            cores=1,
            target_accept=0.8, # Reduced from 0.9 for speed
            random_seed=42,
            progressbar=True
        )

    # ============================================================
    # 4. POST-PROCESSING (Manual Probability Calculation)
    # ============================================================
    print("\n[INFO] Sampling finished. Generating summary...")
    
    # Summary of coefficients
    summary = az.summary(trace, var_names=["beta_L1", "beta_L2", "beta_L3"])
    print(summary)

    # Calculate Probability of Error manually using posterior means
    # (Avoids storing millions of rows in the trace)
    b_L1_mean = trace.posterior["beta_L1"].mean(dim=["chain", "draw"]).values
    i_L1_mean = trace.posterior["intercept_L1"].mean(dim=["chain", "draw"]).values
    
    # Calculate logits for 'Error' (index 0 of L1)
    # Equation: intercept + (Data_L2 * Beta_L1)
    error_logits = i_L1_mean[0] + np.dot(df[HFACS_L2].values, b_L1_mean[:, 0])
    avg_p_error = sigmoid(error_logits).mean()

    print(f"\nEstimated Mean Probability of 'Error' across dataset: {avg_p_error:.4f}")

if __name__ == "__main__":
    mp.freeze_support()
    main()