

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import multiprocessing as mp

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main():
    # 1. LOAD DATA
    DATA_PATH = r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Elahe-oveisi-osu-resaerch\casualty\llm-hfacs\data\processed\step3_hfacs_categories.csv"

    HFACS_L4 = ["Organizational_Process", "Organizational_Climate", "Resource_Management"]
    HFACS_L3 = ["Inadequate_Supervision", "Failure_to_Correct"]
    HFACS_L2 = ["Condition_of_Operators", "Personnel_Factors", "Situational_Factors"]
    HFACS_L1 = ["Error", "Violation"]

    try:
        df = pd.read_csv(DATA_PATH)
        df = df[HFACS_L4 + HFACS_L3 + HFACS_L2 + HFACS_L1].dropna().astype(int)
    except Exception as e:
        print(f"[ERROR] {e}")
        return

    # 2. DEFINE COORDINATES (This is what gives you the names!)
    coords = {
        "L4_factors": HFACS_L4,
        "L3_factors": HFACS_L3,
        "L2_factors": HFACS_L2,
        "L1_factors": HFACS_L1
    }

    # 3. BUILD MODEL
    with pm.Model(coords=coords) as hfacs_model:
        # L4 -> L3
        beta_L3 = pm.Normal("beta_L3", mu=0, sigma=1, dims=("L4_factors", "L3_factors"))
        int_L3 = pm.Normal("intercept_L3", mu=0, sigma=1, dims="L3_factors")
        logits_L3 = int_L3 + pm.math.dot(df[HFACS_L4].values, beta_L3)
        pm.Bernoulli("L3_obs", logit_p=logits_L3, observed=df[HFACS_L3].values)

        # L3 -> L2
        beta_L2 = pm.Normal("beta_L2", mu=0, sigma=1, dims=("L3_factors", "L2_factors"))
        int_L2 = pm.Normal("intercept_L2", mu=0, sigma=1, dims="L2_factors")
        logits_L2 = int_L2 + pm.math.dot(df[HFACS_L3].values, beta_L2)
        pm.Bernoulli("L2_obs", logit_p=logits_L2, observed=df[HFACS_L2].values)

        # L2 -> L1
        beta_L1 = pm.Normal("beta_L1", mu=0, sigma=1, dims=("L2_factors", "L1_factors"))
        int_L1 = pm.Normal("intercept_L1", mu=0, sigma=1, dims="L1_factors")
        logits_L1 = int_L1 + pm.math.dot(df[HFACS_L2].values, beta_L1)
        pm.Bernoulli("L1_obs", logit_p=logits_L1, observed=df[HFACS_L1].values)

        # 4. SAMPLE
        trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4, target_accept=0.9)

    # 5. GENERATE NAMED SUMMARY
    print("\n[INFO] Sampling complete. Results with factor names:")
    summary = az.summary(trace, var_names=["beta_L3", "beta_L2", "beta_L1"])
    print(summary)

    # Example: P(Error) calculation
    b_mean = trace.posterior["beta_L1"].mean(dim=["chain", "draw"]).values
    i_mean = trace.posterior["intercept_L1"].mean(dim=["chain", "draw"]).values
    p_error = sigmoid(i_mean[0] + np.dot(df[HFACS_L2].values, b_mean[:, 0])).mean()
    print(f"\nOverall Mean Probability of Error: {p_error:.4f}")

if __name__ == "__main__":
    mp.freeze_support()
    main()