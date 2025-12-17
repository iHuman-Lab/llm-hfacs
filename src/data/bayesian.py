# ============================================================
# FULL BAYESIAN HFACS MODEL (PyMC) — WINDOWS SAFE
# ============================================================

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import multiprocessing as mp


def main():

    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    DATA_PATH = r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Elahe-oveisi-osu-resaerch\casualty\llm-hfacs\data\processed\step3_hfacs_categories.csv"

    HFACS_COLUMNS = [
        "Organizational_Process",
        "Organizational_Climate",
        "Resource_Management",
        "Inadequate_Supervision",
        "Failure_to_Correct",
        "Condition_of_Operators",
        "Personnel_Factors",
        "Situational_Factors",
        "Error",
        "Violation",
    ]

    df = pd.read_csv(DATA_PATH)
    df = df[HFACS_COLUMNS]

    # sanity check
    for col in HFACS_COLUMNS:
        if not set(df[col].dropna().unique()).issubset({0, 1}):
            raise ValueError(f"{col} is not binary")

    print("[INFO] HFACS dataset ready:", df.shape)

    # ============================================================
    # 2. DEFINE HFACS LEVELS
    # ============================================================

    HFACS_L4 = [
        "Organizational_Process",
        "Organizational_Climate",
        "Resource_Management",
    ]

    HFACS_L3 = [
        "Inadequate_Supervision",
        "Failure_to_Correct",
    ]

    HFACS_L2 = [
        "Condition_of_Operators",
        "Personnel_Factors",
        "Situational_Factors",
    ]

    HFACS_L1 = [
        "Error",
        "Violation",
    ]

    # ============================================================
    # 3. BUILD FULL BAYESIAN HFACS MODEL
    # ============================================================

    with pm.Model() as hfacs_model:

        # -------------------------
        # L4 → L3
        # -------------------------
        beta_L3 = pm.Normal(
            "beta_L3",
            mu=0,
            sigma=1,
            shape=(len(HFACS_L4), len(HFACS_L3)),
        )
        intercept_L3 = pm.Normal(
            "intercept_L3",
            mu=0,
            sigma=1,
            shape=len(HFACS_L3),
        )

        X_L4 = df[HFACS_L4].values
        logits_L3 = intercept_L3 + pm.math.dot(X_L4, beta_L3)
        p_L3 = pm.Deterministic("p_L3", pm.math.sigmoid(logits_L3))

        pm.Bernoulli(
            "L3_obs",
            p=p_L3,
            observed=df[HFACS_L3].values,
        )

        # -------------------------
        # L3 → L2
        # -------------------------
        beta_L2 = pm.Normal(
            "beta_L2",
            mu=0,
            sigma=1,
            shape=(len(HFACS_L3), len(HFACS_L2)),
        )
        intercept_L2 = pm.Normal(
            "intercept_L2",
            mu=0,
            sigma=1,
            shape=len(HFACS_L2),
        )

        X_L3 = df[HFACS_L3].values
        logits_L2 = intercept_L2 + pm.math.dot(X_L3, beta_L2)
        p_L2 = pm.Deterministic("p_L2", pm.math.sigmoid(logits_L2))

        pm.Bernoulli(
            "L2_obs",
            p=p_L2,
            observed=df[HFACS_L2].values,
        )

        # -------------------------
        # L2 → L1
        # -------------------------
        beta_L1 = pm.Normal(
            "beta_L1",
            mu=0,
            sigma=1,
            shape=(len(HFACS_L2), len(HFACS_L1)),
        )
        intercept_L1 = pm.Normal(
            "intercept_L1",
            mu=0,
            sigma=1,
            shape=len(HFACS_L1),
        )

        X_L2 = df[HFACS_L2].values
        logits_L1 = intercept_L1 + pm.math.dot(X_L2, beta_L1)
        p_L1 = pm.Deterministic("p_L1", pm.math.sigmoid(logits_L1))

        pm.Bernoulli(
            "L1_obs",
            p=p_L1,
            observed=df[HFACS_L1].values,
        )

        # -------------------------
        # SAMPLE POSTERIOR (WINDOWS SAFE)
        # -------------------------
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=1,      # force single chain
            cores=1,       # force single process
            target_accept=0.9,
            random_seed=42,
        )

    print("[INFO] Sampling finished")

    # ============================================================
    # 4. POSTERIOR SUMMARY
    # ============================================================

    summary = az.summary(trace, hdi_prob=0.95)
    print(summary)

    # ============================================================
    # 5. EXAMPLE: ERROR POSTERIOR
    # ============================================================

    p_error = trace.posterior["p_L1"].sel(p_L1_dim_1=0)

    print("\nPosterior P(Error):")
    print("Mean:", float(p_error.mean()))
    print("95% HDI:", az.hdi(p_error, hdi_prob=0.95).to_array().values)


# ============================================================
# WINDOWS ENTRY POINT (CRITICAL)
# ============================================================

if __name__ == "__main__":
    mp.freeze_support()
    main()
