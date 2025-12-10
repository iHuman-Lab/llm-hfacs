# =====================================================================
# HFACS Bayesian Network Test Script (Corrected to match your dataset)
# =====================================================================

import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination

# =====================================================================
# 1. Load Your Dataset
# =====================================================================

df = pd.read_csv(
    r"C:\Users\elahe\OneDrive - Oklahoma A and M System\Elahe-oveisi-osu-resaerch\casualty\llm-hfacs\data\processed\ASRS_all_factors_extracted_with_subcats.csv"
).fillna(0)

print("[OK] Dataset loaded:", df.shape)

print("\n[DEBUG] HFACS columns found in dataset:")
print([c for c in df.columns if c.startswith("Level_")])

# =====================================================================
# 2. Define Bayesian Network Based ONLY on Existing Columns
# =====================================================================

model = BayesianNetwork([

    # ---------- Level 4 → Level 3 ----------
    ("Level_4_Organizational_Process", "Level_3_Inadequate_Supervision"),
    ("Level_4_Organizational_Process", "Level_3_Failure_to_Correct_Known_Problems"),
    ("Level_4_Organizational_Process", "Level_3_Planned_Inappropriate_Operations"),
    ("Level_4_Organizational_Process", "Level_3_Supervisory_Violations"),

    ("Level_4_Organizational_Climate", "Level_3_Inadequate_Supervision"),
    ("Level_4_Organizational_Climate", "Level_3_Failure_to_Correct_Known_Problems"),
    ("Level_4_Organizational_Climate", "Level_3_Planned_Inappropriate_Operations"),
    ("Level_4_Organizational_Climate", "Level_3_Supervisory_Violations"),

    ("Level_4_Resource_Management", "Level_3_Inadequate_Supervision"),
    ("Level_4_Resource_Management", "Level_3_Failure_to_Correct_Known_Problems"),
    ("Level_4_Resource_Management", "Level_3_Planned_Inappropriate_Operations"),
    ("Level_4_Resource_Management", "Level_3_Supervisory_Violations"),

    # ---------- Level 3 → Level 2 (only one L2 exists) ----------
    ("Level_3_Inadequate_Supervision", "Level_2_Situational_Factors"),
    ("Level_3_Failure_to_Correct_Known_Problems", "Level_2_Situational_Factors"),
    ("Level_3_Planned_Inappropriate_Operations", "Level_2_Situational_Factors"),
    ("Level_3_Supervisory_Violations", "Level_2_Situational_Factors"),

    # ---------- Level 2 → Level 1 (only one L1 exists) ----------
    ("Level_2_Situational_Factors", "Level_1_Errors"),
])

print("\n[OK] HFACS graph structure created.")


# =====================================================================
# 3. Fit Conditional Probability Tables
# =====================================================================

print("\n[INFO] Fitting Bayesian model using BDeu priors...")

model.fit(
    df,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10
)

print("[OK] Model fitted.")


# =====================================================================
# 4. Create Inference Engine
# =====================================================================

infer = VariableElimination(model)
print("[OK] Inference engine ready.")


# =====================================================================
# 5. Example Query
# =====================================================================

query_result = infer.query(
    variables=["Level_1_Errors"],
    evidence={"Level_4_Organizational_Process": 1}
)

print("\n===================================================")
print(" P(Level_1_Errors | Level_4_Organizational_Process = 1)")
print("===================================================")
print(query_result)
