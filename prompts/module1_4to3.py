import pandas as pd

# ---- Level 3 and Level 4 lists ----
L3_raw = [
    "Contributing Factors:Staffing",
    "Contributing Factors:MEL",
    "Contributing Factors:Manuals",
]

L4_raw = [
    "Contributing Factors:Chart Or Publication",
    "Contributing Factors:Company Policy",
    "Contributing Factors:Logbook Entry",
    "Contributing Factors:Procedure",
    "Contributing Factors:Incorrect / Not Installed / Unavailable Part",
    "Contributing Factors:Human Factors",
    "Contributing Factors:Software and Automation",
    "Contributing Factors:Aircraft",
    "Contributing Factors:Airport",
    "Contributing Factors:Airspace Structure",
    "Contributing Factors:ATC Equipment / Nav Facility / Buildings",
    "Contributing Factors:Equipment / Tooling"
]

df = pd.read_excel("data/ASRS_all_factors.xlsx").fillna(0)

# ---- clean names ----
def clean(x): 
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")

L3 = { clean(c): c for c in L3_raw }
L4 = { clean(c): c for c in L4_raw }

# ---- compute conditional probabilities ----
rows = []

for L4_clean, L4_col in L4.items():
    df_L4_active = df[df[L4_col] == 1]        # filter records where L4 factor occurred
    n = len(df_L4_active)
    
    if n == 0:
        continue
    
    row = {"L4_factor": L4_clean, "N_cases": n}
    
    for L3_clean, L3_col in L3.items():
        prob = df_L4_active[L3_col].mean()    # mean of 0/1 = probability
        row[f"P_{L3_clean}"] = round(prob, 4)
    
    rows.append(row)

result_df = pd.DataFrame(rows)

# ---- save CSV ----
result_df.to_csv("L4_to_L3_probabilities.csv", index=False)

result_df
print(result_df)
result_df.to_csv("L4_to_L3_probabilities.csv", index=False)
print("Saved: L4_to_L3_probabilities.csv")
