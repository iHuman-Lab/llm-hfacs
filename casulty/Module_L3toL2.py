import pandas as pd

# ---------- Level 3 (supervisory layer) ----------
L3_raw = [
    "Contributing Factors:Staffing",
    "Contributing Factors:MEL",
    "Contributing Factors:Manuals",
]

# ---------- Level 2 (preconditions layer) ----------
L2_raw = [

    # Environmental / Technical
    "Contributing Factors:Weather",
    "Contributing Factors:Environment - Non Weather Related",
    "Anomaly:Ground Event / Encounter Weather / Turbulence",
    "Anomaly:Inflight Event / Encounter Weather / Turbulence",
    "Anomaly:Inflight Event / Encounter Wake Vortex Encounter",
    "Anomaly:Inflight Event / Encounter Bird / Animal",
    "Anomaly:Ground Event / Encounter Person / Animal / Bird",
    "Anomaly:Ground Event / Encounter FOD",
    "Anomaly:Ground Event / Encounter Object",
    "Anomaly:Inflight Event / Encounter Object",
    "Anomaly:Ground Event / Encounter Jet Blast",
    "Anomaly:Ground Excursion Ramp",
    "Anomaly:Ground Excursion Runway",
    "Anomaly:Ground Excursion Taxiway",
    "Anomaly:Ground Incursion Ramp",
    "Anomaly:Ground Incursion Runway",
    "Anomaly:Ground Incursion Taxiway",
    "Anomaly:ATC Issue All Types",
    "Anomaly:Aircraft Equipment Problem Critical",
    "Anomaly:Aircraft Equipment Problem Less Severe",
    "Anomaly:Ground Event / Encounter Ground Equipment Issue",
    "Contributing Factors:ATC Equipment / Nav Facility / Buildings",
    "Contributing Factors:Equipment / Tooling",
    "Contributing Factors:Software and Automation",

    # Human Preconditions
    "Human Factors:Communication Breakdown",
    "Human Factors:Confusion",
    "Human Factors:Distraction",
    "Human Factors:Situational Awareness",
    "Human Factors:Time Pressure",
    "Human Factors:Workload",
    "Human Factors:Troubleshooting",
    "Human Factors:Training / Qualification",
    "Human Factors:Human-Machine Interface",
    "Human Factors:Other / Unknown",
    "Human Factors:Fatigue",
    "Human Factors:Physiological - Other",
]

# ---------- Load dataset ----------
cols = L3_raw + L2_raw

df = pd.read_excel(
    r"C:\llm-hfacs\data\ASRS_all_factors.xlsx",
    usecols=cols,
    engine="openpyxl"
).fillna(0)


# ---------- Clean names ----------
def clean(x):
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")

L3 = { clean(c): c for c in L3_raw }
L2 = { clean(c): c for c in L2_raw }

rows = []

# ---------- Compute P(L2 | L3) ----------
for L3_clean, L3_col in L3.items():

    df_L3_active = df[df[L3_col] == 1]
    n = len(df_L3_active)

    if n == 0:
        continue

    row = {"L3_factor": L3_clean, "N_cases": n}

    for L2_clean, L2_col in L2.items():
        prob = df_L3_active[L2_col].mean()
        row[f"P_{L2_clean}"] = round(prob, 4)

    rows.append(row)

result_df = pd.DataFrame(rows)

# ---------- Save ----------
result_df.to_csv("L3_to_L2_probabilities.csv", index=False)
print(result_df)
print("Saved: L3_to_L2_probabilities.csv")
