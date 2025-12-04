import pandas as pd

# ----------------------- Level 2 (Preconditions) -----------------------
L2_raw = [
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

# ----------------------- Level 1 (Unsafe Acts) -----------------------
L1_raw = [
    "Anomaly:Airspace Violation All Types",
    "Anomaly:Conflict Airborne Conflict",
    "Anomaly:Conflict Ground Conflict, Critical",
    "Anomaly:Conflict Ground Conflict, Less Severe",
    "Anomaly:Conflict NMAC",
    "Anomaly:Deviation - Altitude Excursion From Assigned Altitude",
    "Anomaly:Deviation - Altitude Overshoot",
    "Anomaly:Deviation - Altitude Undershoot",
    "Anomaly:Deviation - Speed All Types",
    "Anomaly:Deviation - Track / Heading All Types",
    "Anomaly:Deviation / Discrepancy - Procedural Clearance",
    "Anomaly:Deviation / Discrepancy - Procedural FAR",
    "Anomaly:Deviation / Discrepancy - Procedural Hazardous Material Violation",
    "Anomaly:Deviation / Discrepancy - Procedural Landing Without Clearance",
    "Anomaly:Deviation / Discrepancy - Procedural MEL / CDL",
    "Anomaly:Deviation / Discrepancy - Procedural Other / Unknown",
    "Anomaly:Deviation / Discrepancy - Procedural Published Material / Policy",
    "Anomaly:Deviation / Discrepancy - Procedural Unauthorized Flight Operations (UAS)",
    "Anomaly:Deviation / Discrepancy - Procedural Weight And Balance",
    "Anomaly:Inflight Event / Encounter Loss Of Aircraft Control",
    "Anomaly:Ground Event / Encounter Loss Of Aircraft Control",
    "Anomaly:Inflight Event / Encounter Unstabilized Approach",
    "Anomaly:Inflight Event / Encounter VFR In IMC",
    "Anomaly:Inflight Event / Encounter CFTT / CFIT",
    "Anomaly:Inflight Event / Encounter Fly Away (UAS)",
    "Anomaly:Inflight Event / Encounter Fuel Issue",
    "Anomaly:Ground Event / Encounter Fuel Issue",
    "Anomaly:Ground Event / Encounter Gear Up Landing",
]

# ----------------------- Load only L2 + L1 columns -----------------------
cols = L2_raw + L1_raw

df = pd.read_excel(
    r"C:\llm-hfacs\data\ASRS_all_factors.xlsx",
    usecols=cols,
    engine="openpyxl"
).fillna(0)

# ----------------------- Clean names -----------------------
def clean(x):
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")

L2 = {clean(c): c for c in L2_raw}
L1 = {clean(c): c for c in L1_raw}

rows = []

# ----------------------- Compute P(L1 | L2) -----------------------
for L2_clean, L2_col in L2.items():
    df_active = df[df[L2_col] == 1]
    n = len(df_active)

    if n == 0:
        continue

    row = {"L2_factor": L2_clean, "N_cases": n}

    for L1_clean, L1_col in L1.items():
        row[f"P_{L1_clean}"] = round(df_active[L1_col].mean(), 4)

    rows.append(row)

result_df = pd.DataFrame(rows)

result_df.to_csv("L2_to_L1_probabilities.csv", index=False)
print(result_df)
print("Saved: L2_to_L1_probabilities.csv")
