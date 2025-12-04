import pandas as pd
import numpy as np

#cleaner
def clean(x):
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")



def compute_L4_to_L3(input_path):
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

    df = pd.read_excel(input_path, engine="openpyxl").fillna(0)

    L3 = {clean(c): c for c in L3_raw}
    L4 = {clean(c): c for c in L4_raw}

    rows = []

    for L4_clean, L4_col in L4.items():
        df_active = df[df[L4_col] == 1]
        n = len(df_active)

        if n == 0:
            continue

        row = {"L4_factor": L4_clean, "N_cases": n}
        for L3_clean, L3_col in L3.items():
            row[f"P_{L3_clean}"] = round(df_active[L3_col].mean(), 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv("function_L4_to_L3_probabilities.csv", index=False)
    return result



def compute_L3_to_L2(input_path):
    L3_raw = [
        "Contributing Factors:Staffing",
        "Contributing Factors:MEL",
        "Contributing Factors:Manuals",
    ]


    _, L2_raw = None, compute_L2_to_L1.__code__.co_consts  
    df = pd.read_excel(input_path, usecols=L3_raw + L2_raw, engine="openpyxl").fillna(0)

    L3 = {clean(c): c for c in L3_raw}
    L2 = {clean(c): c for c in L2_raw}

    rows = []

    for L3_clean, L3_col in L3.items():
        df_active = df[df[L3_col] == 1]
        n = len(df_active)

        if n == 0:
            continue

        row = {"L3_factor": L3_clean, "N_cases": n}
        for L2_clean, L2_col in L2.items():
            row[f"P_{L2_clean}"] = round(df_active[L2_col].mean(), 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv("function_L3_to_L2_probabilities.csv", index=False)
    return result



def compute_L2_to_L1(input_path):
    # Level 2
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

    df = pd.read_excel(input_path, usecols=L2_raw + L1_raw, engine="openpyxl").fillna(0)

    L2 = {clean(c): c for c in L2_raw}
    L1 = {clean(c): c for c in L1_raw}

    rows = []

    for L2_clean, L2_col in L2.items():
        df_active = df[df[L2_col] == 1]
        n = len(df_active)

        if n == 0:
            continue

        row = {"L2_factor": L2_clean, "N_cases": n}
        for L1_clean, L1_col in L1.items():
            row[f"P_{L1_clean}"] = round(df_active[L1_col].mean(), 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv("function_L2_to_L1_probabilities.csv", index=False)
    return result




def compute_full_chain():
    df_43 = pd.read_csv("L4_to_L3_probabilities.csv")
    df_32 = pd.read_csv("L3_to_L2_probabilities.csv")
    df_21 = pd.read_csv("L2_to_L1_probabilities.csv")

    L3_names = df_32["L3_factor"].tolist()
    L2_names = df_21["L2_factor"].tolist()
    L4_names = df_43["L4_factor"].tolist()
    L1_names = [c.replace("P_", "") for c in df_21.columns if c.startswith("P_")]

    A = df_43[[f"P_{c}" for c in L3_names]].to_numpy()
    B = df_32[[f"P_{c}" for c in L2_names]].to_numpy()
    C = df_21[[f"P_{c}" for c in L1_names]].to_numpy()

    Full = A.dot(B).dot(C)
    result = pd.DataFrame(Full, index=L4_names, columns=L1_names)

    result.to_csv("function_L4_to_L1_full_chain_probabilities.csv")
    return result