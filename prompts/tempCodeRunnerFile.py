import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, HillClimbSearch, BicScore

# =========================
# 0) LOAD DATA
# =========================
file_path = r"C:\llm-hfacs\data\ASRS_all_factors.xlsx"
df = pd.read_excel(file_path).fillna(0)

# =========================
# 1) NODE LISTS
# =========================

# Level 3: Contributing / organizational factors
CF_nodes = [
    "Contributing Factors:Airport",
    "Contributing Factors:Airspace Structure",
    "Contributing Factors:Chart Or Publication",
    "Contributing Factors:Company Policy",
    "Contributing Factors:Human Factors",
    "Contributing Factors:Incorrect / Not Installed / Unavailable Part",
    "Contributing Factors:Logbook Entry",
    "Contributing Factors:MEL",
    "Contributing Factors:Manuals",
    "Contributing Factors:Procedure",
    "Contributing Factors:Staffing",
    "Contributing Factors:ATC Equipment / Nav Facility / Buildings",
    "Contributing Factors:Aircraft",
    "Contributing Factors:Environment - Non Weather Related",
    "Contributing Factors:Equipment / Tooling",
    "Contributing Factors:Software and Automation",
    "Contributing Factors:Weather",
]

# Level 2: Preconditions
PC_nodes = [
    "Human Factors:Communication Breakdown",
    "Human Factors:Confusion",
    "Human Factors:Distraction",
    "Human Factors:Fatigue",
    "Human Factors:Human-Machine Interface",
    "Human Factors:Other / Unknown",
    "Human Factors:Physiological - Other",
    "Human Factors:Situational Awareness",
    "Human Factors:Time Pressure",
    "Human Factors:Training / Qualification",
    "Human Factors:Troubleshooting",
    "Human Factors:Workload",
    "Anomaly:Ground Event / Encounter Weather / Turbulence",
    "Anomaly:Inflight Event / Encounter Weather / Turbulence",
    "Anomaly:Inflight Event / Encounter Wake Vortex Encounter",
    "Anomaly:Inflight Event / Encounter Bird / Animal",
    "Anomaly:Ground Event / Encounter Person / Animal / Bird",
    "Anomaly:Ground Event / Encounter FOD",
    "Anomaly:Ground Event / Encounter Object",
    "Anomaly:Inflight Event / Encounter Object",
    "Anomaly:Ground Event / Encounter Loss Of VLOS (UAS)",
    "Anomaly:Inflight Event / Encounter Fly Away (UAS)",
]

# Keep only columns that exist in the file
CF_nodes = [c for c in CF_nodes if c in df.columns]
PC_nodes = [c for c in PC_nodes if c in df.columns]

all_nodes = CF_nodes + PC_nodes
df_bn = df[all_nodes].astype("int32")

print("Data shape used for BN:", df_bn.shape)
print("CF nodes:", len(CF_nodes))
print("PC nodes:", len(PC_nodes))

# =========================
# 2) STRUCTURE LEARNING
# =========================
white_list = [(cf, pc) for cf in CF_nodes for pc in PC_nodes]
hc = HillClimbSearch(df_bn)

best_model = hc.estimate(
    scoring_method=BicScore(df_bn),
    max_indegree=3,
    white_list=white_list
)

print("\n=== IMPORTANT DEPENDENCIES (learned BN edges) ===")
for u, v in best_model.edges():
    print(f"{u}  -->  {v}")

# =========================
# 3) PARAMETER LEARNING
# =========================
model = BayesianNetwork(best_model.edges())
model.fit(df_bn, estimator=MaximumLikelihoodEstimator)

print("\nBN fitted successfully with learned sparse structure.")
print("Total nodes in BN:", len(model.nodes()))
print("Total edges in BN:", len(model.edges()))

print("\n=== FINAL NODES INCLUDED IN BAYESIAN NETWORK ===")
print(model.nodes())

# =========================
# 5) CLEAN & ONLY IMPORTANT CPDs (Parents only)
# =========================

# Use only PC nodes that actually remained in the BN
final_PC_nodes = [n for n in PC_nodes if n in model.nodes()]

print("\n===== CLEAN CPDs (ONLY IMPORTANT ONES) =====")

for node in final_PC_nodes:
    parents = model.get_parents(node)
    if parents:
        print(f"\n### {node}  ←  {parents}")
        cpd = model.get_cpds(node)

        # CASE 1: Node has ONE parent  → 2D table → OK
        if len(parents) == 1:
            df_cpd = pd.DataFrame(cpd.values,
                                  columns=[f"{parents[0]}=0", f"{parents[0]}=1"])
            df_cpd.index = [f"{node}=0", f"{node}=1"]
            print(df_cpd)
