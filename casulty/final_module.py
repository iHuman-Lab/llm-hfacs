import pandas as pd
import numpy as np

# ============================================
# 1) LOAD THE THREE MODULE TABLES FROM CSV
# ============================================

# These must already exist from your previous scripts:
#  - L4_to_L3_probabilities.csv
#  - L3_to_L2_probabilities.csv
#  - L2_to_L1_probabilities.csv

df_43 = pd.read_csv("L4_to_L3_probabilities.csv")  # columns: L4_factor, N_cases, P_<L3>
df_32 = pd.read_csv("L3_to_L2_probabilities.csv")  # columns: L3_factor, N_cases, P_<L2>
df_21 = pd.read_csv("L2_to_L1_probabilities.csv")  # columns: L2_factor, N_cases, P_<L1>

# ============================================
# 2) ALIGN NAMES AND ORDERS FOR EACH LEVEL
# ============================================

# L3 names: row labels in L3->L2 module
L3_names = df_32["L3_factor"].tolist()

# L2 names: row labels in L2->L1 module
L2_names = df_21["L2_factor"].tolist()

# L4 names: row labels in L4->L3 module
L4_names = df_43["L4_factor"].tolist()

# L1 names: columns in L2->L1 module (strip prefix "P_")
L1_prob_cols = [c for c in df_21.columns if c.startswith("P_")]
L1_names = [c.replace("P_", "") for c in L1_prob_cols]



# ============================================
# 3) BUILD PROBABILITY MATRICES A, B, C
# ============================================

# A: P(L3 | L4)   shape: (n_L4, n_L3)
A_cols = [f"P_{name}" for name in L3_names]
A = df_43[A_cols].to_numpy()

# B: P(L2 | L3)   shape: (n_L3, n_L2)
B_cols = [f"P_{name}" for name in L2_names]
B = df_32[B_cols].to_numpy()

# C: P(L1 | L2)   shape: (n_L2, n_L1)
C_cols = [f"P_{name}" for name in L1_names]
C = df_21[C_cols].to_numpy()

print("A (L4→L3) shape:", A.shape)
print("B (L3→L2) shape:", B.shape)
print("C (L2→L1) shape:", C.shape)

# ============================================
# 4) FULL CHAIN: P(L1 | L4) = A * B * C
# ============================================

Full = A.dot(B).dot(C)   # shape: (n_L4, n_L1)

# Put into DataFrame with meaningful labels
full_df = pd.DataFrame(Full, index=L4_names, columns=L1_names)

# Save to CSV
full_df.to_csv("L4_to_L1_full_chain_probabilities.csv")
print(full_df.head())
print("Saved: L4_to_L1_full_chain_probabilities.csv")

# ============================================
# 5) HELPER: GET DISTRIBUTION FOR ONE L4 FACTOR
# ============================================

def get_L1_given_L4(l4_name: str):
    """
    Return a sorted probability vector P(L1 | given L4 factor).
    """
    if l4_name not in full_df.index:
        raise ValueError(f"{l4_name} not found in L4 factors.")
    row = full_df.loc[l4_name]
    # sort descending by probability
    return row.sort_values(ascending=False)

# Example usage (replace with your actual L4 factor name):
# print(get_L1_given_L4("Company_Policy") )


# ============================================
# 6) OPTIONAL: HEATMAP (ONLY IF matplotlib INSTALLED)
# ============================================

try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(0.4 * len(L1_names), 0.4 * len(L4_names)))
    im = plt.imshow(full_df.values, aspect="auto")
    plt.colorbar(im, label="P(L1 | L4)")

    plt.yticks(ticks=np.arange(len(L4_names)), labels=L4_names)
    plt.xticks(ticks=np.arange(len(L1_names)), labels=L1_names, rotation=90)

    plt.title("Full HFACS Chain: P(Level-1 Unsafe Acts | Level-4 Organizational Factors)")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("matplotlib not installed; skipping heatmap. "
          "Install via `pip install matplotlib` if you want the plot.")
