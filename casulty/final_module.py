import pandas as pd
import numpy as np



df_43 = pd.read_csv("L4_to_L3_probabilities.csv")  
df_32 = pd.read_csv("L3_to_L2_probabilities.csv")  
df_21 = pd.read_csv("L2_to_L1_probabilities.csv")  



L3_names = df_32["L3_factor"].tolist()

L2_names = df_21["L2_factor"].tolist()


L4_names = df_43["L4_factor"].tolist()


L1_prob_cols = [c for c in df_21.columns if c.startswith("P_")]
L1_names = [c.replace("P_", "") for c in L1_prob_cols]




# A: P(L3 | L4)   
A_cols = [f"P_{name}" for name in L3_names]
A = df_43[A_cols].to_numpy()

# B: P(L2 | L3)   
B_cols = [f"P_{name}" for name in L2_names]
B = df_32[B_cols].to_numpy()

# C: P(L1 | L2)   shape: (n_L2, n_L1)
C_cols = [f"P_{name}" for name in L1_names]
C = df_21[C_cols].to_numpy()

print("A (L4→L3) shape:", A.shape)
print("B (L3→L2) shape:", B.shape)
print("C (L2→L1) shape:", C.shape)



Full = A.dot(B).dot(C)   


full_df = pd.DataFrame(Full, index=L4_names, columns=L1_names)

# Save to CSV
full_df.to_csv("L4_to_L1_full_chain_probabilities.csv")
print(full_df.head())
print("Saved: L4_to_L1_full_chain_probabilities.csv")



def get_L1_given_L4(l4_name: str):
    """
    Return a sorted probability vector P(L1 | given L4 factor).
    """
    if l4_name not in full_df.index:
        raise ValueError(f"{l4_name} not found in L4 factors.")
    row = full_df.loc[l4_name]
    
    return row.sort_values(ascending=False)


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
