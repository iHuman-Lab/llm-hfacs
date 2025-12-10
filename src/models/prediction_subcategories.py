import pandas as pd
import os



def compute_full_chain():
    """
    Computes:
        P(L1 | L4) = P(L1 | L2) × P(L2 | L3) × P(L3 | L4)
    """
    chain_files = [
        ("L4", "L3", "./data/processed/L3_given_L4.csv"),
        ("L3", "L2", "./data/processed/L2_given_L3.csv"),
        ("L2", "L1", "./data/processed/L1_given_L2.csv"),
    ]

    matrices = []
    index_names = None
    final_col_names = None

    for parent, child, file in chain_files:

        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")

        df = pd.read_csv(file)

        parent_col = f"{parent}_subcategory"
        parent_names = df[parent_col].tolist()

        prob_cols = [c for c in df.columns if c.startswith("P_")]
        M = df[prob_cols].to_numpy()
        matrices.append(M)

        if index_names is None:
            index_names = parent_names

        final_col_names = [c.replace("P_", "") for c in prob_cols]

    # Multiply in sequence
    Full = matrices[0]
    for M in matrices[1:]:
        Full = Full.dot(M)

    # Build final output
    result = pd.DataFrame(Full, index=index_names, columns=final_col_names)
    return result
