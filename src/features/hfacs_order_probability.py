from functools import reduce
from itertools import product
from pathlib import Path

import pandas as pd

HFACS_ORDER = [
    # Level 4 — Organizational Influences
    [
        "Organizational_Process",
        "Organizational_Climate",
        "Resource_Management",
    ],
    # Level 3 — Unsafe Supervision
    [
        "Inadequate_Supervision",
    ],
    # Level 2 — Preconditions for Unsafe Acts
    [
        "Condition_of_Operators",
        "Personnel_Factors",
        "Situational_Factors",
    ],
    # Level 1 — Unsafe Acts
    [
        "Error",
        "Violation",
    ],
]


def conditional_probabilities_hfacs(df, parents, children, parent_label):
    """
    Computes P(child | parent) only for HFACS-allowed directions.
    """

    rows = []

    for parent in parents:
        if parent not in df.columns:
            raise ValueError(f"Missing HFACS column: {parent}")

        df_parent = df[df[parent] == 1]
        n = len(df_parent)

        row = {
            f"{parent_label}_category": parent,
            "N_cases": int(n),
        }

        if n == 0:
            for child in children:
                row[f"P_{child}"] = 0.0
        else:
            for child in children:
                if child not in df.columns:
                    raise ValueError(f"Missing HFACS column: {child}")
                row[f"P_{child}"] = round(df_parent[child].mean(), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_hfacs_ordered_probabilities(
    df,
    hfacs_order=HFACS_ORDER,
    output_dir="./data/processed",
):
    """
    Computes conditional probabilities strictly following HFACS order.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = []

    for i in range(len(hfacs_order) - 1):
        parents = hfacs_order[i]
        children = hfacs_order[i + 1]

        level_from = 4 - i
        level_to = 3 - i

        prob_df = conditional_probabilities_hfacs(
            df,
            parents=parents,
            children=children,
            parent_label=f"HFACS_L{level_from}",
        )

        filename = f"HFACS_L{level_from}_to_L{level_to}.csv"
        prob_df.to_csv(Path(output_dir) / filename, index=False)

        results.append(prob_df)

    return results


def compute_all_full_hfacs_chains(
    hfacs_order,
    processed_dir="./data/processed",
):
    """
    Computes all full HFACS chains (L4 → L3 → L2 → L1),
    including both Error and Violation.
    """

    results = []
    L4, L3, L2, L1 = hfacs_order

    for chain in product(L4, L3, L2, L1):
        probs = []

        for i in range(3):  # L4→L3, L3→L2, L2→L1
            parent, child = chain[i], chain[i + 1]
            file = Path(processed_dir) / f"HFACS_L{4 - i}_to_L{3 - i}.csv"

            df = pd.read_csv(file)
            parent_col = [c for c in df.columns if c.endswith("_category")][0]
            row = df[df[parent_col] == parent]

            probs.append(row[f"P_{child}"].iloc[0])

        results.append(
            {
                "Chain": " → ".join(chain),
                "Chained_Probability": round(reduce(lambda x, y: x * y, probs), 6),
            }
        )

    return pd.DataFrame(results)


def compute_combined_hfacs_matrix(
    hfacs_order,
    processed_dir="./data/processed",
    filename="HFACS_L4_to_L1_combined.csv",
):
    """
    Combines HFACS conditional probability tables via matrix multiplication
    and SAVES the result to disk.

    Output:
        ./data/processed/HFACS_L4_to_L1_combined.csv
    """

    matrices = []
    index_names = None
    final_col_names = None

    # iterate L4→L3, L3→L2, L2→L1
    for i in range(len(hfacs_order) - 1):
        level_from = 4 - i
        level_to = 3 - i

        file = Path(processed_dir) / f"HFACS_L{level_from}_to_L{level_to}.csv"
        df = pd.read_csv(file)

        parent_col = [c for c in df.columns if c.endswith("_category")][0]
        prob_cols = [c for c in df.columns if c.startswith("P_")]

        M = df[prob_cols].to_numpy()
        matrices.append(M)

        if index_names is None:
            index_names = df[parent_col].tolist()

        final_col_names = [c.replace("P_", "") for c in prob_cols]

    # combine matrices
    Full = matrices[0]
    for M in matrices[1:]:
        Full = Full.dot(M)

    result_df = pd.DataFrame(
        Full,
        index=index_names,
        columns=final_col_names,
    )

    # ===== SAVE =====
    output_path = Path(processed_dir) / filename
    result_df.to_csv(output_path)

    print(f"[INFO] Combined HFACS matrix saved to {output_path}")

    return result_df
