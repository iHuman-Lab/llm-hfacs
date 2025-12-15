import pandas as pd
from pathlib import Path
from functools import reduce
from itertools import product

# ============================================================
# HFACS ORDER (THEORY-CONSTRAINED)
# ============================================================

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
        "Failure_to_Correct",
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


# ============================================================
# CONDITIONAL PROBABILITIES (HFACS-ORDERED ONLY)
# ============================================================

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


# ============================================================
# COMPUTE ALL HFACS-ORDERED PROBABILITY TABLES
# ============================================================

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


# ============================================================
# COMPUTE A FULL HFACS CHAIN (SCALAR, VALID)
# ============================================================

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
            file = Path(processed_dir) / f"HFACS_L{4-i}_to_L{3-i}.csv"

            df = pd.read_csv(file)
            parent_col = [c for c in df.columns if c.endswith("_category")][0]
            row = df[df[parent_col] == parent]

            probs.append(row[f"P_{child}"].iloc[0])

        results.append({
            "Chain": " → ".join(chain),
            "Chained_Probability": round(reduce(lambda x, y: x * y, probs), 6),
        })

    return pd.DataFrame(results)
