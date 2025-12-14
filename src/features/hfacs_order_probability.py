import pandas as pd
from pathlib import Path
from functools import reduce

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

def compute_full_hfacs_chain(
    chain,
    processed_dir="./data/processed",
):
    """
    Computes chained HFACS probability:
    P(B|A) × P(C|B) × ...
    """

    probs = []

    for i in range(len(chain) - 1):
        parent = chain[i]
        child = chain[i + 1]

        file = Path(processed_dir) / f"P_given_{parent}.csv"
        if not file.exists():
            raise FileNotFoundError(f"Missing probability file: {file}")

        df = pd.read_csv(file)

        col = f"P_{child}"
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {file}")

        probs.append(df[col].iloc[0])

    final_prob = reduce(lambda x, y: x * y, probs)

    return pd.DataFrame(
        {
            "Chain": [" → ".join(chain)],
            "Chained_Probability": [round(final_prob, 6)],
        }
    )


# ============================================================
# EXAMPLE USAGE (COMMENT OUT IF CALLED FROM main.py)
# ============================================================

if __name__ == "__main__":

    # Load your processed dataset with HFACS categories already created
    df = pd.read_csv("./data/processed/processed_output.csv")

    print("[INFO] Computing HFACS-ordered conditional probabilities...")
    compute_hfacs_ordered_probabilities(df)

    print("[INFO] Computing example HFACS chain...")
    chain = [
        "Organizational_Process",
        "Inadequate_Supervision",
        "Error",
    ]

    chain_df = compute_full_hfacs_chain(chain)
    print(chain_df)
