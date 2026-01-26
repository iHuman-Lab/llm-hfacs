import pandas as pd
from pathlib import Path


def conditional_probabilities(
    df: pd.DataFrame,
    parent_categories: list[str],
    child_categories: list[str],
    parent_label: str = "HFACS",
) -> pd.DataFrame:
    """
    Computes P(child_category | parent_category)
    using HFACS category columns that already exist in df.
    """

    rows = []

    for parent in parent_categories:
        if parent not in df.columns:
            raise ValueError(f"Missing HFACS column: {parent}")

        df_parent = df[df[parent] == 1]
        n = len(df_parent)

        row = {
            f"{parent_label}_category": parent,
            "N_cases": int(n),
        }

        if n == 0:
            for child in child_categories:
                row[f"P_{child}"] = 0.0
        else:
            for child in child_categories:
                if child not in df.columns:
                    raise ValueError(f"Missing HFACS column: {child}")
                row[f"P_{child}"] = round(df_parent[child].mean(), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def compute_all_hfacs_probabilities(
    df: pd.DataFrame,
    hfacs_map: dict,
    output_dir: str = "./data/processed",
):
    """
    Computes P(Category_j | Category_i) for HFACS categories
    explicitly defined in hfacs_map.
    """

    hfacs_categories = list(hfacs_map.keys())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = {}

    for parent in hfacs_categories:
        parent_list = [parent]
        child_list = [c for c in hfacs_categories if c != parent]

        prob_df = conditional_probabilities(
            df,
            parent_categories=parent_list,
            child_categories=child_list,
            parent_label="HFACS",
        )

        output_path = Path(output_dir) / f"P_given_{parent}.csv"
        prob_df.to_csv(output_path, index=False)

        results[parent] = prob_df

    return results
