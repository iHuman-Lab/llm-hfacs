import pandas as pd


def build_subcategory_columns(df, config, level_name):
    """
    Builds binary subcategory columns.
    Example output for Level_1:
        Level_1_Errors
        Level_1_Violations
    """
    level_dict = config[level_name]  # dict of subcategory → list of ASRS columns
    output = {}

    for subcat, cols in level_dict.items():

        # YAML empty list → skip
        if not cols:
            print(f"[SKIP] {level_name} → {subcat} (no columns).")
            continue

        # Keep only columns that exist in dataframe
        valid_cols = [c for c in cols if c in df.columns]

        if len(valid_cols) == 0:
            print(f"[SKIP] {level_name} → {subcat} (no valid ASRS columns).")
            continue

        # Create new binary column
        new_col = f"{level_name}_{subcat}"
        df[new_col] = df[valid_cols].max(axis=1)

        output[subcat] = new_col

    return output



def compute_conditional_probabilities(df, parent_cols, child_cols, parent_name, child_name):
    """
    Computes P(child | parent) for each pair of subcategories.
    """
    rows = []

    for p_subcat, p_col in parent_cols.items():

        df_active = df[df[p_col] == 1]
        n = len(df_active)

        row = {
            f"{parent_name}_subcategory": p_subcat,
            "N_cases": int(n)
        }

        # Parent never occurs
        if n == 0:
            for c_subcat in child_cols.keys():
                row[f"P_{c_subcat}"] = 0.0

            rows.append(row)
            continue

        # Normal case
        for c_subcat, c_col in child_cols.items():
            row[f"P_{c_subcat}"] = round(df_active[c_col].mean(), 4)

        rows.append(row)

    return pd.DataFrame(rows)
