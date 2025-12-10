import pandas as pd


def clean_text(x):
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")


def conditional_probabilities(df, input, output, config):
    input_level = {clean_text(c): c for c in config[input]}
    output_level = {clean_text(c): c for c in config[output]}

    rows = []

    for input_levelclean, input_levelcol in input_level.items():
        df_active = df[df[input_levelcol] == 1]
        n = len(df_active)

        if n == 0:
            continue

        row = {f"{input}_factors": input_levelclean, "N_cases": n}
        for output_levelclean, output_levelcol in output_level.items():
            row[f"P_{output_levelclean}"] = round(df_active[output_levelcol].mean(), 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        f"./data/processed/{output}_probabilities_given_{input}.csv",
        index=False,
    )


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


def conditional_probabilities_subcategory(df, parent_cols, child_cols, parent_name):
    """
    Computes P(child | parent) for each pair of subcategories.
    """
    rows = []

    for p_subcat, p_col in parent_cols.items():
        df_active = df[df[p_col] == 1]
        n = len(df_active)

        row = {f"{parent_name}_subcategory": p_subcat, "N_cases": int(n)}

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
