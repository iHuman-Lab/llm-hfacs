import pandas as pd


def load_raw_dataset(path, save_cleaned=True):
    # Load CSV with two-row header
    df = pd.read_csv(path, header=[0, 1])

    # Flatten MultiIndex columns → "Events_Anomaly"
    df.columns = df.columns.map(lambda x: f"{x[0]}_{x[1]}".strip())

    # Save cleaned header file
    if save_cleaned:
        df.to_csv("./data/processed/step1_cleaned_header.csv", index=False)

    return df


def extract_factor_columns(df, source_columns, save_step=True):
    result = df.copy()

    # Collect unique factors for each key
    factor_sets = {key: set() for key in source_columns}

    def split_factors(x):
        if not isinstance(x, str):
            return []
        return [p.strip() for p in x.split(";") if p.strip()]

    # Build factor sets
    for key, col in source_columns.items():
        for val in df[col]:
            factor_sets[key].update(split_factors(val))

    # Create binary 1/0 factor columns
    for key, col in source_columns.items():
        for factor in sorted(factor_sets[key]):  # alphabetical consistency
            new_col = f"{key}_{factor}"
            result[new_col] = df[col].apply(
                lambda x: int(factor in split_factors(x))
            )

    # Save step result
    if save_step:
        result.to_csv("./data/processed/step2_factor_expanded.csv", index=False)

    return result


def create_hfacs_categories(df, category_map, save_step=True):
    result = df.copy()

    for cat, cols in category_map.items():

        # Only include existing columns
        valid = [c for c in cols if c in result.columns]
        if not valid:
            continue

        # HFACS category is 1 if ANY contributing factor is present
        result[cat] = result[valid].sum(axis=1).apply(lambda v: 1 if v > 0 else 0)

    # Save HFACS step
    if save_step:
        result.to_csv("./data/processed/step3_hfacs_categories.csv", index=False)

    return result


def save_outputs(df, csv_path, excel_path):
    df.to_csv(csv_path, index=False)
    df.to_excel(excel_path, index=False)