import pandas as pd


def extract_factors(config_subcategories, keywords=None):
    """
    Reads ASRS .xls file safely, extracts factors,
    creates binary factor columns, and saves processed dataset.
    """

    if keywords is None:
        keywords = {
            "Human Factors": "Human",
            "Anomaly": "Anomaly",
            "Contributing Factors": "Contributing",
        }

    # Load raw file from config_subcategories
    df = pd.read_excel(config_subcategories["raw_data_path"], header=1)

    # Detect columns that contain factor strings
    cols = {}
    for cname in df.columns:
        for key, kw in keywords.items():
            if kw in str(cname):
                cols[key] = cname

    created_cols = {}

    # Convert multi-value text fields to binary factor columns
    for key, col in cols.items():

        df[col] = df[col].astype(str)

        factors = set()
        for entry in df[col].dropna():
            for f in entry.split(";"):
                f = f.strip()
                if f:
                    factors.add(f)

        factors_list = sorted(factors)

        # Create binary columns for each factor
        for f in factors_list:
            newcol = f"{key}:{f}"
            df[newcol] = df[col].apply(lambda x: 1 if f in x else 0)

        created_cols[key] = factors_list

    # Save final processed file
    df.to_csv(config_subcategories["output_path"], index=False)
