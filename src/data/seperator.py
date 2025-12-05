import pandas as pd


def extract_factors(config, keywords=None):
    """
    Reads ASRS .xls file safely, converts to .xlsx, extracts factors,
    creates binary factor columns, and saves final ASRS_all_factors.xlsx.
    """
    if keywords is None:
        keywords = {
            "Human Factors": "Human",
            "Anomaly": "Anomaly",
            "Contributing Factors": "Contributing",
        }

    df = pd.read_excel(config["raw_data_path"], header=1)
    cols = {}
    for cname in df.columns:
        for key, kw in keywords.items():
            if kw in str(cname):
                cols[key] = cname

    created_cols = {}

    for key, col in cols.items():
        df[col] = df[col].astype(str)

        factors = set()
        for entry in df[col].dropna():
            for f in entry.split(";"):
                f = f.strip()
                if f:
                    factors.add(f)

        factors_list = sorted(factors)

        for f in factors_list:
            newcol = f"{key}:{f}"
            df[newcol] = df[col].apply(lambda x: 1 if f in x else 0)

        created_cols[key] = factors_list

    df.to_csv(config["output_path"], index=False)
