

import pandas as pd

def extract_factors(input_path, output_path, keywords=None):
    """
    Extract factor columns, create binary factor columns, and export Excel file.

    Parameters:
        input_path (str): path of source Excel file.
        output_path (str): where to save output.
        keywords (dict): optional keyword dictionary:
            {"Human Factors": "Human", "Anomaly": "Anomaly", ...}
    """

    if keywords is None:
        keywords = {
            "Human Factors": "Human",
            "Anomaly": "Anomaly",
            "Contributing Factors": "Contributing"
        }

    df = pd.read_excel(input_path, header=1)

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
            for f in entry.split(';'):
                f = f.strip()
                if f:
                    factors.add(f)
        factors_list = sorted(factors)
        for f in factors_list:
            newcol = f"{key}:{f}"
            df[newcol] = df[col].apply(lambda x: 1 if f in x else 0)
        created_cols[key] = factors_list

    df = df.loc[:, ~df.columns.str.contains('nan')]
    df.to_excel(output_path, index=False)

    return {"columns": created_cols, "output_file": output_path}
