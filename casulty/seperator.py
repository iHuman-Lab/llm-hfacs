import pandas as pd
import os

def extract_factors(input_path, output_path, keywords=None):
    """
    Reads ASRS .xls file safely, converts to .xlsx, extracts factors,
    creates binary factor columns, and saves final ASRS_all_factors.xlsx.
    """

    # ---------------------------------------------------------
    # 0. KEYWORDS (default)
    # ---------------------------------------------------------
    if keywords is None:
        keywords = {
            "Human Factors": "Human",
            "Anomaly": "Anomaly",
            "Contributing Factors": "Contributing"
        }

    # ---------------------------------------------------------
    # 1. READ .xls SAFELY
    # ---------------------------------------------------------
    print("Reading XLS using engine='xlrd' ...")

    df = pd.read_excel(input_path, header=1, engine="xlrd")

    # Save original to .xlsx first
    print("Saving temporary XLSX ...")
    temp_xlsx = "temp_converted.xlsx"
    df.to_excel(temp_xlsx, index=False)

    # Now load clean .xlsx (safe for all engines)
    df = pd.read_excel(temp_xlsx)

    # ---------------------------------------------------------
    # 2. FIND RELEVANT FACTOR COLUMNS
    # ---------------------------------------------------------
    cols = {}
    for cname in df.columns:
        for key, kw in keywords.items():
            if kw in str(cname):
                cols[key] = cname

    # ---------------------------------------------------------
    # 3. CREATE BINARY FACTOR COLUMNS
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 4. DROP ANY 'nan' COLUMNS
    # ---------------------------------------------------------
    df = df.loc[:, ~df.columns.str.contains("nan")]

    # ---------------------------------------------------------
    # 5. SAVE FINAL ASRS_all_factors.xlsx
    # ---------------------------------------------------------
    df.to_excel(output_path, index=False)
    print("Saved final:", output_path)

    # Remove temporary file
    if os.path.exists(temp_xlsx):
        os.remove(temp_xlsx)

    return {"columns": created_cols, "output_file": output_path}
