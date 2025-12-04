import pandas as pd
import os

# ==========================================================
# FILE PATHS
# ==========================================================
input_path = r"C:\llm-hfacs\data\ASRS_dataset.xlsx"        # Already XLSX
output_path = r"C:\llm-hfacs\data\ASRS_all_factors.xlsx"


# ==========================================================
# 0. KEYWORDS
# ==========================================================
keywords = {
    "Human Factors": "Human",
    "Anomaly": "Anomaly",
    "Contributing Factors": "Contributing"
}


# ==========================================================
# 1. READ XLSX SAFELY (NO xlrd)
# ==========================================================
print("Reading XLSX file ...")
df = pd.read_excel(input_path, header=1)   # No engine needed


# ==========================================================
# 2. FIND RELEVANT FACTOR COLUMNS
# ==========================================================
cols = {}
for cname in df.columns:
    for key, kw in keywords.items():
        if kw in str(cname):
            cols[key] = cname


# ==========================================================
# 3. CREATE BINARY FACTOR COLUMNS
# ==========================================================
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


# ==========================================================
# 4. REMOVE ANY `nan` COLUMNS
# ==========================================================
df = df.loc[:, ~df.columns.str.contains("nan")]


# ==========================================================
# 5. SAVE FINAL FILE
# ==========================================================
df.to_excel(output_path, index=False)
print("\nSaved final processed dataset as:", output_path)

print("\nCreated binary factor columns:")
print(created_cols)
