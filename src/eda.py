import argparse
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_PRECONDITION_COLUMNS = [
    "PE100",
    "PE200",
    "PP100",
    "PT100",
    "PC100",
    "PC200",
    "PC300",
]

PRECONDITION_GROUPS = {
    "Situational_Factors": ["PE100", "PE200"],
    "Personnel_Factors": ["PP100", "PT100"],
    "Condition_of_Operators": ["PC100", "PC200", "PC300"],
}

CLASS_ORDER = ["AE100 only", "AE200 only", "Both", "None"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported input file type: {path}")


def is_present(series: pd.Series) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.ne("") & values.str.lower().ne("nan")


def derive_four_class(df: pd.DataFrame) -> pd.Series:
    if "AE100" not in df.columns or "AE200" not in df.columns:
        raise ValueError("Dataset must contain AE100 and AE200 columns.")

    ae100 = is_present(df["AE100"])
    ae200 = is_present(df["AE200"])

    labels = pd.Series("None", index=df.index)
    labels.loc[ae100 & ~ae200] = "AE100 only"
    labels.loc[~ae100 & ae200] = "AE200 only"
    labels.loc[ae100 & ae200] = "Both"
    return labels


def existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def compute_counts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    precondition_cols = existing_columns(df, DEFAULT_PRECONDITION_COLUMNS)
    precondition_counts = (
        pd.Series(
            {col: is_present(df[col]).sum() for col in precondition_cols},
            name="Row_Count",
        )
        .rename_axis("Precondition_Code")
        .reset_index()
    )

    class_counts = (
        derive_four_class(df)
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename("Accident_Count")
        .rename_axis("Four_Class")
        .reset_index()
    )

    return {"precondition_counts": precondition_counts, "class_counts": class_counts}


def print_counts(counts: dict[str, pd.DataFrame]) -> None:
    print("\n=== Rows with Each Precondition Code ===")
    print(counts["precondition_counts"].to_string(index=False))
    print("\n=== Accidents per Four-Class Label ===")
    print(counts["class_counts"].to_string(index=False))


def build_eda(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    precondition_cols = existing_columns(df, DEFAULT_PRECONDITION_COLUMNS)
    if not precondition_cols:
        raise ValueError(
            "No precondition columns found. Expected at least one of: "
            + ", ".join(DEFAULT_PRECONDITION_COLUMNS)
        )

    present = pd.DataFrame(
        {col: is_present(df[col]) for col in precondition_cols},
        index=df.index,
    )

    detail_count = present.sum(axis=1).astype(int)
    four_class = derive_four_class(df)

    group_present = pd.DataFrame(index=df.index)
    for group, columns in PRECONDITION_GROUPS.items():
        cols = existing_columns(df, columns)
        if cols:
            group_present[group] = present[cols].any(axis=1)

    group_count = group_present.sum(axis=1).astype(int)

    accident_level = pd.DataFrame(
        {
            "Four_Class": four_class,
            "Precondition_Detail_Count": detail_count,
            "Precondition_Group_Count": group_count,
        }
    )

    id_cols = existing_columns(df, ["ev_id", "ntsb_no"])
    for col in reversed(id_cols):
        accident_level.insert(0, col, df[col])

    for col in precondition_cols:
        accident_level[col] = present[col].astype(int)
    for col in group_present.columns:
        accident_level[col] = group_present[col].astype(int)

    detail_distribution = (
        accident_level["Precondition_Detail_Count"]
        .value_counts()
        .rename_axis("Precondition_Detail_Count")
        .reset_index(name="Accident_Count")
        .sort_values("Precondition_Detail_Count")
    )

    group_distribution = (
        accident_level["Precondition_Group_Count"]
        .value_counts()
        .rename_axis("Precondition_Group_Count")
        .reset_index(name="Accident_Count")
        .sort_values("Precondition_Group_Count")
    )

    class_distribution = (
        accident_level["Four_Class"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .rename_axis("Four_Class")
        .reset_index(name="Accident_Count")
    )

    detail_by_class = pd.crosstab(
        accident_level["Precondition_Detail_Count"],
        accident_level["Four_Class"],
    ).reindex(columns=CLASS_ORDER, fill_value=0)

    group_by_class = pd.crosstab(
        accident_level["Precondition_Group_Count"],
        accident_level["Four_Class"],
    ).reindex(columns=CLASS_ORDER, fill_value=0)

    precondition_by_class = pd.DataFrame(
        {cls: present.loc[four_class == cls].sum().astype(int) for cls in CLASS_ORDER}
    )
    precondition_by_class.insert(0, "Total", present.sum().astype(int))

    return {
        "accident_level": accident_level,
        "detail_distribution": detail_distribution,
        "group_distribution": group_distribution,
        "class_distribution": class_distribution,
        "detail_by_class": detail_by_class,
        "group_by_class": group_by_class,
        "precondition_by_class": precondition_by_class,
    }


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def default_input_from_config(config: dict) -> Path:
    return Path(config["paths"]["ghfacs_data_dir"]) / config["llm"]["input"]


def print_report(tables: dict[str, pd.DataFrame], total: int) -> None:
    print(f"\nTotal accidents: {total}")

    print("\n=== Four-Class Distribution ===")
    print(tables["class_distribution"].to_string(index=False))

    print("\n=== Accidents by Number of Detailed Preconditions ===")
    print(tables["detail_distribution"].to_string(index=False))

    print("\n=== Detailed Precondition Count by Four-Class Label ===")
    print(tables["detail_by_class"].to_string())

    print("\n=== Accidents by Number of Precondition Groups ===")
    print(tables["group_distribution"].to_string(index=False))

    print("\n=== Precondition Group Count by Four-Class Label ===")
    print(tables["group_by_class"].to_string())

    print("\n=== Individual Preconditions by Four-Class Label ===")
    print(tables["precondition_by_class"].to_string())


def save_tables(
    tables: dict[str, pd.DataFrame], output_dir: Path, prefix: str = "eda_"
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output_dir / f"{prefix}{name}.csv", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description="EDA for GHFACS preconditions and four unsafe-act classes."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=root / "configs" / "config.yaml",
        help="Path to config.yaml.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional dataset path. Defaults to llm.input under paths.ghfacs_data_dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "intermediate",
        help="Directory where EDA CSV files will be written.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print the report without writing CSV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)

    input_path = (
        args.input.resolve()
        if args.input
        else default_input_from_config(config, config_path)
    )
    df = read_table(input_path)

    print(f"Input file: {input_path}")
    print_counts(compute_counts(df))

    tables = build_eda(df)
    print_report(tables, len(df))

    if not args.no_save:
        save_tables(tables, args.output_dir)
        print(f"\nSaved EDA tables to: {args.output_dir}")


if __name__ == "__main__":
    main()
