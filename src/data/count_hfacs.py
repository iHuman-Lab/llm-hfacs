import csv
from pathlib import Path
from typing import Dict, Tuple


def count_hfacs(
    filepath: Path | str = "data/processed/step3_hfacs_categories.csv",
) -> Tuple[Dict[str, int], int]:
    """Count HFACS boolean/category columns in the processed HFACS CSV.

    Returns (counts_dict, total_rows).
    """
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # known HFACS keys (leaves + higher-level categories)
        hfacs_keys = [
            "Error",
            "Violation",
            "Situational_Factors",
            "Personnel_Factors",
            "Condition_of_Operators",
            "Inadequate_Supervision",
            "Failed_to_Correct_Problem",
            "Planned_Inappropriate_Operations",
            "Supervisory_Violation",
            "Organizational_Climate",
            "Resource_Management/Organizational_Process",
        ]

        # intersect with actual columns (some may be normalized differently)
        cols = [c for c in reader.fieldnames if c in hfacs_keys]
        if not cols:
            # fallback: pick header names that contain HFACS tokens
            tokens = [
                "Error",
                "Violation",
                "Inadequate",
                "Personnel",
                "Situational",
                "Condition",
                "Organizational",
                "Resource",
            ]
            cols = [c for c in reader.fieldnames if any(tok in c for tok in tokens)]

        counts = {c: 0 for c in cols}
        rows = 0
        for r in reader:
            rows += 1
            for c in cols:
                v = r.get(c, "")
                try:
                    val = int(float(v))
                except Exception:
                    val = (
                        1 if str(v).strip().lower() in ("y", "yes", "true", "t") else 0
                    )
                counts[c] += int(bool(val))

    return counts, rows


def main() -> None:
    counts, rows = count_hfacs()
    print(f"Total rows: {rows}")
    for c, v in counts.items():
        print(f"{c}: {v}")


if __name__ == "__main__":
    main()
