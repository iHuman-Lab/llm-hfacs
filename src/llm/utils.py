from __future__ import annotations

from pathlib import Path

import pandas as pd


def _reader(path):
    return pd.read_csv if Path(path).suffix.lower() == ".csv" else pd.read_excel


def flatten(x: dict) -> dict:
    return {
        f"{k}__{a}": b
        for k, v in x.items()
        if isinstance(v, dict)
        for a, b in v.items()
    } | {
        k: ", ".join(map(str, v)) if isinstance(v, list) else v
        for k, v in x.items()
        if not isinstance(v, dict)
    }


def finals(x: dict, narr: str, final_cols: set) -> dict:
    return {
        k: v
        for k, v in x.items()
        if any(k == f or k.endswith(f"__{f}") for f in final_cols)
        or k in {"original_index", narr, "skip_reason", "error"}
    }


def four_class(ae100: bool, ae200: bool, classes: list[str]) -> str:
    ae100_only, ae200_only, both, neither = classes
    if ae100 and ae200:
        return both
    if ae100:
        return ae100_only
    if ae200:
        return ae200_only
    return neither
