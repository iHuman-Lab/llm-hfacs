from __future__ import annotations

from pathlib import Path
import os
import re

import pandas as pd


def resolve_openai_api_key(llm_cfg: dict) -> str:
    config_key = llm_cfg.get("api_key")
    source = "llm.api_key" if config_key else "OPENAI_API_KEY"
    key = (config_key or os.getenv("OPENAI_API_KEY", "")).strip()
    if not key:
        raise RuntimeError(
            "Missing API key. Set llm.api_key in config.yaml or OPENAI_API_KEY env var."
        )
    if re.fullmatch(r"sk-[.]+", key) or "..." in key:
        if not config_key and source == "OPENAI_API_KEY":
            raise RuntimeError(
                "llm.api_key is empty in configs/config.yaml, and OPENAI_API_KEY "
                "is set to a placeholder value. Save a real key in config.yaml or "
                "replace the OPENAI_API_KEY environment variable."
            )
        raise RuntimeError(
            f"{source} is set to a placeholder value, not a real OpenAI API key."
        )
    return key


def _reader(path):
    if Path(path).suffix.lower() == ".csv":
        return lambda p, **kw: pd.read_csv(p, low_memory=False, **kw)
    return pd.read_excel


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
