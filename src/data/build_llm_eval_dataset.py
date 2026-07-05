from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from models.random_forest import make_four_class_target_from_config

_LABEL_NAMES = {0: "Neither", 1: "Error", 2: "Violation", 3: "Both"}
_N_CLASSES = len(_LABEL_NAMES)


def build_llm_dataset(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample a balanced dataset and split into LLM-train and held-out test sets.

    Controlled entirely by config['dataset']:
      source       — filename inside ghfacs_data_dir to sample from
      n_total      — total accidents (must be divisible by 4)
      train_split  — fraction sent to the LLM for classification (e.g. 0.8)
      train_output — filename saved in ghfacs_data_dir for the LLM input
      test_output  — filename saved in ghfacs_data_dir for evaluation
    """
    cfg = config["dataset"]
    data_dir = Path(config["paths"]["ghfacs_data_dir"])
    random_state = config["models"]["random_state"]

    n_total = int(cfg["n_total"])
    if n_total % _N_CLASSES != 0:
        raise ValueError(
            f"dataset.n_total ({n_total}) must be divisible by {_N_CLASSES}"
        )
    n_per_class = n_total // _N_CLASSES

    train_split = float(cfg["train_split"])
    if not (0 < train_split < 1):
        raise ValueError(f"dataset.train_split must be in (0, 1), got {train_split}")
    test_split = round(1.0 - train_split, 10)

    df = pd.read_csv(data_dir / cfg["source"], low_memory=False)
    y = make_four_class_target_from_config(df, config)

    parts = []
    for cls, name in _LABEL_NAMES.items():
        idx = y[y == cls].index
        if len(idx) < n_per_class:
            raise ValueError(
                f"Class '{name}' has only {len(idx)} rows, need {n_per_class}"
            )
        parts.append(df.loc[idx].sample(n=n_per_class, random_state=random_state))

    sampled = (
        pd.concat(parts)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    y_sampled = make_four_class_target_from_config(sampled, config)

    train_df, test_df = train_test_split(
        sampled,
        test_size=test_split,
        stratify=y_sampled,
        random_state=random_state,
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    y_train = make_four_class_target_from_config(train_df, config)
    y_test = make_four_class_target_from_config(test_df, config)

    print(f"\nLLM dataset: {n_total} total, {n_per_class} per class")
    print(f"  Train ({train_split:.0%}): {len(train_df)} rows")
    print(f"  Test  ({test_split:.0%}): {len(test_df)} rows")
    print("  Class breakdown:")
    for cls, name in _LABEL_NAMES.items():
        print(
            f"    {name:10s}: train={int((y_train == cls).sum()):4d}, "
            f"test={int((y_test == cls).sum()):4d}"
        )

    train_out = data_dir / cfg["train_output"]
    test_out = data_dir / cfg["test_output"]
    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    print(f"\nSaved train → {train_out}")
    print(f"Saved test  → {test_out}")

    return train_df, test_df
