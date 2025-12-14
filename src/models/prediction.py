import pandas as pd
from pathlib import Path
from functools import reduce


def compute_full_chain(
    chain: list[str],
    processed_dir: str = "./data/processed",
):
    """
    Computes chained HFACS probabilities:
    P(chain[i+1] | chain[i]) multiplied along the chain.

    Returns a single-row DataFrame.
    """

    probs = []
    labels = []

    for i in range(len(chain) - 1):
        parent = chain[i]
        child = chain[i + 1]

        file = Path(processed_dir) / f"P_given_{parent}.csv"
        if not file.exists():
            raise FileNotFoundError(f"Missing file: {file}")

        df = pd.read_csv(file)

        col = f"P_{child}"
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {file}")

        p = df[col].iloc[0]
        probs.append(p)
        labels.append(f"P({child}|{parent})")

    # multiply scalars
    final_prob = reduce(lambda x, y: x * y, probs)

    return pd.DataFrame(
        {
            "Chain": [" → ".join(chain)],
            "Chained_Probability": [round(final_prob, 6)],
        }
    )
