import pandas as pd
import numpy as np
from functools import reduce

import pandas as pd


def compute_full_chain(levels=[4, 3, 2, 1]):
    matrices = []  # list of numpy matrices in order
    index_names = None  # left-side names for each matrix
    final_col_names = None  # right-side names of last matrix

    for level in levels[:-1]:
        # load file dynamically
        file = (
            f"./data/processed/level_{level - 1}_probabilities_given_level_{level}.csv"
        )
        df = pd.read_csv(file)

        # left and right factor names from file
        left_names = df[f"level_{level}_factors"].tolist()

        # probability column names in this file
        prob_cols = [c for c in df.columns if c.startswith("P_")]

        # extract matrix
        M = df[prob_cols].to_numpy()
        matrices.append(M)

        # store index names only for the first matrix
        if index_names is None:
            index_names = left_names

        # save column names from final matrix
        final_col_names = [c.replace("P_", "") for c in prob_cols]

    # multiply all matrices in left-to-right order
    Full = matrices[0]
    for M in matrices[1:]:
        Full = Full.dot(M)

    # build result dataframe
    result = pd.DataFrame(Full, index=index_names, columns=final_col_names)

    return result
