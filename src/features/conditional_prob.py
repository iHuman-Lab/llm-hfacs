import pandas as pd


def clean_text(x):
    return x.split(":")[-1].strip().replace(" ", "_").replace("/", "_")


def compute_conditional_probabilities(df, input, output, config):
    input_level = {clean_text(c): c for c in config[input]}
    output_level = {clean_text(c): c for c in config[output]}

    rows = []

    for input_levelclean, input_levelcol in input_level.items():
        df_active = df[df[input_levelcol] == 1]
        n = len(df_active)

        if n == 0:
            continue

        row = {f"{input}_factors": input_levelclean, "N_cases": n}
        for output_levelclean, output_levelcol in output_level.items():
            row[f"P_{output_levelclean}"] = round(df_active[output_levelcol].mean(), 4)

        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(
        f"./data/processed/{output}_probabilities_given_{input}.csv",
        index=False,
    )
