import yaml
import pandas as pd
from data.seperator import extract_factors
from features.conditional_prob import compute_conditional_probabilities
from models.prediction import compute_full_chain
from utils import skip_run

# The configuration file
with open("./configs/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)


with skip_run("skip", "extract_factors") as check, check():
    extract_factors(config)


with skip_run("skip", "compute_probability") as check, check():
    df = pd.read_csv(config["output_path"]).fillna(0)
    compute_conditional_probabilities(df, "level_4", "level_3", config)
    compute_conditional_probabilities(df, "level_3", "level_2", config)
    compute_conditional_probabilities(df, "level_2", "level_1", config)


with skip_run("run", "compute_final_probability") as check, check():
    result = compute_full_chain(levels=[4, 3, 2, 1])
    print(result)
