from pathlib import Path

import pandas as pd

from features.hfacs_dag import build_dag_from_edges, plot_hfacs_layered

EDGE_CSV = Path("data/processed/hfacs_dag_edges.csv")
OUT_PNG = Path("data/processed/hfacs_dag_plot.png")

if not EDGE_CSV.exists():
    raise SystemExit(f"Missing edge CSV: {EDGE_CSV}")

edge_df = pd.read_csv(EDGE_CSV)
G = build_dag_from_edges(edge_df)

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plot_hfacs_layered(G, save_path=str(OUT_PNG))
print(f"Saved plot to: {OUT_PNG}")
