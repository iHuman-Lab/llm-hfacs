from pathlib import Path
import sys

# Ensure src is on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from features.hfacs_dag import build_dag_from_edges, plot_hfacs_layered  # noqa: E402

EDGE_CSV = Path("data/processed/hfacs_dag_edges.csv")
OUT_PDF = Path("data/processed/hfacs_dag_plot.pdf")

if not EDGE_CSV.exists():
    raise SystemExit(f"Missing edge CSV: {EDGE_CSV}")

edge_df = pd.read_csv(EDGE_CSV)
G = build_dag_from_edges(edge_df)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
plot_hfacs_layered(G, save_path=str(OUT_PDF))
print(f"Saved plot to: {OUT_PDF}")
