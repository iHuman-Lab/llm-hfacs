from __future__ import annotations

from pathlib import Path

import networkx as nx
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from features.balancing import balance_undersample
from features.utils import make_three_class_target_from_config
from visualization.dag_graph import plot_dag


# ---------------------------------------------------------------------------
# Shared DAG utilities
# ---------------------------------------------------------------------------

def _matrix_to_edges(categories, M):
    directed, undirected = [], []
    n = len(categories)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = M[i, j], M[j, i]
            if a == -1 and b == 1:
                directed.append((categories[i], categories[j]))
            elif a == 1 and b == -1:
                directed.append((categories[j], categories[i]))
            elif a == -1 and b == -1:
                undirected.append((categories[i], categories[j]))
    return directed, undirected


def _orient_undirected_edges_to_dag(G, undirected_edges, sinks):
    for u, v in undirected_edges:
        if G.has_edge(u, v) or G.has_edge(v, u):
            continue
        for a, b in [(u, v), (v, u)]:
            if a in sinks and b not in sinks:
                continue
            G.add_edge(a, b)
            if nx.is_directed_acyclic_graph(G):
                break
            G.remove_edge(a, b)
    return G


def export_dag_outputs(G: nx.DiGraph, output_dir: str, data_path: str | None = None) -> None:
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(list(G.edges()), columns=["parent", "child"]).to_csv(
        outdir / "learned_dag_edges.csv", index=False
    )

    if data_path is not None and Path(data_path).exists():
        df = pd.read_csv(data_path)
        condprobs = [
            {
                "parent": parent,
                "child": child,
                "P(child=1|parent=1)": df[df[parent] == 1][child].mean()
                if (df[parent] == 1).any()
                else float("nan"),
            }
            for parent, child in G.edges()
        ]
        pd.DataFrame(condprobs).to_csv(
            outdir / "learned_dag_conditional_probabilities.csv", index=False
        )
    else:
        print(f"[WARN] Could not find data file for conditional probabilities: {data_path}")

    nx.to_pandas_adjacency(G, nodelist=list(G.nodes()), weight=None).to_csv(
        outdir / "learned_dag_adjacency_matrix.csv"
    )

    try:
        from networkx.drawing.nx_pydot import write_dot
        write_dot(G, str(outdir / "learned_dag.dot"))
    except Exception:
        pass

    is_dag = nx.is_directed_acyclic_graph(G)
    with open(outdir / "learned_dag_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Nodes: {G.number_of_nodes()}\n")
        f.write(f"Edges: {G.number_of_edges()}\n")
        f.write(f"Is DAG: {is_dag}\n")
        if is_dag:
            f.write("One topological order:\n")
            f.write(" -> ".join(nx.topological_sort(G)) + "\n")


# ---------------------------------------------------------------------------
# HFACS 3-class DAG
# ---------------------------------------------------------------------------

def load_hfacs_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")
    return df


def learn_dag_ges(config: dict, df: pd.DataFrame) -> tuple[nx.DiGraph, dict]:
    from causallearn.search.ScoreBased.GES import ges

    dag_cfg = config["ghfacs_dag"]
    categories = list(config["hfacs_categories"].keys())
    sink_nodes = dag_cfg["sink_nodes"]
    score_func = dag_cfg["score_func"]
    max_indegree = dag_cfg.get("max_indegree")

    X = df[categories].astype(int).to_numpy()
    record = ges(X, score_func=score_func, maxP=max_indegree, parameters=None)
    directed, undirected = _matrix_to_edges(categories, record["G"].graph)

    sinks = set(sink_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(categories)
    G.add_edges_from(directed)

    for s in sinks:
        for other in list(G.successors(s)):
            G.remove_edge(s, other)

    G = _orient_undirected_edges_to_dag(G, undirected, sinks)

    for s in sink_nodes:
        if list(G.successors(s)):
            raise ValueError(f"Sink node {s} has outgoing edges after orientation!")

    return G, record


def evaluate_dag_predictions(
    G: nx.DiGraph, df: pd.DataFrame, tie_break: str = "error"
) -> pd.Series:
    error_parents = list(G.predecessors("Error"))
    viol_parents = list(G.predecessors("Violation"))

    preds = []
    for _, row in df.iterrows():
        pe = any(row.get(p, 0) == 1 for p in error_parents)
        pv = any(row.get(p, 0) == 1 for p in viol_parents)
        if pe and pv:
            preds.append(1 if tie_break == "error" else 2)
        elif pe:
            preds.append(1)
        elif pv:
            preds.append(2)
        else:
            preds.append(0)

    return pd.Series(preds, index=df.index, dtype=int)


def run_hfacs_dag(config: dict) -> None:
    data_path = config["paths"]["processed_csv"]
    output_dir = config["paths"]["dag_output_dir"]

    df = load_hfacs_data(data_path)
    y3 = make_three_class_target_from_config(df, config)
    df = df[(y3 > 0) & (y3 != -1)].copy()

    G, record = learn_dag_ges(config, df)
    export_dag_outputs(G, output_dir, data_path=data_path)
    plot_dag(G, save_path=str(Path(output_dir) / "learned_dag.pdf"))

    score = record.get("score")
    score_str = str(score) if score is not None else "N/A"
    print(f"[INFO] GES complete. Score={score_str}. Outputs in {output_dir}")
    (Path(output_dir) / "learned_dag_score.txt").write_text(
        f"GES Score: {score_str}\n", encoding="utf-8"
    )


def run_dag_evaluation(config: dict) -> None:
    data_path = config["paths"]["processed_csv"]
    output_dir = config["paths"]["dag_output_dir"]
    categories = list(config["hfacs_categories"].keys())

    df = load_hfacs_data(data_path)
    y3 = make_three_class_target_from_config(df, config)
    mask = y3 != -1
    df = df.loc[mask].copy()
    y3 = y3.loc[mask]

    G, _ = learn_dag_ges(config, df)

    df_eval = df[categories].assign(y3=y3)
    df_bal = balance_undersample(df_eval, "y3", seed=config["models"]["random_state"])
    y3_bal = df_bal.pop("y3")

    y_pred = evaluate_dag_predictions(
        G, df_bal[categories], tie_break=config["svm"].get("tie_break", "error")
    )

    labels = [0, 1, 2]
    label_names = ["Neither", "Error", "Violation"]
    cm = confusion_matrix(y3_bal, y_pred, labels=labels)
    acc = accuracy_score(y3_bal, y_pred)

    print("\nDAG prediction results (balanced dataset)")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=pred) {label_names}:\n{cm}")
    print("\nClassification report:\n")
    print(classification_report(y3_bal, y_pred, labels=labels, target_names=label_names, digits=4))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        cm,
        index=[f"true_{n}" for n in label_names],
        columns=[f"pred_{n}" for n in label_names],
    ).to_csv(out / "dag_confusion_matrix.csv")
    pd.DataFrame({"y_true": y3_bal.values, "y_pred": y_pred.values}).to_csv(
        out / "dag_predictions.csv", index=False
    )
    print(f"\nSaved → {out / 'dag_confusion_matrix.csv'}")


# ---------------------------------------------------------------------------
# GHFACS 4-class DAG
# ---------------------------------------------------------------------------

def _load_and_binarize(data_path: str, nodes: list[str]) -> pd.DataFrame:
    df = pd.read_excel(data_path)
    missing = [c for c in nodes if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    return df[nodes].notna().astype(int)


def learn_ghfacs_dag(config: dict, df: pd.DataFrame) -> tuple[nx.DiGraph, dict]:
    from causallearn.search.ScoreBased.GES import ges

    gdcfg = config["ghfacs_dag"]
    nodes = list(gdcfg["nodes"])
    sink_nodes = list(gdcfg["sink_nodes"])
    score_func = gdcfg["score_func"]
    max_indegree = gdcfg.get("max_indegree")

    X = df[nodes].to_numpy()
    record = ges(X, score_func=score_func, maxP=max_indegree, parameters=None)
    directed, undirected = _matrix_to_edges(nodes, record["G"].graph)

    sinks = set(sink_nodes)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(directed)

    for s in sinks:
        for other in list(G.successors(s)):
            G.remove_edge(s, other)

    G = _orient_undirected_edges_to_dag(G, undirected, sinks)

    for s in sink_nodes:
        if list(G.successors(s)):
            raise ValueError(f"Sink node {s} has outgoing edges after orientation!")

    return G, record


def run_ghfacs_dag(config: dict) -> None:
    gdcfg = config["ghfacs_dag"]
    data_path = str(Path(config["paths"]["ghfacs_data_dir"]) / config["llm"]["input"])
    output_dir = config["paths"]["ghfacs_dag_output_dir"]

    df = _load_and_binarize(data_path, list(gdcfg["nodes"]))
    df = df[df.any(axis=1)].copy()
    print(f"[INFO] Rows with at least one active GHFACS node: {len(df)}")

    G, record = learn_ghfacs_dag(config, df)
    export_dag_outputs(G, output_dir, data_path=None)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    condprobs = []
    for parent, child in G.edges():
        p_child_given_parent = (
            df[df[parent] == 1][child].mean() if (df[parent] == 1).any() else float("nan")
        )
        p_child_baseline = df[child].mean()
        condprobs.append({
            "parent": parent,
            "child": child,
            "P(child=1|parent=1)": round(p_child_given_parent, 4),
            "P(child=1)_baseline": round(p_child_baseline, 4),
            "lift": round(p_child_given_parent / p_child_baseline, 4)
            if p_child_baseline > 0 else float("nan"),
        })
    pd.DataFrame(condprobs).to_csv(
        out / "ghfacs_dag_conditional_probabilities.csv", index=False
    )

    plot_dag(G, save_path=str(out / "ghfacs_dag.pdf"))

    score = record.get("score")
    score_str = str(score) if score is not None else "N/A"
    print(f"[INFO] GES complete. Score={score_str}. Outputs in {output_dir}")
    (out / "ghfacs_dag_score.txt").write_text(f"GES Score: {score_str}\n", encoding="utf-8")
