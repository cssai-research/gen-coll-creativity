"""Experiment D: Random Walk Length in Isolation

This experiment measures how random-walk sampling alone (without inspiration/integration) changes
the observed properties of sampled subgraphs. For each graph, we sample subgraphs via random walks
of varying lengths and compute absolute metrics of the sampled subgraph.

Key idea: isolate the effect of random walk length on sampling outcomes.

Storage: results are written to a single SQLite DB per configuration under:
    outputs/parameter_validation/baseline_D/rw<steps>_iter<iters>_nodes<nodes>/results.db

Plotting: use --plot_only with one or more DBs or config directories.
"""
from __future__ import annotations

import sys
import os

# Add the root directory (hnds) to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)



import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from core.utils import UtilityFunctions
import time
import random
import json
import argparse
import sqlite3
import uuid
import hashlib
import numpy as np
import pandas as pd
from typing import Optional, List


# --- Helper functions for output dirs and progress saving/loading ---

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_progress(data: dict, output_path: str) -> None:
    """Save a JSON object to an explicit path."""
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def load_progress(output_path: str):
    """Load a JSON object from an explicit path, or return None."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f)
    return None


# --- Modern argparse interface, helpers, and deterministic seeding ---
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Experiment D or plot results from SQLite.")

    # Run config
    parser.add_argument("--random_walk_steps", type=int, default=20, help="Random walk length.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per starting node.")
    parser.add_argument("--starting_nodes", type=int, default=5, help="Number of starting nodes to sample per graph.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")

    # Optional chunking over graphs
    parser.add_argument("--num_graphs", type=int, default=500, help="How many graphs to process (max 500).")
    parser.add_argument("--graph_offset", type=int, default=0, help="Offset into the 500-graph list (for chunking).")

    # Debug option (kept minimal; default off)
    parser.add_argument("--save_node_files", type=int, default=0, help="If 1, also save per-graph/per-node JSON files for debugging. Default 0.")

    # Plotting
    parser.add_argument("--plot_only", type=int, default=0, help="If 1, do not run; only generate plots from existing SQLite DB(s).")
    parser.add_argument("--plot_dbs", nargs="+", default=None, help="List of SQLite DB paths OR config result directories containing results.db.")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="diameter",
        choices=["diameter", "num_concepts", "unique_concepts", "overlap", "all"],
        help="Which metric to plot. Use 'all' to create a multi-panel figure."
    )
    parser.add_argument(
        "--plot_style",
        type=str,
        default="box",
        choices=["box", "points_iqr"],
        help="Plot style: boxplot or discrete points with IQR bars."
    )
    parser.add_argument("--plot_out", type=str, default=None, help="Optional output filename (saved as PDF).")

    return parser.parse_args()


def _stable_int_seed(*parts, mod: int = 2**31 - 1) -> int:
    """Create a deterministic integer seed from arbitrary parts (stable across processes)."""
    h = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % mod


def _resolve_db_path(path_or_dir: str) -> str:
    if path_or_dir.endswith(".db"):
        return path_or_dir
    return os.path.join(path_or_dir, "results.db")


def _metric_to_column(metric: str) -> str:
    mapping = {
        "diameter": "diameter_values",
        "num_concepts": "num_concept_values",
        "unique_concepts": "unique_concept_values",
        "overlap": "overlap_values",
    }
    return mapping[metric]


def _all_metrics() -> List[str]:
    return ["diameter", "num_concepts", "unique_concepts", "overlap"]


def _metric_display_name(metric: str) -> str:
    mapping = {
        "diameter": "diameter",
        "num_concepts": "# concepts",
        "unique_concepts": "# unique concepts",
        "overlap": "overlap",
    }
    return mapping.get(metric, metric)


# --- Deterministic random walk sampling for Experiment D ---
def process_graph_with_steps(args):
    graph, graph_idx, random_walk_steps, iterations, starting_nodes_count, global_seed = args

    # Deterministic starting-node selection
    nodes_sorted = sorted(list(graph.nodes))
    if len(nodes_sorted) <= starting_nodes_count:
        sampled_starting_nodes = nodes_sorted
    else:
        rng_nodes = random.Random(_stable_int_seed(global_seed, graph_idx, "starting_nodes"))
        sampled_starting_nodes = rng_nodes.sample(nodes_sorted, k=starting_nodes_count)

    graph_results = {}
    for starting_node in sampled_starting_nodes:
        graph_results[starting_node] = {
            "diameter_value_list": [],
            "num_concept_count_list": [],
            "unique_concept_count_list": [],
            "overlap_score_list": [],
        }

        for iter_idx in range(iterations):
            seed_walk = _stable_int_seed(global_seed, graph_idx, starting_node, iter_idx, random_walk_steps, "walk")
            UtilityFunctions.set_global_seed(seed_walk)

            path = UtilityFunctions.random_walk_path(graph, random_walk_steps, starting_node)
            subgraph = UtilityFunctions.subgraph_from_path(graph, path)
            if subgraph is None:
                continue

            diameter = UtilityFunctions.compute_diameter(subgraph)
            num_concepts = UtilityFunctions.compute_num_concepts_accessed(subgraph.nodes)
            unique_concepts = len(set(subgraph.nodes))
            overlap_score = UtilityFunctions.compute_overlap(graph, subgraph)

            graph_results[starting_node]["diameter_value_list"].append(diameter)
            graph_results[starting_node]["num_concept_count_list"].append(num_concepts)
            graph_results[starting_node]["unique_concept_count_list"].append(unique_concepts)
            graph_results[starting_node]["overlap_score_list"].append(overlap_score)

    return graph_idx, graph_results


# --- SQLite plotting helpers for Experiment D ---
def load_metric_values_from_db(db_path: str, metric: str):
    metrics = _all_metrics() if metric == "all" else [metric]
    cols = [_metric_to_column(m) for m in metrics]

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT random_walk_steps FROM runs")
    rw_rows = cur.fetchall()
    rw_length = rw_rows[0][0] if rw_rows else None

    cur.execute(f"SELECT {', '.join(cols)} FROM graph_results")
    rows = cur.fetchall()
    conn.close()

    out = {m: [] for m in metrics}
    for row in rows:
        for m, json_list in zip(metrics, row):
            if json_list is None:
                continue
            try:
                out[m].extend(json.loads(json_list))
            except Exception:
                continue

    if metric == "all":
        return rw_length, out
    return rw_length, out[metric]


def plot_experimentD_from_sqlite(db_paths, metric: str, style: str, out_path: Optional[str] = None):
    metrics_to_plot = _all_metrics() if metric == "all" else [metric]

    # Build compact config labels
    config_labels = []
    for p in db_paths:
        base = os.path.basename(os.path.dirname(_resolve_db_path(p)))
        config_labels.append(base)

    # Guardrails: enforce comparable configurations
    common_iter = None
    common_nodes = None
    common_seed = None
    try:
        iters = []
        nodes = []
        seeds = []
        for p in db_paths:
            conn_tmp = sqlite3.connect(_resolve_db_path(p))
            cur_tmp = conn_tmp.cursor()
            cur_tmp.execute("SELECT DISTINCT iterations, starting_nodes, seed FROM runs")
            rows = cur_tmp.fetchall()
            conn_tmp.close()
            if rows:
                iters.append(rows[0][0])
                nodes.append(rows[0][1])
                seeds.append(rows[0][2])
        if len(set(iters)) == 1:
            common_iter = iters[0]
        if len(set(nodes)) == 1:
            common_nodes = nodes[0]
        if len(set(seeds)) == 1:
            common_seed = seeds[0]
    except Exception:
        pass

    if len(db_paths) > 1:
        if common_iter is None:
            raise ValueError("Incompatible DBs: --iterations differs across configs.")
        if common_nodes is None:
            raise ValueError("Incompatible DBs: --starting_nodes differs across configs.")
        if common_seed is None:
            raise ValueError("Incompatible DBs: --seed differs across configs.")

    # Collect data per metric
    metric_records = {m: [] for m in metrics_to_plot}
    for p in db_paths:
        db_path = _resolve_db_path(p)
        rw, vals = load_metric_values_from_db(db_path, metric)
        if rw is None:
            continue

        if metric == "all":
            for m in metrics_to_plot:
                for v in vals.get(m, []):
                    metric_records[m].append({"rw": rw, "value": v})
        else:
            for v in vals:
                metric_records[metric].append({"rw": rw, "value": v})

    if any(len(metric_records[m]) == 0 for m in metrics_to_plot):
        raise ValueError("No records found to plot for one or more metrics. Check DB paths.")

    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics_to_plot):
        df = pd.DataFrame(metric_records[m]).sort_values("rw")
        disp = _metric_display_name(m)

        if style == "box":
            sns.boxplot(data=df, x="rw", y="value", ax=ax)
            ax.set_ylabel(disp)

        elif style == "points_iqr":
            grouped = df.groupby("rw")["value"]
            summary = grouped.agg(
                median="median",
                q25=lambda x: np.percentile(x, 25),
                q75=lambda x: np.percentile(x, 75),
            ).reset_index()

            x = summary["rw"].values
            y = summary["median"].values
            yerr_lower = y - summary["q25"].values
            yerr_upper = summary["q75"].values - y

            ax.errorbar(x, y, yerr=[yerr_lower, yerr_upper], fmt="o", capsize=4)
            ax.set_ylabel(disp)

        ax.set_xlabel("Random walk length")
        ax.set_title(disp)

    fig.suptitle("Experiment D: sampling outcomes vs random walk length")

    annotation_lines = []
    annotation_lines.append("Compared configs: " + ", ".join(config_labels))
    if common_iter is not None:
        annotation_lines.append(f"iterations = {common_iter}")
    if common_nodes is not None:
        annotation_lines.append(f"starting_nodes = {common_nodes}")
    if common_seed is not None:
        annotation_lines.append(f"seed = {common_seed}")

    fig.text(0.5, 0.01, " | ".join(annotation_lines), ha="center", va="bottom", fontsize=9)

    # Always save into a plots/ directory
    if len(db_paths) == 1:
        base_dir = os.path.dirname(_resolve_db_path(db_paths[0]))
    else:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base_dir = os.path.join(REPO_ROOT, "outputs", "parameter_validation", "baseline_D")

    plots_dir = os.path.join(base_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if out_path is None:
        # Compact, safe filename using rw labels only
        rw_labels = []
        for lbl in config_labels:
            if lbl.startswith("rw"):
                rw_labels.append(lbl.split("_")[0])
        rw_tag = "_".join(rw_labels)
        base_name = f"all_metrics_{style}__{rw_tag}.pdf" if metric == "all" else f"{metric}_{style}__{rw_tag}.pdf"
        out_path = os.path.join(plots_dir, base_name)
    else:
        out_path = os.path.join(plots_dir, os.path.basename(out_path))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    print(f"Saved plot to {final_path}")

    return final_path



# --- Central runner-friendly Experiment D function ---

def run_experimentD(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    random_walk_steps: int = 20,
    iterations: int = 5,
    starting_nodes: int = 5,
    num_graphs: int = 500,
    graph_offset: int = 0,
    save_node_files: bool = False,
    output_dir: Optional[str] = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
    plot_metric: str = "all",
    plot_style: str = "box",
) -> tuple[dict, list]:
    """Run Experiment D for one random-walk-length configuration.

    Returns:
        (results_dict, plot_paths)

    Notes:
      - Results are stored in SQLite under: <output_dir>/results/rw<steps>_iter<iters>_nodes<nodes>/results.db
      - If resume=True and results.db exists, computation is skipped and only plotting is performed (if make_plot=True).
    """
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()

    if output_dir is None:
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        output_dir = os.path.join(REPO_ROOT, "outputs", "parameter_validation", "baseline_D")

    # Load graphs for this configuration
    file_pattern = os.path.join(root_dir, "data", "saved_cog_nets", f"G_n_{num_nodes}_k_{average_degree}_*.edgelist.gz")
    graphs = UtilityFunctions.load_graphs(file_pattern)
    if len(graphs) == 0:
        raise FileNotFoundError(f"No graphs found matching the pattern: {file_pattern}")

    # Optional chunking
    graph_offset = max(0, int(graph_offset))
    num_graphs = int(num_graphs)
    end_idx = min(len(graphs), graph_offset + num_graphs)
    graphs_slice = list(enumerate(graphs))[graph_offset:end_idx]

    # Output locations
    config_folder_name = f"rw{random_walk_steps}_iter{iterations}_nodes{starting_nodes}"
    config_results_dir = os.path.join(output_dir, config_folder_name)
    _ensure_dir(config_results_dir)

    db_path = os.path.join(config_results_dir, "results.db")
    metadata_filename = os.path.join(config_results_dir, "metadata.json")

    did_compute = False
    time_taken = None

    if not (resume and os.path.exists(db_path)):
        did_compute = True
        start_time = time.time()

        num_processes = cpu_count()
        run_id = str(uuid.uuid4())

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Improve write performance
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        try:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    random_walk_steps INTEGER,
                    iterations INTEGER,
                    starting_nodes INTEGER,
                    seed INTEGER,
                    graph_offset INTEGER,
                    num_graphs INTEGER,
                    timestamp TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS graph_results (
                    run_id TEXT,
                    graph_idx INTEGER,
                    starting_node INTEGER,
                    diameter_values TEXT,
                    num_concept_values TEXT,
                    unique_concept_values TEXT,
                    overlap_values TEXT
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_lookup ON graph_results (graph_idx, starting_node);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_graph_run ON graph_results (run_id);")

            cur.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (run_id, random_walk_steps, iterations, starting_nodes, seed, graph_offset, len(graphs_slice)),
            )
            conn.commit()

            with Pool(processes=num_processes) as pool:
                args_list = [
                    (graph, idx, random_walk_steps, iterations, starting_nodes, seed)
                    for idx, graph in graphs_slice
                ]

                for count, (graph_idx, graph_result) in enumerate(
                    tqdm(
                        pool.imap(process_graph_with_steps, args_list),
                        total=len(args_list),
                        desc=f"Processing Graphs (RW={random_walk_steps})",
                    )
                ):
                    for starting_node, metrics in graph_result.items():
                        cur.execute(
                            "INSERT INTO graph_results VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                graph_idx,
                                starting_node,
                                json.dumps(metrics["diameter_value_list"]),
                                json.dumps(metrics["num_concept_count_list"]),
                                json.dumps(metrics["unique_concept_count_list"]),
                                json.dumps(metrics["overlap_score_list"]),
                            ),
                        )

                        if save_node_files:
                            graph_folder = os.path.join(config_results_dir, f"graph_{graph_idx}")
                            _ensure_dir(graph_folder)
                            node_result = {
                                "graph_index": graph_idx,
                                "starting_node": starting_node,
                                "metrics": metrics,
                            }
                            node_filename = os.path.join(graph_folder, f"node_{starting_node}.json")
                            save_progress(node_result, node_filename)

                    if count % 50 == 0:
                        conn.commit()

            conn.commit()

            end_time = time.time()
            time_taken = end_time - start_time

        finally:
            try:
                conn.commit()
            except Exception:
                pass
            conn.close()

        metadata = {
            "experiment": "experimentD",
            "N": num_nodes,
            "K": average_degree,
            "random_walk_steps": random_walk_steps,
            "iterations": iterations,
            "starting_nodes": starting_nodes,
            "seed": seed,
            "machine_id": machine_id,
            "time_taken": time_taken,
            "graph_offset": graph_offset,
            "num_graphs": len(graphs_slice),
            "db_path": db_path,
        }
        if save_json:
            save_progress(metadata, metadata_filename)

    else:
        meta = load_progress(metadata_filename)
        if isinstance(meta, dict) and "time_taken" in meta:
            time_taken = meta.get("time_taken")

    plot_paths: list = []
    if make_plot:
        out = plot_experimentD_from_sqlite(
            [config_results_dir],
            metric=plot_metric,
            style=plot_style,
            out_path=None,
        )
        plot_paths.append(out)

    results = {
        "experiment": "experimentD",
        "config": {
            "N": num_nodes,
            "K": average_degree,
            "seed": seed,
            "random_walk_steps": random_walk_steps,
            "iterations": iterations,
            "starting_nodes": starting_nodes,
            "num_graphs": num_graphs,
            "graph_offset": graph_offset,
        },
        "output_dir": output_dir,
        "config_results_dir": config_results_dir,
        "db_path": db_path,
        "metadata_path": metadata_filename,
        "time_taken": time_taken,
        "did_compute": did_compute,
        "plot_metric": plot_metric,
        "plot_style": plot_style,
    }

    return results, plot_paths

# --- Main block: run or plot ---

if __name__ == '__main__':
    args = parse_arguments()

    # Plot-only mode
    if bool(args.plot_only):
        if not args.plot_dbs:
            raise ValueError("--plot_only 1 requires --plot_dbs with one or more DB paths or config directories")
        out = plot_experimentD_from_sqlite(args.plot_dbs, args.plot_metric, args.plot_style, args.plot_out)
        print(f"Plot: {out}")
        raise SystemExit(0)

    results, plot_paths = run_experimentD(
        num_nodes=100,  # NOTE: CLI keeps historical defaults; central runner overrides these.
        average_degree=4,
        seed=args.seed,
        random_walk_steps=args.random_walk_steps,
        iterations=args.iterations,
        starting_nodes=args.starting_nodes,
        num_graphs=args.num_graphs,
        graph_offset=args.graph_offset,
        save_node_files=bool(args.save_node_files),
        output_dir=os.path.join(root_dir, "outputs", "parameter_validation", "baseline_D"),
        resume=True,
        save_json=True,
        make_plot=True,
        plot_metric=args.plot_metric,
        plot_style=args.plot_style,
    )

    print("Done.")
    print(f"DB: {results.get('db_path')}")
    if plot_paths:
        for p in plot_paths:
            print(f"Plot: {p}")
