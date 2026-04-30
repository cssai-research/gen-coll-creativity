"""Experiment C: Sensitivity of Inspiration Outcomes to Random Walk Length.

This script evaluates how inspiration-induced changes in sampled subgraph properties vary
with the random walk length used to sample subgraphs.

IMPORTANT (experimental validity): to compare different random walk lengths fairly, the
sampled (source,target) graph pairs should be held constant across runs. This file
therefore supports generating a persistent pair-set file (JSON) that is reused across
multiple runs/configurations.

NOTE (storage): By default, results are written to a single SQLite database per config
folder (results.db) to avoid file explosion and to support incremental runs.

Debugging: If you need to inspect individual pair/start-node outputs as JSON, run with
--save_pair_files 1 (this will create many files; keep --num_pairs small).
"""
from __future__ import annotations

import sys
import os

# Add the root directory (hnds) to the Python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_dir not in sys.path:
    sys.path.append(root_dir)


import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from core.utils import UtilityFunctions

import time
import random
import numpy as np

import argparse
import json
import seaborn as sns
import pandas as pd
import sqlite3
import uuid
import hashlib
from typing import Optional, List

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experiments with configurable parameters.")
    parser.add_argument("--random_walk_steps", type=int, default=20, help="Random walk length.")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per starting node.")
    parser.add_argument("--starting_nodes", type=int, default=5, help="Number of starting nodes to sample.")
    # parser.add_argument("--normalize_by", type=str, default=None, choices=["maximum_value", "random_walk_length", None], help="Normalization method.")
    parser.add_argument("--num_pairs", type=int, default=5000, help="Number of sampled ordered graph pairs to process from the persistent pair set. Use a small value for quick debug runs.")
    parser.add_argument("--pair_offset", type=int, default=0, help="Offset into the persistent pair set (0-indexed). Enables running in small increments.")
    parser.add_argument("--pair_set_seed", type=int, default=42, help="Seed used to generate (or identify) the persistent pair set.")
    parser.add_argument("--pair_set_dir", type=str, default=None, help="Directory to store/reuse persistent pair-set JSON files. Defaults to experiments/experimentC/pair_sets.")
    parser.add_argument("--save_pair_files", type=int, default=0, help="If 1, also save per-pair/per-node JSON files for debugging. Default is 0 (disabled).")

    # --- New CLI arguments for plotting from SQLite ---
    parser.add_argument("--plot_only", type=int, default=0, help="If 1, do not run the experiment; only generate plots from existing SQLite DB(s).")
    parser.add_argument("--plot_dbs", nargs="+", default=None, help="List of SQLite DB paths OR config result directories containing results.db.")
    parser.add_argument(
        "--plot_metric",
        type=str,
        default="diameter",
        choices=["diameter", "num_concepts", "unique_concepts", "overlap", "all"],
        help="Which metric to plot. Use 'all' to create a 1x4 multi-panel figure."
    )
    parser.add_argument(
        "--plot_style",
        type=str,
        default="box",
        choices=["box", "median_iqr", "points_iqr"],
        help="Plot style: boxplot, median+IQR band, or discrete points with IQR bars."
    )
    parser.add_argument("--plot_out", type=str, default=None, help="Output path (saved as PDF). Defaults to <config_dir>/<metric>_<style>.pdf if one DB is provided, else ./expC_<metric>_<style>.pdf")

    return parser.parse_args()


def _stable_int_seed(*parts, mod: int = 2**31 - 1) -> int:
    """Create a deterministic integer seed from arbitrary parts.

    Avoids Python's built-in hash randomization (which differs per process).
    """
    h = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % mod

# Wrapper function to unpack arguments for process_pair
def process_pair_with_args(args):
    """Multiprocessing helper: unpacks args and calls process_pair."""
    idx1, idx2, pair, random_walk_steps, iterations, starting_nodes_count, global_seed = args
    return process_pair(idx1, idx2, pair, random_walk_steps, iterations, starting_nodes_count, global_seed)

# Process a pair of graphs
def process_pair(idx1, idx2, pair, random_walk_steps, iterations, starting_nodes_count, global_seed: int):
    """Run inspiration integration for one (source_graph, target_graph) pair.

    For each valid starting node (present in both graphs), this samples a source subgraph
    and a target subgraph (BEFORE), integrates the source subgraph edges into a copy of the
    target graph, then samples a target subgraph again (AFTER). Metric deltas are computed
    across iterations.
    """
    source_graph, target_graph = pair
    pair_results = {}

    # Deterministic starting-node selection for fair cross-configuration comparisons
    shared_nodes = sorted(set(source_graph.nodes).intersection(set(target_graph.nodes)))
    if len(shared_nodes) <= starting_nodes_count:
        valid_starting_nodes = shared_nodes
    else:
        rng_nodes = random.Random(_stable_int_seed(global_seed, idx1, idx2, "starting_nodes"))
        valid_starting_nodes = rng_nodes.sample(shared_nodes, k=starting_nodes_count)

    for starting_node in valid_starting_nodes:
        pair_results[starting_node] = {
            "diameter_value_list": [],
            "num_concept_count_list": [],
            "unique_concept_count_list": [],
            "overlap_score_list": [],
        }

        for iter_idx in range(iterations):
            # Deterministic seeds for reproducibility across multiprocessing and repeated runs
            seed_source = _stable_int_seed(global_seed, idx1, idx2, starting_node, iter_idx, random_walk_steps, "source")
            seed_target = _stable_int_seed(global_seed, idx1, idx2, starting_node, iter_idx, random_walk_steps, "target")

            # BEFORE
            UtilityFunctions.set_global_seed(seed_source)
            source_path = UtilityFunctions.random_walk_path(source_graph, random_walk_steps, starting_node)
            source_subgraph = UtilityFunctions.subgraph_from_path(source_graph, source_path)

            # Use the same seed for target BEFORE and AFTER walks to reduce Monte Carlo noise
            UtilityFunctions.set_global_seed(seed_target)
            target_path = UtilityFunctions.random_walk_path(target_graph, random_walk_steps, starting_node)
            target_subgraph_before = UtilityFunctions.subgraph_from_path(target_graph, target_path)

            if source_subgraph is None or target_subgraph_before is None:
                continue

            diameter_before = UtilityFunctions.compute_diameter(target_subgraph_before)
            num_concepts_before = UtilityFunctions.compute_num_concepts_accessed(target_subgraph_before.nodes)
            concepts_before = set(target_subgraph_before.nodes)
            overlap_score = UtilityFunctions.compute_overlap(source_subgraph, target_subgraph_before)

            # AFTER
            target_graph_copy = target_graph.copy()
            for edge in source_subgraph.edges:
                target_graph_copy.add_edge(*edge)

            UtilityFunctions.set_global_seed(seed_target)
            target_path_after = UtilityFunctions.random_walk_path(target_graph_copy, random_walk_steps, starting_node)
            target_subgraph_after = UtilityFunctions.subgraph_from_path(target_graph_copy, target_path_after)

            if target_subgraph_after is None:
                continue

            diameter_after = UtilityFunctions.compute_diameter(target_subgraph_after)
            num_concepts_after = UtilityFunctions.compute_num_concepts_accessed(target_subgraph_after.nodes)
            concepts_after = set(target_subgraph_after.nodes)

            # Calculate deltas
            diameter_delta = abs(diameter_after - diameter_before)
            num_concepts_delta = abs(num_concepts_after - num_concepts_before)
            # Newly gained concepts after inspiration
            unique_concepts_delta = len(concepts_after - concepts_before)

            # Append results
            pair_results[starting_node]["diameter_value_list"].append(diameter_delta)
            pair_results[starting_node]["num_concept_count_list"].append(num_concepts_delta)
            pair_results[starting_node]["unique_concept_count_list"].append(unique_concepts_delta)
            pair_results[starting_node]["overlap_score_list"].append(overlap_score)

    return pair_results

# DEPRECATED: JSON-based plotting (kept for historical reference). Prefer SQLite plotting via --plot_only.
def plot_results_from_json(json_filename):
    """DEPRECATED.

    JSON-based plotting is no longer maintained. Use SQLite plotting via:
      --plot_only 1 --plot_dbs <...>

    This stub is kept to avoid breaking older notebooks/scripts that may import it.
    """
    raise RuntimeError(
        "plot_results_from_json is deprecated. Use plot_experimentC_from_sqlite (SQLite plotting) instead."
    )

# DEPRECATED: JSON-based plotting (kept for historical reference). Prefer SQLite plotting via --plot_only.
def plot_multiple_json_results(json_filenames):
    """DEPRECATED.

    JSON-based plotting is no longer maintained. Use SQLite plotting via:
      --plot_only 1 --plot_dbs <...>

    This stub is kept to avoid breaking older notebooks/scripts that may import it.
    """
    raise RuntimeError(
        "plot_multiple_json_results is deprecated. Use plot_experimentC_from_sqlite (SQLite plotting) instead."
    )

# --- New plotting helpers for SQLite-based Experiment C results ---
def _resolve_db_path(path_or_dir: str) -> str:
    """Accept either a direct .db path or a config directory containing results.db."""
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

# Helper: list of all metrics
def _all_metrics() -> List[str]:
    return ["diameter", "num_concepts", "unique_concepts", "overlap"]

# Human-readable display names for metrics (for plots)
def _metric_display_name(metric: str) -> str:
    """Human-readable labels for plots.

    Note: diameter and num_concepts are stored as inspiration-induced deltas.
    unique_concepts is stored as newly gained concepts (after - before).
    overlap is the overlap score between source and target BEFORE inspiration.
    """
    mapping = {
        "diameter": "Δ diameter",
        "num_concepts": "Δ # concepts",
        "unique_concepts": "# new concepts",
        "overlap": "overlap",
    }
    return mapping.get(metric, metric)


def load_metric_values_from_db(db_path: str, metric: str):
    """Load and flatten all metric values from a single Experiment C SQLite DB.
    Supports metric == "all" to return a dict of all metrics."""
    metrics = _all_metrics() if metric == "all" else [metric]
    cols = [_metric_to_column(m) for m in metrics]
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # RW length is stored in runs; one DB is expected to be one config.
    cur.execute("SELECT DISTINCT random_walk_steps FROM runs")
    rw_rows = cur.fetchall()
    rw_length = rw_rows[0][0] if rw_rows else None

    cur.execute(f"SELECT {', '.join(cols)} FROM pair_results")
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


# All plots are saved under <experimentC>/results/<config>/plots/
def plot_experimentC_from_sqlite(
    db_paths,
    metric: str,
    style: str,
    out_path: Optional[str] = None
):
    """Plot Experiment C results across one or more SQLite DBs. Supports plotting all metrics in a multi-panel figure."""
    metrics_to_plot = _all_metrics() if metric == "all" else [metric]

    # Build compact config labels from DB paths (avoid overly long filenames)
    config_labels = []
    for p in db_paths:
        base = os.path.basename(os.path.dirname(_resolve_db_path(p)))
        config_labels.append(base)

    # Try to detect common parameters for annotation clarity
    common_iter = None
    common_nodes = None
    try:
        iters = []
        nodes = []
        for p in db_paths:
            conn_tmp = sqlite3.connect(_resolve_db_path(p))
            cur_tmp = conn_tmp.cursor()
            cur_tmp.execute("SELECT DISTINCT iterations, starting_nodes FROM runs")
            rows = cur_tmp.fetchall()
            conn_tmp.close()
            if rows:
                iters.append(rows[0][0])
                nodes.append(rows[0][1])
        if len(set(iters)) == 1:
            common_iter = iters[0]
        if len(set(nodes)) == 1:
            common_nodes = nodes[0]
    except Exception:
        pass

    # Guardrails: warn or stop on incompatible configurations
    if len(db_paths) > 1:
        if common_iter is None:
            raise ValueError(
                "Incompatible DBs: --iterations differs across configs. "
                "Do not compare results generated with different iteration counts."
            )
        if common_nodes is None:
            raise ValueError(
                "Incompatible DBs: --starting_nodes differs across configs. "
                "Do not compare results generated with different starting-node counts."
            )

        try:
            pair_seeds = []
            offsets = []
            for p in db_paths:
                conn_tmp = sqlite3.connect(_resolve_db_path(p))
                cur_tmp = conn_tmp.cursor()
                cur_tmp.execute("SELECT DISTINCT pair_offset, pair_set_seed FROM runs")
                rows = cur_tmp.fetchall()
                conn_tmp.close()
                if rows:
                    offsets.append(rows[0][0])
                    pair_seeds.append(rows[0][1])
            if len(set(pair_seeds)) != 1:
                raise ValueError(
                    "Incompatible DBs: pair_set_seed differs across configs. "
                    "RW-length sensitivity requires the same underlying pair set."
                )
            # Offsets may differ (incremental runs), but warn if they do
            if len(set(offsets)) != 1:
                print(
                    "WARNING: DBs were generated with different pair_offset values. "
                    "This is OK if offsets are complementary, but results may not be directly paired."
                )
        except sqlite3.Error:
            pass

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

    # Validate
    if any(len(metric_records[m]) == 0 for m in metrics_to_plot):
        raise ValueError("No records found to plot for one or more metrics. Check --plot_dbs paths and that results.db contains data.")

    # Prepare figure
    n = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, m in zip(axes, metrics_to_plot):
        df = pd.DataFrame(metric_records[m]).sort_values("rw")
        disp = _metric_display_name(m)

        if style == "box":
            sns.boxplot(data=df, x="rw", y="value", ax=ax)
            ax.set_ylabel(disp)

        elif style == "median_iqr":
            grouped = df.groupby("rw")["value"]
            summary = grouped.agg(
                median="median",
                q25=lambda x: np.percentile(x, 25),
                q75=lambda x: np.percentile(x, 75),
            ).reset_index()
            ax.fill_between(summary["rw"], summary["q25"], summary["q75"], alpha=0.3)
            ax.plot(summary["rw"], summary["median"], marker="o")
            ax.set_xlabel("Random walk length")
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

            ax.errorbar(
                x,
                y,
                yerr=[yerr_lower, yerr_upper],
                fmt="o",
                capsize=4,
            )
            ax.set_xlabel("Random walk length")
            ax.set_ylabel(disp)

        ax.set_title(disp)

    fig.suptitle("Experiment C: metrics vs random walk length")

    # Annotate compared configurations for traceability
    annotation_lines = []
    annotation_lines.append("Compared configs: " + ", ".join(config_labels))
    annotation_lines.append(f"iterations = {common_iter}")
    annotation_lines.append(f"starting_nodes = {common_nodes}")

    # Offset warning (non-fatal)
    try:
        if len(set(offsets)) != 1:
            annotation_lines.append("WARNING: different pair_offset values")
    except Exception:
        pass

    fig.text(
        0.5,
        0.01,
        " | ".join(annotation_lines),
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Always save plots inside a `plots/` directory under the experiment results
    if len(db_paths) == 1:
        base = os.path.dirname(_resolve_db_path(db_paths[0]))
    else:
        # If multiple DBs are plotted together, default to the experimentC results directory
        REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        base = os.path.join(REPO_ROOT, "outputs", "parameter_validation", "inspiration_C")

    plots_dir = os.path.join(base, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if out_path is None:
        if len(db_paths) == 1:
            base_name = f"{metric}_{style}.pdf" if metric != "all" else f"all_metrics_{style}.pdf"
        else:
            # Compact, safe filename using RW sweep labels only
            rw_labels = []
            for lbl in config_labels:
                if lbl.startswith("rw"):
                    rw_labels.append(lbl.split("_")[0])
            rw_tag = "_".join(rw_labels)
            base_name = (
                f"{metric}_{style}__{rw_tag}.pdf"
                if metric != "all"
                else f"all_metrics_{style}__{rw_tag}.pdf"
            )
        out_path = os.path.join(plots_dir, base_name)
    else:
        out_path = os.path.join(plots_dir, os.path.basename(out_path))

    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    print(f"Saved plot to {final_path}")
    return final_path

if __name__ == '__main__':
    args = parse_arguments()

    # --- Early exit: plotting mode from SQLite ---
    if bool(args.plot_only):
        if not args.plot_dbs:
            raise ValueError("--plot_only 1 requires --plot_dbs with one or more DB paths or config directories")
        plot_experimentC_from_sqlite(args.plot_dbs, args.plot_metric, args.plot_style, args.plot_out)
        raise SystemExit(0)

    random_walk_steps = args.random_walk_steps
    iterations = args.iterations
    starting_nodes_count = args.starting_nodes
    num_pairs = args.num_pairs
    pair_offset = args.pair_offset
    pair_set_seed = args.pair_set_seed
    pair_set_dir = args.pair_set_dir

    save_pair_files = bool(args.save_pair_files)
    run_id = str(uuid.uuid4())

    seed = 42
    UtilityFunctions.set_global_seed(seed)

    machine_id = UtilityFunctions.get_machine_id()

    # Load graphs (hardcoded)
    N = 100  # Number of nodes
    k = 4  # Average degree
    file_pattern = os.path.join("data/saved_cog_nets/", f"G_n_{N}_k_{k}_*.edgelist.gz")  # manually set to N=100, K=4
    rewired_graphs = UtilityFunctions.load_graphs(file_pattern)

    if len(rewired_graphs) != 500: # only for this N and k
        raise ValueError(f"Expected 500 graphs, but found {len(rewired_graphs)}.")

    # --- Persistent pair set (indices) for fair cross-RW-length comparisons ---
    base_dir = os.path.dirname(__file__)
    if pair_set_dir is None:
        pair_set_dir = os.path.join(base_dir, "pair_sets")
    os.makedirs(pair_set_dir, exist_ok=True)

    pair_set_filename = os.path.join(
        pair_set_dir,
        f"pairs_nGraphs{len(rewired_graphs)}_N{N}_k{k}_seed{pair_set_seed}_n5000.json",
    )

    if os.path.exists(pair_set_filename):
        with open(pair_set_filename, "r") as f:
            pair_set_indices = [tuple(x) for x in json.load(f)]
    else:
        # Build all unordered index pairs, then include both directions to form ordered pairs.
        graph_indices = list(range(len(rewired_graphs)))
        unordered = list(combinations(graph_indices, 2))
        ordered = unordered + [(j, i) for (i, j) in unordered]
        random.seed(pair_set_seed)
        pair_set_indices = random.sample(ordered, 5000)
        with open(pair_set_filename, "w") as f:
            json.dump(pair_set_indices, f, indent=4)

    # Select a slice to enable incremental runs
    if num_pairs is None or num_pairs < 0:
        selected_pair_indices = pair_set_indices[pair_offset:]
    else:
        selected_pair_indices = pair_set_indices[pair_offset: pair_offset + num_pairs]

    sampled_indices = list(selected_pair_indices)
    sampled_pairs = [(rewired_graphs[i], rewired_graphs[j]) for (i, j) in sampled_indices]

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(REPO_ROOT, "outputs", "parameter_validation", "inspiration_C")
    config_folder_name = f"rw{random_walk_steps}_iter{iterations}_nodes{starting_nodes_count}"
    config_results_dir = os.path.join(results_dir, config_folder_name)
    os.makedirs(config_results_dir, exist_ok=True)
    sampled_indices_filename = os.path.join(config_results_dir, "selected_pair_indices.json")
    with open(sampled_indices_filename, "w") as f:
        json.dump(sampled_indices, f, indent=4)

    db_path = os.path.join(config_results_dir, "results.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Improve write performance for incremental appends
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    try:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                random_walk_steps INTEGER,
                iterations INTEGER,
                starting_nodes INTEGER,
                pair_offset INTEGER,
                num_pairs INTEGER,
                seed INTEGER,
                timestamp TEXT
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS pair_results (
                run_id TEXT,
                idx1 INTEGER,
                idx2 INTEGER,
                starting_node INTEGER,
                diameter_values TEXT,
                num_concept_values TEXT,
                unique_concept_values TEXT,
                overlap_values TEXT
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_results_lookup ON pair_results (idx1, idx2, starting_node);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_results_run ON pair_results (run_id);")

        conn.commit()

        cur.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (run_id, random_walk_steps, iterations, starting_nodes_count, pair_offset, len(sampled_indices), seed)
        )
        conn.commit()

        start_time = time.time()

        num_processes = cpu_count()

        with Pool(processes=num_processes) as pool:
            args_list = [
                (idx1, idx2, sampled_pairs[i], random_walk_steps, iterations, starting_nodes_count, seed)
                for i, (idx1, idx2) in enumerate(sampled_indices)
            ]
            for pair_idx, pair_result in enumerate(tqdm(pool.imap(process_pair_with_args, args_list), total=len(sampled_pairs), desc=f"Processing Pairs (Steps={random_walk_steps})")):
                idx1, idx2 = sampled_indices[pair_idx]

                for starting_node, metrics in pair_result.items():
                    cur.execute(
                        "INSERT INTO pair_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            run_id,
                            idx1,
                            idx2,
                            starting_node,
                            json.dumps(metrics["diameter_value_list"]),
                            json.dumps(metrics["num_concept_count_list"]),
                            json.dumps(metrics["unique_concept_count_list"]),
                            json.dumps(metrics["overlap_score_list"]),
                        ),
                    )

                if save_pair_files:
                    pair_folder = os.path.join(config_results_dir, f"pair_{idx1}_{idx2}")
                    os.makedirs(pair_folder, exist_ok=True)
                    # Save each starting node's metrics as a separate JSON file
                    for starting_node, metrics in pair_result.items():
                        node_result = {
                            "pair_indices": (idx1, idx2),
                            "starting_node": starting_node,
                            "metrics": metrics
                        }
                        node_filename = os.path.join(pair_folder, f"node_{starting_node}.json")
                        with open(node_filename, "w") as f:
                            json.dump(node_result, f, indent=4)

                if pair_idx % 50 == 0:
                    conn.commit()

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing took: {elapsed_time:.2f} seconds")

        # Save metadata to JSON
        metadata_filename = os.path.join(config_results_dir, "metadata.json")
        metadata = {
            "random_walk_steps": random_walk_steps,
            "iterations": iterations,
            "starting_nodes": starting_nodes_count,
            "seed": seed,
            "machine_id": machine_id,
            "time_taken_seconds": elapsed_time,
            "num_pairs": len(sampled_indices),
            "pair_offset": pair_offset,
            "pair_set_seed": pair_set_seed,
            "pair_set_file": pair_set_filename,
        }
        with open(metadata_filename, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)
        print(f"Metadata saved to {metadata_filename}")

    finally:
        try:
            conn.commit()
        except Exception:
            pass
        conn.close()
        print(f"SQLite results written to {db_path}")

# NOTE: No plots are generated in the main run loop. Plotting is handled by the SQLite
# helper above (plot_experimentC_from_sqlite), typically via --plot_only.


# ==========================
# Central-runner entrypoint
# ==========================

def run_experimentC(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    random_walk_steps: int = 20,
    iterations: int = 5,
    starting_nodes: int = 5,
    num_pairs: int = 5000,
    pair_offset: int = 0,
    pair_set_seed: int = 42,
    pair_set_dir: Optional[str] = None,
    save_pair_files: bool = False,
    output_dir: Optional[str] = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
    plot_metric: str = "all",
    plot_style: str = "box",
):
    """Run Experiment C from the all-experiments runner.

    Returns:
        (results_dict, plot_paths)

    Notes:
      - Results are stored in SQLite under: <output_dir>/results/rw<steps>_iter<iters>_nodes<nodes>/results.db
      - If resume=True and results.db exists, computation is skipped and only plotting is performed (if make_plot=True).
    """
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()

    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    # Load graphs
    file_pattern = os.path.join(root_dir, "data", "saved_cog_nets", f"G_n_{num_nodes}_k_{average_degree}_*.edgelist.gz")
    rewired_graphs = UtilityFunctions.load_graphs(file_pattern)
    if len(rewired_graphs) == 0:
        raise FileNotFoundError(f"No graphs found matching the pattern: {file_pattern}")

    # Persistent pair-set directory (defaults to repo experiment folder for reuse across runs)
    if pair_set_dir is None:
        pair_set_dir = os.path.join(os.path.dirname(__file__), "pair_sets")
    os.makedirs(pair_set_dir, exist_ok=True)

    pair_set_filename = os.path.join(
        pair_set_dir,
        f"pairs_nGraphs{len(rewired_graphs)}_N{num_nodes}_k{average_degree}_seed{pair_set_seed}_n5000.json",
    )

    if os.path.exists(pair_set_filename):
        with open(pair_set_filename, "r") as f:
            pair_set_indices = [tuple(x) for x in json.load(f)]
    else:
        graph_indices = list(range(len(rewired_graphs)))
        unordered = list(combinations(graph_indices, 2))
        ordered = unordered + [(j, i) for (i, j) in unordered]
        random.seed(pair_set_seed)
        pair_set_indices = random.sample(ordered, 5000)
        with open(pair_set_filename, "w") as f:
            json.dump(pair_set_indices, f, indent=4)

    # Select a slice to enable incremental runs
    if num_pairs is None or int(num_pairs) < 0:
        selected_pair_indices = pair_set_indices[int(pair_offset):]
    else:
        selected_pair_indices = pair_set_indices[int(pair_offset): int(pair_offset) + int(num_pairs)]

    if not selected_pair_indices:
        raise ValueError("No pairs selected. Check num_pairs/pair_offset.")

    sampled_indices = list(selected_pair_indices)
    sampled_pairs = [(rewired_graphs[i], rewired_graphs[j]) for (i, j) in sampled_indices]

    # Output paths
    config_folder_name = f"rw{int(random_walk_steps)}_iter{int(iterations)}_nodes{int(starting_nodes)}"
    config_results_dir = os.path.join(output_dir, config_folder_name)
    os.makedirs(config_results_dir, exist_ok=True)

    sampled_indices_filename = os.path.join(config_results_dir, "selected_pair_indices.json")
    with open(sampled_indices_filename, "w") as f:
        json.dump(sampled_indices, f, indent=4)

    db_path = os.path.join(config_results_dir, "results.db")
    metadata_filename = os.path.join(config_results_dir, "metadata.json")

    did_compute = False
    elapsed_time = None

    if not (resume and os.path.exists(db_path)):
        did_compute = True
        run_id = str(uuid.uuid4())

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Improve write performance
        cur.execute("PRAGMA journal_mode=WAL;")
        cur.execute("PRAGMA synchronous=NORMAL;")

        try:
            # Create tables
            cur.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    random_walk_steps INTEGER,
                    iterations INTEGER,
                    starting_nodes INTEGER,
                    pair_offset INTEGER,
                    num_pairs INTEGER,
                    seed INTEGER,
                    pair_set_seed INTEGER,
                    timestamp TEXT
                )
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS pair_results (
                    run_id TEXT,
                    idx1 INTEGER,
                    idx2 INTEGER,
                    starting_node INTEGER,
                    diameter_values TEXT,
                    num_concept_values TEXT,
                    unique_concept_values TEXT,
                    overlap_values TEXT
                )
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_results_lookup ON pair_results (idx1, idx2, starting_node);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_results_run ON pair_results (run_id);")
            conn.commit()

            cur.execute(
                "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                (run_id, int(random_walk_steps), int(iterations), int(starting_nodes), int(pair_offset), len(sampled_indices), int(seed), int(pair_set_seed)),
            )
            conn.commit()

            start_time = time.time()
            num_processes = cpu_count()

            args_list = [
                (idx1, idx2, sampled_pairs[i], int(random_walk_steps), int(iterations), int(starting_nodes), int(seed))
                for i, (idx1, idx2) in enumerate(sampled_indices)
            ]

            with Pool(processes=num_processes) as pool:
                for pair_idx, pair_result in enumerate(
                    tqdm(
                        pool.imap(process_pair_with_args, args_list),
                        total=len(sampled_pairs),
                        desc=f"Processing Pairs (Steps={random_walk_steps})",
                    )
                ):
                    idx1, idx2 = sampled_indices[pair_idx]

                    for starting_node, metrics in pair_result.items():
                        cur.execute(
                            "INSERT INTO pair_results VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                idx1,
                                idx2,
                                starting_node,
                                json.dumps(metrics["diameter_value_list"]),
                                json.dumps(metrics["num_concept_count_list"]),
                                json.dumps(metrics["unique_concept_count_list"]),
                                json.dumps(metrics["overlap_score_list"]),
                            ),
                        )

                    if save_pair_files:
                        pair_folder = os.path.join(config_results_dir, f"pair_{idx1}_{idx2}")
                        os.makedirs(pair_folder, exist_ok=True)
                        for starting_node, metrics in pair_result.items():
                            node_result = {
                                "pair_indices": (idx1, idx2),
                                "starting_node": starting_node,
                                "metrics": metrics,
                            }
                            node_filename = os.path.join(pair_folder, f"node_{starting_node}.json")
                            with open(node_filename, "w") as f:
                                json.dump(node_result, f, indent=4)

                    if pair_idx % 50 == 0:
                        conn.commit()

            conn.commit()
            end_time = time.time()
            elapsed_time = end_time - start_time

            metadata = {
                "experiment": "experimentC",
                "random_walk_steps": int(random_walk_steps),
                "iterations": int(iterations),
                "starting_nodes": int(starting_nodes),
                "seed": int(seed),
                "machine_id": machine_id,
                "time_taken_seconds": elapsed_time,
                "num_pairs": len(sampled_indices),
                "pair_offset": int(pair_offset),
                "pair_set_seed": int(pair_set_seed),
                "pair_set_file": pair_set_filename,
                "db_path": db_path,
            }
            if save_json:
                with open(metadata_filename, "w") as f:
                    json.dump(metadata, f, indent=4)

        finally:
            try:
                conn.commit()
            except Exception:
                pass
            conn.close()

    else:
        # Resume: try to read metadata for elapsed time
        if os.path.exists(metadata_filename):
            try:
                with open(metadata_filename, "r") as f:
                    meta = json.load(f)
                elapsed_time = meta.get("time_taken_seconds")
            except Exception:
                pass

    plot_paths: list = []
    if make_plot:
        out = plot_experimentC_from_sqlite([config_results_dir], plot_metric, plot_style, None)
        if out is not None:
            plot_paths.append(out)

    results = {
        "experiment": "experimentC",
        "config": {
            "N": num_nodes,
            "K": average_degree,
            "seed": int(seed),
            "random_walk_steps": int(random_walk_steps),
            "iterations": int(iterations),
            "starting_nodes": int(starting_nodes),
            "num_pairs": len(sampled_indices),
            "pair_offset": int(pair_offset),
            "pair_set_seed": int(pair_set_seed),
        },
        "output_dir": output_dir,
        "config_results_dir": config_results_dir,
        "db_path": db_path,
        "metadata_path": metadata_filename,
        "results_path": db_path,
        "time_taken": elapsed_time,
        "did_compute": did_compute,
        "plot_metric": plot_metric,
        "plot_style": plot_style,
    }

    return results, plot_paths