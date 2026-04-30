"""Experiment E: Overlap as a Mechanistic Driver of Inspiration

This experiment tests whether overlap between sampled source and target subgraphs predicts
inspiration outcomes (e.g., novelty and structural change) in a source→target integration step.

Design notes (why we do things this way):
- Random-walk length is treated as fixed (chosen based on Experiments C/D). Experiment E focuses
  on overlap as the explanatory variable.
- Pair sampling is WITHOUT replacement (unique pairs) to avoid overweighting a subset of pairs and
  to improve between-pair generalization.
- Runs are incremental and resumable: we use a persistent pair set and store results in a single
  SQLite database per configuration.
- Reproducibility: we use deterministic seeds per (pair, starting_node, iteration).

Outputs
-------
A single SQLite DB per configuration:
  experiments/experimentE/results/<config>/results.db
Plots are saved into:
  experiments/experimentE/results/<config>/plots/

"""

from __future__ import annotations

import os
import sys
import json
import time
import datetime
import uuid
import hashlib
import sqlite3
import argparse
import random
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import linregress

# Prefer package-style imports. If executed directly, fall back to adding
# the project root to sys.path.
try:
    from core.utils import UtilityFunctions
    from core.inspiration import integrate_subgraph
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from core.utils import UtilityFunctions
    from core.inspiration import integrate_subgraph

# -------------------------
# Small I/O helpers
# -------------------------

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


# -------------------------
# Paths / local output dirs
# -------------------------
BASE_DIR = os.path.dirname(__file__)
RESULTS_ROOT = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_ROOT, exist_ok=True)

PAIRSET_DIR = os.path.join(RESULTS_ROOT, "pair_sets")
os.makedirs(PAIRSET_DIR, exist_ok=True)


# -------------------------
# Deterministic seeding
# -------------------------

def _stable_int_seed(*parts, mod=(2**31 - 1)):
    """Stable deterministic integer seed from arbitrary parts."""
    h = hashlib.sha256(repr(parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") % mod


# -------------------------
# CLI
# -------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Experiment E (SQLite) or plot results from SQLite.")

    # Legacy shim: allow --mode run/plot/test
    parser.add_argument(
        "--mode",
        type=str,
        default="run",
        choices=["run", "plot", "test"],
        help="Legacy shim. Prefer --plot_only 1 for plotting.",
    )

    # Run configuration
    parser.add_argument("--random_walk_steps", type=int, default=20, help="Random walk length (fixed for Experiment E).")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations per starting node.")
    parser.add_argument("--starting_nodes", type=int, default=10, help="Number of starting nodes per pair.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")

    # Pair-set / incremental execution
    parser.add_argument("--pair_set_seed", type=int, default=42, help="Seed defining the persistent pair set.")
    parser.add_argument("--num_pairs", type=int, default=500, help="Number of unique pairs to process in this run.")
    parser.add_argument("--pair_offset", type=int, default=0, help="Offset into the persistent pair set (for chunking).")

    # Plotting
    parser.add_argument("--plot_only", type=int, default=0, help="If 1, do not run; only plot from SQLite.")
    parser.add_argument(
        "--plot_dbs",
        nargs="+",
        default=None,
        help="One or more config directories or .db paths to plot. If multiple, they must be compatible.",
    )
    parser.add_argument("--plot_out", type=str, default=None, help="Optional output filename (saved as PDF).")

    return parser.parse_args()


# -------------------------
# Pair set helpers
# -------------------------

# The following legacy helpers are unused: run_experimentE() uses local helpers rooted at output_dir.
# def _pairset_path(pair_set_seed: int, N: int, K: int):
#     return os.path.join(PAIRSET_DIR, f"pairset_n{N}_k{K}_seed{pair_set_seed}.json")
#
#
# def load_or_create_pairset(pair_metadata, pair_set_seed: int, N: int, K: int):
#     """Create a persistent shuffled list of pair_index values (unique, without replacement)."""
#     path = _pairset_path(pair_set_seed, N, K)
#     if os.path.exists(path):
#         with open(path, "r") as f:
#             data = json.load(f)
#         # Back-compat: allow list directly or dict with key
#         if isinstance(data, list):
#             return data
#         return data.get("pair_indices", [])
#
#     rng = random.Random(pair_set_seed)
#     pair_indices = [p["pair_index"] for p in pair_metadata]
#     rng.shuffle(pair_indices)
#
#     with open(path, "w") as f:
#         json.dump({"pair_set_seed": pair_set_seed, "pair_indices": pair_indices}, f, indent=2)
#
#     return pair_indices


# -------------------------
# SQLite
# -------------------------

def _resolve_db_path(path_or_dir: str):
    if path_or_dir.endswith(".db"):
        return path_or_dir
    return os.path.join(path_or_dir, "results.db")


def init_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            random_walk_steps INTEGER,
            iterations INTEGER,
            starting_nodes INTEGER,
            seed INTEGER,
            pair_set_seed INTEGER,
            pair_offset INTEGER,
            num_pairs INTEGER,
            timestamp TEXT
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS pair_results (
            run_id TEXT,
            pair_index INTEGER,
            src_id TEXT,
            tgt_id TEXT,
            starting_node INTEGER,
            overlap_vals TEXT,
            diameter_delta_abs TEXT,
            diameter_delta_signed TEXT,
            num_concepts_delta_abs TEXT,
            unique_concepts_gained TEXT
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_lookup ON pair_results (pair_index, starting_node);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pair_run ON pair_results (run_id);")

    conn.commit()
    return conn


# -------------------------
# Computation
# -------------------------

def process_pair_wrapper(args):
    """Pool wrapper."""
    try:
        return process_pair(*args)
    except Exception as e:
        pair_index = args[2]
        print(f"Error processing pair_index={pair_index}: {e}")
        return None


def process_pair(source_graph, target_graph, pair_index: int, src_mod: float, tgt_mod: float,
                 iterations: int, random_walk_steps: int, starting_nodes: int,
                 global_seed: int):
    """Compute overlap and inspiration outcomes for one ordered source→target pair."""

    # Deterministic starting-node selection from SOURCE graph
    nodes_sorted = sorted(list(source_graph.nodes))
    if len(nodes_sorted) <= starting_nodes:
        sampled_starting_nodes = nodes_sorted
    else:
        rng_nodes = random.Random(_stable_int_seed(global_seed, pair_index, "starting_nodes"))
        sampled_starting_nodes = rng_nodes.sample(nodes_sorted, k=starting_nodes)

    base_tg = target_graph  # pristine

    rows = []  # one row per starting_node

    for starting_node in sampled_starting_nodes:
        overlap_list = []
        dia_abs_list = []
        dia_signed_list = []
        num_abs_list = []
        uniq_gain_list = []

        # If starting node not present in target, skip (can happen if graphs differ in node ids)
        if starting_node not in base_tg:
            continue

        for iter_idx in range(iterations):
            # Deterministic seed for this (pair, node, iter, rw)
            seed_walk = _stable_int_seed(global_seed, pair_index, starting_node, iter_idx, random_walk_steps, "walk")
            UtilityFunctions.set_global_seed(seed_walk)

            source_subgraph = UtilityFunctions.subgraph_from_path(
                source_graph,
                UtilityFunctions.random_walk_path(source_graph, random_walk_steps, starting_node),
            )
            target_before = UtilityFunctions.subgraph_from_path(
                base_tg,
                UtilityFunctions.random_walk_path(base_tg, random_walk_steps, starting_node),
            )
            if source_subgraph.number_of_nodes() == 0 or target_before.number_of_nodes() == 0:
                continue

            diameter_before = UtilityFunctions.compute_diameter(target_before)
            concepts_before = set(target_before.nodes)
            overlap_score = UtilityFunctions.compute_overlap(source_subgraph, target_before)

            tg_after = integrate_subgraph(base_tg, source_subgraph)

            target_after = UtilityFunctions.subgraph_from_path(
                tg_after,
                UtilityFunctions.random_walk_path(tg_after, random_walk_steps, starting_node),
            )
            if target_after.number_of_nodes() == 0:
                continue

            diameter_after = UtilityFunctions.compute_diameter(target_after)
            concepts_after = set(target_after.nodes)

            overlap_list.append(float(overlap_score))
            dia_abs_list.append(float(abs(diameter_after - diameter_before)))
            dia_signed_list.append(float(diameter_after - diameter_before))
            num_abs_list.append(float(abs(len(concepts_after) - len(concepts_before))))
            uniq_gain_list.append(float(len(concepts_after - concepts_before)))

        # Only store if we got any samples
        if overlap_list:
            rows.append(
                {
                    "pair_index": int(pair_index),
                    "src_id": str(source_graph.graph.get("id", "")),
                    "tgt_id": str(target_graph.graph.get("id", "")),
                    "src_mod": float(src_mod),
                    "tgt_mod": float(tgt_mod),
                    "starting_node": int(starting_node),
                    "overlap_vals": overlap_list,
                    "diameter_delta_abs": dia_abs_list,
                    "diameter_delta_signed": dia_signed_list,
                    "num_concepts_delta_abs": num_abs_list,
                    "unique_concepts_gained": uniq_gain_list,
                }
            )

    return rows


# -------------------------
# Plotting from SQLite
# -------------------------

def _clean_xy(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _add_linear_trend(ax, x, y):
    x, y = _clean_xy(x, y)
    if len(x) < 3:
        return
    s = linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    ys = s.slope * xs + s.intercept
    ax.plot(xs, ys, linewidth=2, alpha=0.9)
    r2 = s.rvalue**2
    ax.text(
        0.98,
        0.02,
        f"slope={s.slope:.3f}\nR²={r2:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
    )


def bootstrap_slope_cluster(x, y, groups, B=1000, seed=0):
    """Cluster bootstrap slope of y~x, resampling clusters with replacement."""
    rng = np.random.default_rng(seed)

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    groups = np.asarray(groups, dtype=object)[mask]

    g2idx = defaultdict(list)
    for i, g in enumerate(groups):
        g2idx[g].append(i)

    uniq = np.array(list(g2idx.keys()), dtype=object)
    G = len(uniq)
    if G == 0 or len(x) < 3:
        return np.nan, (np.nan, np.nan)

    slopes = []
    for _ in range(B):
        boots = rng.choice(uniq, size=G, replace=True)
        take = []
        for gi in boots:
            take.extend(g2idx[gi])
        take = np.asarray(take, dtype=int)
        if take.size < 3:
            continue
        s = linregress(x[take], y[take])
        if np.isfinite(s.slope):
            slopes.append(s.slope)

    if not slopes:
        return np.nan, (np.nan, np.nan)

    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return float(np.mean(slopes)), (float(lo), float(hi))


def _read_run_header(db_path: str):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT random_walk_steps, iterations, starting_nodes, seed, pair_set_seed, pair_offset, num_pairs, timestamp FROM runs ORDER BY timestamp DESC LIMIT 1")
    row = cur.fetchone()
    conn.close()
    return row


def plot_experimentE_from_sqlite(db_path_or_dir: str, out_path: str = None):
    db_path = _resolve_db_path(db_path_or_dir)
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    header = _read_run_header(db_path)
    if header:
        rw, iters, nodes, seed, pair_set_seed, pair_offset, num_pairs, ts = header
    else:
        rw = iters = nodes = seed = pair_set_seed = pair_offset = num_pairs = None
        ts = "?"

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT pair_index, starting_node, overlap_vals, diameter_delta_abs, diameter_delta_signed, num_concepts_delta_abs, unique_concepts_gained FROM pair_results")
    rows = cur.fetchall()
    conn.close()

    overlap_vals, dia_abs, dia_signed, num_abs, uniq_gains = [], [], [], [], []
    pair_ids_all, pair_ids_signed = [], []

    # Per-pair means (Panel 4)
    pair_to_ovlp = defaultdict(list)
    pair_to_uniq = defaultdict(list)

    for pair_index, starting_node, ov_json, dia_abs_json, dia_signed_json, num_abs_json, uniq_json in rows:
        try:
            ov_list = json.loads(ov_json) if ov_json else []
            dia_abs_list = json.loads(dia_abs_json) if dia_abs_json else []
            dia_signed_list = json.loads(dia_signed_json) if dia_signed_json else []
            num_abs_list = json.loads(num_abs_json) if num_abs_json else []
            uniq_list = json.loads(uniq_json) if uniq_json else []
        except Exception:
            continue

        # Align by length (defensive)
        L = min(len(ov_list), len(dia_abs_list), len(dia_signed_list), len(num_abs_list), len(uniq_list))
        ov_list = ov_list[:L]
        dia_abs_list = dia_abs_list[:L]
        dia_signed_list = dia_signed_list[:L]
        num_abs_list = num_abs_list[:L]
        uniq_list = uniq_list[:L]

        for i in range(L):
            ov = float(ov_list[i])
            overlap_vals.append(ov)
            dia_abs.append(float(dia_abs_list[i]))
            dia_signed.append(float(dia_signed_list[i]))
            num_abs.append(float(num_abs_list[i]))
            uniq_gains.append(float(uniq_list[i]))
            pair_ids_all.append(str(pair_index))
            pair_ids_signed.append(str(pair_index))

            pair_to_ovlp[str(pair_index)].append(ov)
            pair_to_uniq[str(pair_index)].append(float(uniq_list[i]))

    # Pair means
    pair_mean_x, pair_mean_y, pair_ids_mean = [], [], []
    for pid in pair_to_ovlp.keys():
        xs = pair_to_ovlp[pid]
        ys = pair_to_uniq[pid]
        if xs and ys:
            pair_mean_x.append(float(np.mean(xs)))
            pair_mean_y.append(float(np.mean(ys)))
            pair_ids_mean.append(pid)

    # Figure
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    title = f"Experiment E • RW={rw} • iters={iters} • start_nodes={nodes} • pairs≈{num_pairs} • seed={seed} • pairset={pair_set_seed} • offset={pair_offset}"
    fig.suptitle(title, y=0.98, fontsize=10)

    # Panel 1: overlap vs diameter delta (signed + show abs via label)
    x1 = np.asarray(overlap_vals, float)
    y1 = np.asarray(dia_signed, float)
    axes[0].scatter(x1, y1, alpha=0.5)
    axes[0].set_title("Overlap vs Δ diameter (signed)")
    axes[0].set_xlabel("Overlap")
    axes[0].set_ylabel("Δ diameter (signed)")
    axes[0].axhline(0, linestyle="--", linewidth=1)
    _add_linear_trend(axes[0], x1, y1)
    ms1, (lo1, hi1) = bootstrap_slope_cluster(x1, y1, groups=pair_ids_signed, B=1000, seed=0)
    if np.isfinite(ms1):
        axes[0].text(
            0.98,
            0.98,
            f"boot slope≈{ms1:.3f}\n95% CI [{lo1:.3f}, {hi1:.3f}]",
            transform=axes[0].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Panel 2: overlap vs num concepts delta (absolute)
    x2 = np.asarray(overlap_vals, float)
    y2 = np.asarray(num_abs, float)
    axes[1].scatter(x2, y2, alpha=0.5)
    axes[1].set_title("Overlap vs Δ # concepts (absolute)")
    axes[1].set_xlabel("Overlap")
    axes[1].set_ylabel("Δ # concepts (absolute)")
    _add_linear_trend(axes[1], x2, y2)
    ms2, (lo2, hi2) = bootstrap_slope_cluster(x2, y2, groups=pair_ids_all, B=1000, seed=0)
    if np.isfinite(ms2):
        axes[1].text(
            0.98,
            0.98,
            f"boot slope≈{ms2:.3f}\n95% CI [{lo2:.3f}, {hi2:.3f}]",
            transform=axes[1].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Panel 3: overlap vs unique gains
    x3 = np.asarray(overlap_vals, float)
    y3 = np.asarray(uniq_gains, float)
    axes[2].scatter(x3, y3, alpha=0.5)
    axes[2].set_title("Overlap vs # new concepts")
    axes[2].set_xlabel("Overlap")
    axes[2].set_ylabel("# new concepts")
    _add_linear_trend(axes[2], x3, y3)
    ms3, (lo3, hi3) = bootstrap_slope_cluster(x3, y3, groups=pair_ids_all, B=1000, seed=0)
    if np.isfinite(ms3):
        axes[2].text(
            0.98,
            0.98,
            f"boot slope≈{ms3:.3f}\n95% CI [{lo3:.3f}, {hi3:.3f}]",
            transform=axes[2].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Panel 4: pair means
    x4 = np.asarray(pair_mean_x, float)
    y4 = np.asarray(pair_mean_y, float)
    axes[3].scatter(x4, y4, alpha=0.6)
    axes[3].set_title("Overlap vs # new concepts (pair means)")
    axes[3].set_xlabel("Overlap")
    axes[3].set_ylabel("mean # new concepts per pair")
    _add_linear_trend(axes[3], x4, y4)
    ms4, (lo4, hi4) = bootstrap_slope_cluster(x4, y4, groups=pair_ids_mean, B=1000, seed=0)
    if np.isfinite(ms4):
        axes[3].text(
            0.98,
            0.98,
            f"boot slope≈{ms4:.3f}\n95% CI [{lo4:.3f}, {hi4:.3f}]",
            transform=axes[3].transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )

    # Save
    config_dir = os.path.dirname(db_path)
    plots_dir = os.path.join(config_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if out_path is None:
        out_path = os.path.join(plots_dir, "overlap_effects.pdf")
    else:
        out_path = os.path.join(plots_dir, os.path.basename(out_path))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    print(f"Saved plot to {final_path}")

    return final_path



# -------------------------
# run_experimentE: Central runner importable function
# -------------------------

def run_experimentE(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    random_walk_steps: int = 20,
    iterations: int = 10,
    starting_nodes: int = 10,
    pair_set_seed: int = 42,
    num_pairs: int = 500,
    pair_offset: int = 0,
    output_dir: None = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
    plot_out: None = None,
) -> tuple[dict, list]:
    """Run Experiment E.

    Returns:
        (results_dict, plot_paths)

    Notes:
      - Results are stored in SQLite under: <output_dir>/results/<config>/results.db
      - If resume=True and DB exists, computation is skipped and only plotting runs (if make_plot=True).
    """
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()

    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    results_root = os.path.join(output_dir, "results")
    pairset_dir = os.path.join(results_root, "pair_sets")
    _ensure_dir(results_root)
    _ensure_dir(pairset_dir)

    # Data location
    data_dir = os.path.abspath(os.path.join(root_dir, "data", "saved_cog_nets"))

    graph_metadata_file = os.path.join(data_dir, f"graph_metadata_master_n{num_nodes}_k{average_degree}.json")
    pair_metadata_file = os.path.join(data_dir, f"pair_index_master_n{num_nodes}_k{average_degree}.json")

    if not os.path.exists(graph_metadata_file):
        raise FileNotFoundError(f"Graph metadata not found: {graph_metadata_file}")
    if not os.path.exists(pair_metadata_file):
        raise FileNotFoundError(f"Pair metadata not found: {pair_metadata_file}")

    with open(graph_metadata_file, "r") as f:
        graph_metadata = json.load(f)
    with open(pair_metadata_file, "r") as f:
        pair_metadata = json.load(f)

    # Load all graphs by graph_id
    graphs = {}
    for g in graph_metadata:
        file_path = os.path.join(data_dir, g["filename"])
        graph = nx.read_edgelist(file_path, create_using=nx.Graph(), nodetype=int)
        graph_id = str(g.get("graph_id"))
        graph.graph["id"] = graph_id
        graphs[graph_id] = graph

    # Local pairset helpers (avoid reliance on module globals)
    def _pairset_path_local(pair_set_seed_local: int, N: int, K: int):
        return os.path.join(pairset_dir, f"pairset_n{N}_k{K}_seed{pair_set_seed_local}.json")

    def load_or_create_pairset_local(pair_metadata_local, pair_set_seed_local: int, N: int, K: int):
        path = _pairset_path_local(pair_set_seed_local, N, K)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            return data.get("pair_indices", [])

        rng = random.Random(pair_set_seed_local)
        pair_indices_local = [p["pair_index"] for p in pair_metadata_local]
        rng.shuffle(pair_indices_local)

        with open(path, "w") as f:
            json.dump({"pair_set_seed": pair_set_seed_local, "pair_indices": pair_indices_local}, f, indent=2)

        return pair_indices_local

    # Persistent pair set (unique pairs)
    pair_indices = load_or_create_pairset_local(pair_metadata, pair_set_seed, num_nodes, average_degree)
    if pair_offset < 0:
        pair_offset = 0

    slice_start = pair_offset
    slice_end = min(len(pair_indices), pair_offset + num_pairs)
    selected_pair_indices = pair_indices[slice_start:slice_end]

    if not selected_pair_indices:
        raise ValueError("No pairs selected. Check pair_offset/num_pairs.")

    # Map pair_index -> metadata row
    meta_by_index = {p["pair_index"]: p for p in pair_metadata}

    # Config folder name
    config_name = (
        f"rw{int(random_walk_steps)}_iter{int(iterations)}_nodes{int(starting_nodes)}_"
        f"pairs{len(selected_pair_indices)}_seed{int(seed)}_pairset{int(pair_set_seed)}_off{int(pair_offset)}"
    )
    config_dir = os.path.join(results_root, config_name)
    _ensure_dir(config_dir)

    db_path = os.path.join(config_dir, "results.db")
    metadata_path = os.path.join(config_dir, "metadata.json")

    did_compute = False
    elapsed = None

    # Resume behavior
    if not (resume and os.path.exists(db_path)):
        did_compute = True

        conn = init_db(db_path)
        cur = conn.cursor()

        run_id = str(uuid.uuid4())
        cur.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                int(random_walk_steps),
                int(iterations),
                int(starting_nodes),
                int(seed),
                int(pair_set_seed),
                int(pair_offset),
                len(selected_pair_indices),
                datetime.datetime.now().isoformat(),
            ),
        )
        conn.commit()

        pool_args = []
        for pair_index in selected_pair_indices:
            p = meta_by_index[pair_index]
            src_id_raw = p.get("src_id", p.get("source_graph_id"))
            tgt_id_raw = p.get("tgt_id", p.get("target_graph_id"))
            src_id = str(src_id_raw)
            tgt_id = str(tgt_id_raw)

            if src_id not in graphs or tgt_id not in graphs:
                raise KeyError(f"Pair metadata refers to missing graph id(s): src={src_id} tgt={tgt_id}")

            src_mod = p.get("source_modularity", p.get("src_mod", 0.0))
            tgt_mod = p.get("target_modularity", p.get("tgt_mod", 0.0))

            pool_args.append(
                (
                    graphs[src_id],
                    graphs[tgt_id],
                    int(pair_index),
                    float(src_mod),
                    float(tgt_mod),
                    int(iterations),
                    int(random_walk_steps),
                    int(starting_nodes),
                    int(seed),
                )
            )

        start_time = time.time()
        nproc = max(1, cpu_count())

        try:
            with Pool(processes=nproc) as pool:
                for i, rows in enumerate(
                    tqdm(pool.imap(process_pair_wrapper, pool_args), total=len(pool_args), desc="Processing pairs")
                ):
                    if not rows:
                        continue

                    for r in rows:
                        cur.execute(
                            "INSERT INTO pair_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                run_id,
                                r["pair_index"],
                                r["src_id"],
                                r["tgt_id"],
                                r["starting_node"],
                                json.dumps(r["overlap_vals"]),
                                json.dumps(r["diameter_delta_abs"]),
                                json.dumps(r["diameter_delta_signed"]),
                                json.dumps(r["num_concepts_delta_abs"]),
                                json.dumps(r["unique_concepts_gained"]),
                            ),
                        )

                    if i % 25 == 0:
                        conn.commit()

        finally:
            conn.commit()
            conn.close()

        elapsed = float(time.time() - start_time)

        metadata = {
            "experiment": "experimentE",
            "run_id": run_id,
            "N": num_nodes,
            "K": average_degree,
            "random_walk_steps": int(random_walk_steps),
            "iterations": int(iterations),
            "starting_nodes": int(starting_nodes),
            "seed": int(seed),
            "machine_id": machine_id,
            "pair_set_seed": int(pair_set_seed),
            "pair_offset": int(pair_offset),
            "num_pairs": len(selected_pair_indices),
            "time_taken_seconds": elapsed,
            "db_path": db_path,
        }
        if save_json:
            save_progress(metadata, metadata_path)

    else:
        # Resume path: if we are reusing an existing DB, the run time might not
        # be present (e.g., metadata.json deleted). Ensure we always return a
        # numeric value so callers can safely cast/format.
        meta = load_progress(metadata_path)
        if isinstance(meta, dict):
            val = meta.get("time_taken_seconds")
            if val is None:
                val = meta.get("time_taken")
            elapsed = float(val) if val is not None else 0.0
        else:
            elapsed = 0.0

    plot_paths: list = []
    if make_plot:
        out = plot_experimentE_from_sqlite(config_dir, out_path=plot_out)
        plot_paths.append(out)

    results = {
        "experiment": "experimentE",
        "config": {
            "N": num_nodes,
            "K": average_degree,
            "seed": int(seed),
            "random_walk_steps": int(random_walk_steps),
            "iterations": int(iterations),
            "starting_nodes": int(starting_nodes),
            "pair_set_seed": int(pair_set_seed),
            "num_pairs": len(selected_pair_indices),
            "pair_offset": int(pair_offset),
        },
        "output_dir": output_dir,
        "config_dir": config_dir,
        "db_path": db_path,
        "metadata_path": metadata_path,
        "time_taken": elapsed,
        "did_compute": did_compute,
    }

    return results, plot_paths

# Preferred usage (from repo root):
#   python -m experiments.experimentE.experimentE
# Direct execution also works via the import fallback above.
def main():
    args = parse_arguments()

    # Legacy shim
    if args.mode == "plot":
        args.plot_only = 1
    if args.mode == "test":
        # small test defaults
        args.random_walk_steps = 5
        args.iterations = 1
        args.starting_nodes = 2
        args.num_pairs = 10

    # Plot-only
    if bool(args.plot_only):
        if not args.plot_dbs:
            raise ValueError("--plot_only 1 requires --plot_dbs with a config directory or results.db")
        for p in args.plot_dbs:
            out = plot_experimentE_from_sqlite(p, out_path=args.plot_out)
            print(f"Plot: {out}")
        return

    results, plot_paths = run_experimentE(
        num_nodes=100,  # NOTE: CLI keeps historical defaults; central runner overrides these.
        average_degree=4,
        seed=args.seed,
        random_walk_steps=args.random_walk_steps,
        iterations=args.iterations,
        starting_nodes=args.starting_nodes,
        pair_set_seed=args.pair_set_seed,
        num_pairs=args.num_pairs,
        pair_offset=args.pair_offset,
        output_dir=os.path.dirname(__file__),
        resume=True,
        save_json=True,
        make_plot=True,
        plot_out=args.plot_out,
    )

    print("Done.")
    print(f"DB: {results.get('db_path')}")
    if plot_paths:
        for p in plot_paths:
            print(f"Plot: {p}")


if __name__ == "__main__":
    main()