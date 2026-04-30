# PIPELINE 1


# cog_graphs_generator.py

import os
import gzip
import networkx as nx
import numpy as np
from tqdm import tqdm
import csv
import math

import random


# Prefer package-style imports. If executed directly, fall back to adding
# the project root to sys.path.
import sys

try:
    from core.utils import UtilityFunctions
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from core.utils import UtilityFunctions

import glob
import re
from typing import Set


# -------------------------
# Core options
# -------------------------
CONFIGS = [
    {"n": 100, "k": 4, "count": 500},
    {"n": 300, "k": 10, "count": 200},
]

# Where to save
OUTPUT_DIR = os.path.join("data", "new_saved_cog_nets")

# Where to save stats
STATS_DIR = os.path.join(OUTPUT_DIR, "stats")

# RewiringModule parameters
REWIRING_ITERATIONS = 5000

# Reproducibility
GLOBAL_SEED = 42

# Target modularity sweep controls (notebook-style)
HIGH_TEMPLATE_P = 0.01   # more lattice-like -> higher modularity
LOW_TEMPLATE_P = 0.80    # more random-like -> lower modularity
MODULARITY_TOL = 0.01
REWIRE_RESTARTS = 3      # random restarts per target if we get stuck
MODULARITY_RECOMPUTE_EVERY = 1  # NOTE: modularity is computed for each candidate move (new_m is always evaluated).

# --- Performance safeguards (do NOT change the goal: still targets a modularity sweep) ---
# If we haven't improved the absolute error to target for this many iterations, abort the restart early.
STAGNATION_PATIENCE = 800

# Only count an improvement if the error decreases by at least this much.
MIN_ERROR_IMPROVEMENT = 1e-4

# Skip expensive global stats (diameter / avg shortest path) during generation for larger graphs.
# These can be recomputed later during analysis if needed.
SKIP_EXPENSIVE_STATS_IF_N_GE = 200


def make_template_graph(n: int, k: int, p: float) -> nx.Graph:
    """
    Create a Watts-Strogatz template graph.
    """
    if k <= 0 or k % 2 != 0:
        raise ValueError(f"k must be a positive even integer. Given k={k}")
    if k >= n / 2:
        raise ValueError(f"k must be < n/2 for watts_strogatz_graph. Given n={n}, k={k}")
    if not (0 <= p <= 1):
        raise ValueError(f"p must be between 0 and 1. Given p={p}")

    return nx.watts_strogatz_graph(n, k, p)


def _existing_indices_for_config(n: int, k: int) -> Set[int]:
    """Return the set of already-generated graph indices for a given (n, k)."""
    pattern = os.path.join(OUTPUT_DIR, f"G_n_{n}_k_{k}_*.edgelist.gz")
    files = glob.glob(pattern)

    idxs: Set[int] = set()
    # Match the trailing index: G_n_<n>_k_<k>_<idx>.edgelist.gz
    rx = re.compile(rf"G_n_{n}_k_{k}_(\d+)\.edgelist\.gz$")
    for fp in files:
        m = rx.search(os.path.basename(fp))
        if m:
            try:
                idxs.add(int(m.group(1)))
            except ValueError:
                pass
    return idxs


def _next_index_for_config(n: int, k: int) -> int:
    """Return the next index to use for filenames for a given (n, k)."""
    idxs = _existing_indices_for_config(n, k)
    return (max(idxs) + 1) if idxs else 0


def save_graph_edgelist_gz(G: nx.Graph, filepath: str) -> None:
    """Save a graph as a gzipped edgelist.

    We write the edgelist ourselves ("u v\n") to avoid NetworkX's differing
    expectations about whether the file handle is binary or text across versions.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Text mode ensures we're writing `str`, and gzip will handle compression.
    with gzip.open(filepath, "wt", encoding="utf-8", newline="\n") as f:
        for u, v in G.edges():
            f.write(f"{u} {v}\n")




def compute_graph_stats(G: nx.Graph, modularity_value: float, *, skip_expensive: bool = False) -> dict:
    """Compute a set of useful graph statistics (best-effort)."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    avg_degree = (2.0 * n_edges / n_nodes) if n_nodes else 0.0

    density = nx.density(G) if n_nodes > 1 else 0.0
    avg_clustering = nx.average_clustering(G) if n_nodes > 1 else 0.0

    is_conn = nx.is_connected(G) if n_nodes > 0 else False

    diameter = ""
    avg_spl = ""
    if (not skip_expensive) and is_conn and n_nodes > 1:
        # These are expensive (all-pairs shortest paths). Skip during generation for large n.
        diameter = nx.diameter(G)
        avg_spl = nx.average_shortest_path_length(G)

    return {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "avg_degree": avg_degree,
        "density": density,
        "avg_clustering": avg_clustering,
        "is_connected": int(is_conn),
        "diameter": diameter,
        "avg_shortest_path_length": avg_spl,
        "modularity": modularity_value,
    }


def write_stats_csv(rows: list, outpath: str) -> None:
    """Write rows (list of dicts) to CSV."""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(outpath, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def safe_modularity(G: nx.Graph) -> float:
    """Compute modularity robustly using the project's UtilityFunctions."""
    try:
        return float(UtilityFunctions.compute_modularity(G))
    except Exception:
        return 0.0


def rewire_to_target_modularity(
    G: nx.Graph,
    target_modularity: float,
    max_iterations: int = REWIRING_ITERATIONS,
    tol: float = MODULARITY_TOL,
    restarts: int = REWIRE_RESTARTS,
    recompute_every: int = MODULARITY_RECOMPUTE_EVERY,
) -> tuple[nx.Graph, float, int, int]:
    """
    Notebook-style greedy rewiring:
    repeatedly replace one edge (u,v) with (u,w) if it moves modularity closer to target.

    Returns:
      (rewired_graph, achieved_modularity, iterations_used, success_flag)
    """
    if G.number_of_nodes() == 0:
        return G.copy(), 0.0, 0, 0

    best_graph = G.copy()
    best_m = safe_modularity(best_graph)
    best_it = 0
    best_success = 1 if abs(best_m - target_modularity) < tol else 0

    # Random restarts can help escape local plateaus
    for _restart in range(max(1, restarts)):
        Gr = G.copy()
        it_ = 0

        current_m = safe_modularity(Gr)
        if abs(current_m - target_modularity) < tol:
            return Gr, current_m, it_, 1

        # Stagnation tracking: if error to target stops improving, abort this restart early.
        best_err = abs(current_m - target_modularity)
        last_improve_it = 0

        # Main greedy loop
        while it_ < max_iterations:
            it_ += 1

            # Note: current_m is updated only on accepted moves; new_m requires a modularity computation each iteration.

            # pick a node with at least 1 neighbor
            node = random.choice(list(Gr.nodes()))
            if Gr.degree[node] == 0:
                continue

            u = node
            v = random.choice(list(Gr.neighbors(u)))

            # choose w not equal to u/v and not already connected to u
            possible = set(Gr.nodes()) - {u, v} - set(Gr.neighbors(u))
            if not possible:
                continue

            w = random.choice(list(possible))

            # apply tentative rewire
            Gr.remove_edge(u, v)
            Gr.add_edge(u, w)

            new_m = safe_modularity(Gr)

            # revert if not closer to target
            if abs(new_m - target_modularity) >= abs(current_m - target_modularity):
                Gr.remove_edge(u, w)
                Gr.add_edge(u, v)
            else:
                current_m = new_m

            # Update stagnation bookkeeping
            cur_err = abs(current_m - target_modularity)
            if (best_err - cur_err) > MIN_ERROR_IMPROVEMENT:
                best_err = cur_err
                last_improve_it = it_

            # Early abort if we have not made meaningful progress for a long time
            if (it_ - last_improve_it) >= STAGNATION_PATIENCE:
                break

            if abs(current_m - target_modularity) < tol:
                return Gr, current_m, it_, 1

        # Track best across restarts
        final_m = safe_modularity(Gr)
        if abs(final_m - target_modularity) < abs(best_m - target_modularity):
            best_graph = Gr
            best_m = final_m
            best_it = it_
            best_success = 1 if abs(best_m - target_modularity) < tol else 0

    return best_graph, best_m, best_it, best_success


def generate_and_save_for_config(n: int, k: int, count: int) -> None:
    """
    Generate 'count' cognitive graphs for a single (n, k) config.

    Current approach (notebook-style):
    - Build a high-modularity and low-modularity Watts–Strogatz template.
    - Compute their modularities and linearly sweep target modularity values between them.
    - For each target, start from the high-modularity template and greedily rewire edges
      to move modularity closer to the target.

    Outputs:
    - Gzipped edgelists under OUTPUT_DIR
    - Per-config CSV under STATS_DIR with both target/achieved modularity and graph stats.

    Performance safeguards:
    - Early-stop a restart when progress toward the target has stagnated.
    - Skip expensive global stats (diameter / avg shortest path) for large n during generation.
    """
    # Resume support: detect already-generated files and continue.
    existing_idxs = _existing_indices_for_config(n, k)
    existing_count = len(existing_idxs)

    if existing_idxs:
        expected = set(range(0, max(existing_idxs) + 1))
        missing = sorted(expected - existing_idxs)
        if missing:
            print(
                f"[warn] Detected missing indices for (n={n}, k={k}) (e.g., {missing[:5]}...). "
                "Will append new graphs with fresh indices instead of filling gaps."
            )

    if existing_count >= count:
        print(
            f"[skip] Already have {existing_count} graphs for (n={n}, k={k}) in {OUTPUT_DIR}. "
            f"Target is {count}."
        )
        return

    # Stats output per-config
    os.makedirs(STATS_DIR, exist_ok=True)
    stats_outpath = os.path.join(STATS_DIR, f"graph_stats_n_{n}_k_{k}.csv")

    # Build templates to estimate achievable modularity range
    high_template = make_template_graph(n, k, HIGH_TEMPLATE_P)
    low_template = make_template_graph(n, k, LOW_TEMPLATE_P)

    high_m = safe_modularity(high_template)
    low_m = safe_modularity(low_template)

    # Ensure sweep goes from higher -> lower modularity
    if low_m > high_m:
        high_template, low_template = low_template, high_template
        high_m, low_m = low_m, high_m

    targets = np.linspace(high_m, low_m, count)

    next_idx = _next_index_for_config(n, k)
    produced = existing_count

    stats_rows = []

    pbar = tqdm(
        total=count,
        initial=existing_count,
        desc=f"Generating graphs (n={n}, k={k})",
        unit="graph",
    )

    # If resuming, continue using the same target ordering by index
    while produced < count:
        target = float(targets[produced])

        # Start each graph from the HIGH template and rewire toward its target
        G_rewired, achieved_m, it_used, success = rewire_to_target_modularity(
            high_template,
            target_modularity=target,
            max_iterations=REWIRING_ITERATIONS,
            tol=MODULARITY_TOL,
            restarts=REWIRE_RESTARTS,
            recompute_every=MODULARITY_RECOMPUTE_EVERY,
        )

        idx = next_idx
        filename = f"G_n_{n}_k_{k}_{idx}.edgelist.gz"
        outpath = os.path.join(OUTPUT_DIR, filename)
        save_graph_edgelist_gz(G_rewired, outpath)

        row = {"n": n, "k": k, "idx": idx, "filename": filename}
        row.update(
            compute_graph_stats(
                G_rewired,
                achieved_m,
                skip_expensive=(n >= SKIP_EXPENSIVE_STATS_IF_N_GE),
            )
        )
        row.update(
            {
                "target_modularity": target,
                "achieved_modularity": achieved_m,
                "abs_target_error": abs(achieved_m - target),
                "rewire_iterations_used": it_used,
                "rewire_success": int(success),
                "template_high_p": HIGH_TEMPLATE_P,
                "template_low_p": LOW_TEMPLATE_P,
                "template_high_modularity": high_m,
                "template_low_modularity": low_m,
                "skipped_expensive_stats": int(n >= SKIP_EXPENSIVE_STATS_IF_N_GE),
            }
        )
        stats_rows.append(row)

        next_idx += 1
        produced += 1
        pbar.update(1)

    pbar.close()

    if stats_rows:
        write_stats_csv(stats_rows, stats_outpath)
        print(f"[stats] Wrote: {stats_outpath}")


def main() -> None:
    UtilityFunctions.set_global_seed(GLOBAL_SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for cfg in CONFIGS:
        generate_and_save_for_config(cfg["n"], cfg["k"], cfg["count"])

    print(f"Done. Files saved under: {OUTPUT_DIR}")
    print(f"Stats CSVs saved under: {STATS_DIR}")


if __name__ == "__main__":
    main()