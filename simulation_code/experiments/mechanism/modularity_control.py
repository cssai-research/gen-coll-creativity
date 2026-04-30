"""
Experiment A — Modularity as a cognitive control parameter
(renamed to: modularity_control.py)

Paper section: Methods / Model validation
"""


from __future__ import annotations

# Showing how creativity varies as we turn the 'modularity' knob

import os
import sys



# Prefer package-style imports. If this file is executed directly, fall back to
# adding the project root to sys.path.
try:
    from core.utils import UtilityFunctions
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from core.utils import UtilityFunctions


import json
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm



def save_progress(data: dict, output_path: str) -> None:
    """Save progress/results to a JSON file at an explicit path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Progress saved to {output_path}")


def plot_modularity_vs_rewiring_probability(
    rewiring_probabilities,
    modularity_means,
    modularity_stds,
    N,
    K,
    *,
    plots_dir: str,
    plot_filename: str,
) -> str:
    """Plots modularity vs rewiring probability (optionally with error bands) and saves the plot.

    Args:
        rewiring_probabilities (list[float]): List of rewiring probabilities.
        modularity_means (list[float]): Mean modularity at each probability.
        modularity_stds (list[float] | None): Std-dev modularity at each probability (can be None).
        N (int): Number of nodes in the graph.
        K (int): Average degree of the graph.
        plots_dir (str): Directory to save the plot.
        plot_filename (str): Filename for the saved plot.
    Returns:
        str: Path to the saved plot file.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(rewiring_probabilities, modularity_means, marker='o', linestyle='-', color='b', label='Mean Modularity')

    if modularity_stds is not None:
        lower = [m - s for m, s in zip(modularity_means, modularity_stds)]
        upper = [m + s for m, s in zip(modularity_means, modularity_stds)]
        plt.fill_between(rewiring_probabilities, lower, upper, alpha=0.2, label='±1 std')

    plt.title(f"Modularity vs Rewiring Probability (N={N}, K={K})")
    plt.xlabel("Rewiring Probability")
    plt.ylabel("Modularity")
    plt.legend()
    plt.grid(True)

    os.makedirs(plots_dir, exist_ok=True)

    plot_file = os.path.join(plots_dir, plot_filename)
    final_path = UtilityFunctions.save_figure_pdf(plt.gcf(), plot_file, dpi=300, tight=True)
    plt.close()
    print(f"Plot saved to {final_path}")
    return final_path

def load_progress(output_path: str):
    """Load progress/results from a JSON file at an explicit path."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            print(f"Loaded progress from {output_path}")
            return json.load(f)
    return None




def mean_std(values):
    """Return (mean, std) with NaNs ignored. Std uses sample std (ddof=1) when possible."""
    arr = np.array(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def run_ws_baseline(
    num_nodes: int,
    average_degree: int,
    rewiring_probabilities: list[float],
    num_graphs_per_p: int,
    *,
    results: dict,
    results_path: str,
) -> dict:
    """Run the notebook-style WS baseline and persist incremental progress.

    Returns the updated results dict.
    """
    modularity_by_p: dict[str, list[float]] = {}

    print("Notebook baseline: sweeping WS rewiring probability p and computing modularity...")
    for p in tqdm(rewiring_probabilities, desc="p sweep (WS baseline)"):
        vals: list[float] = []
        for _ in range(num_graphs_per_p):
            G = nx.watts_strogatz_graph(num_nodes, average_degree, p)
            try:
                vals.append(float(UtilityFunctions.compute_modularity(G)))
            except Exception as e:
                print(f"Modularity failed for p={p}: {e}")
                vals.append(float("nan"))

        # Use string keys for JSON stability
        modularity_by_p[str(p)] = vals

        results["steps"]["ws_baseline"] = {
            "modularity_by_p": modularity_by_p,
            "rewiring_probabilities": rewiring_probabilities,
            "num_graphs_per_p": num_graphs_per_p,
        }
        save_progress(results, results_path)

    return results


def plot_ws_baseline(results: dict, *, plots_dir: str, plot_filename: str) -> str:
    """Plot modularity vs p from saved ws_baseline results and return plot path."""
    ws = results["steps"]["ws_baseline"]
    rewiring_probabilities = ws["rewiring_probabilities"]
    modularity_by_p = ws["modularity_by_p"]

    means: list[float] = []
    stds: list[float] = []
    for p in rewiring_probabilities:
        m, s = mean_std(modularity_by_p[str(p)])
        means.append(m)
        stds.append(s)

    return plot_modularity_vs_rewiring_probability(
        rewiring_probabilities,
        means,
        stds,
        results["num_nodes"],
        results["average_degree"],
        plots_dir=plots_dir,
        plot_filename=plot_filename,
    )


def run_experimentA(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    rewiring_probabilities=None,
    num_graphs_per_p: int = 15,
    output_dir: str | None = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
):
    """Run Experiment A with provided parameters.

    Returns:
        (results_dict, plot_path_or_none)
    """
    # Consistent environment setup
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()
    start_time = time.time()

    if rewiring_probabilities is None:
        rewiring_probabilities = [round(x, 3) for x in list(np.linspace(0, 1, 21))]

    # Default outputs go under a centralized _outputs directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "_outputs", "modularity_control")

    results_dir = os.path.join(output_dir, "results")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    results_filename = f"results_ws_N{num_nodes}_K{average_degree}_seed{seed}.json"
    results_path = os.path.join(results_dir, results_filename)

    plot_filename = f"modularity_vs_rewiring_probability_N{num_nodes}_K{average_degree}_seed{seed}.pdf"

    # Load progress for this specific configuration if requested
    results = (load_progress(results_path) if resume else None) or {
        "experiment": "experimentA",
        "seed": seed,
        "machine_id": machine_id,
        "num_nodes": num_nodes,
        "average_degree": average_degree,
        "results_path": results_path,
        "steps": {},
    }

    # Always trust passed parameters
    results["seed"] = seed
    results["num_nodes"] = num_nodes
    results["average_degree"] = average_degree
    results["results_path"] = results_path

    try:
        if "ws_baseline" not in results["steps"]:
            results = run_ws_baseline(
                num_nodes,
                average_degree,
                rewiring_probabilities,
                num_graphs_per_p,
                results=results,
                results_path=results_path,
            )

        plot_path = None
        if make_plot:
            plot_path = plot_ws_baseline(
                results,
                plots_dir=plots_dir,
                plot_filename=plot_filename,
            )

        # Timing
        end_time = time.time()
        results["time_taken"] = end_time - start_time

        if save_json:
            save_progress(results, results_path)

        return results, plot_path

    except Exception as e:
        print(f"An error occurred: {e}")
        if save_json and isinstance(results, dict):
            save_progress(results, results_path)
        raise


# Preferred usage (from repo root):
#   python -m experiments.mechanism.modularity_control
# Direct execution also works via the import fallback above.
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run Experiment A (WS modularity sweep).")
    parser.add_argument("--num-nodes", type=int, default=100, help="Number of nodes (N).")
    parser.add_argument("--average-degree", type=int, default=4, help="Average degree (K).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num-graphs-per-p", type=int, default=15, help="Graphs per rewiring probability.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write results JSON and plots (defaults to experiments/mechanism/_outputs/modularity_control).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load/continue from an existing results JSON.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate the plot.",
    )

    args = parser.parse_args()

    results, plot_path = run_experimentA(
        num_nodes=args.num_nodes,
        average_degree=args.average_degree,
        seed=args.seed,
        num_graphs_per_p=args.num_graphs_per_p,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        make_plot=not args.no_plot,
        save_json=True,
    )

    # Helpful final print
    print("Done.")
    print(f"Results JSON: {results.get('results_path')}")
    if plot_path:
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()