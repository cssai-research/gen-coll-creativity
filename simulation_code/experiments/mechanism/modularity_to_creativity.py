from __future__ import annotations
"""Experiment B runner.

Provides an importable API:
  run_experimentB(...) -> (results_dict, plot_paths)

- results_dict is JSON-serializable.
- plot_paths is a list of saved plot file paths (may be empty if make_plot=False).

This file can also be executed directly as a CLI.
"""

# Standard library imports
import os
import multiprocessing
import sys
import json
import time


import argparse

# Third-party imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Prefer package-style imports. If executed directly, fall back to adding
# the project root to sys.path.
try:
    from core.utils import UtilityFunctions
    from core.metrics import GraphAnalyzer
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from core.utils import UtilityFunctions
    from core.metrics import GraphAnalyzer


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_progress(data: dict, output_path: str) -> None:
    """Save results to an explicit JSON file path."""
    _ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def load_progress(output_path: str):
    """Load results from an explicit JSON file path, or return None."""
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            return json.load(f)
    return None


def _plot_regression(report_df: pd.DataFrame, *, x: str, y: str, out_path: str, scatter_alpha: float) -> str:
    """Create and save a regression plot; returns out_path."""
    _ensure_dir(os.path.dirname(out_path))

    # Create an explicit figure so we can save/close deterministically.
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    sns.regplot(x=x, y=y, data=report_df, ci=95, scatter_kws={"alpha": scatter_alpha}, ax=ax)

    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    plt.close(fig)
    return final_path


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run experimentB with configurable parameters.")
    parser.add_argument("--num-nodes", type=int, default=100, help="Number of nodes (N).")
    parser.add_argument("--average-degree", type=int, default=4, help="Average degree (K).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--random-walk-steps", type=int, default=50, help="Number of random walk steps.")
    parser.add_argument("--eval-repeats", type=int, default=30, help="Number of evaluation repeats.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write results JSON and plots (defaults to experiments/mechanism/_outputs/modularity_to_creativity).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not load/continue from an existing results JSON.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not generate plots.",
    )
    return parser.parse_args()


def evaluation_helper(params):
    """Helper function to perform evaluation on rewired graphs."""
    rewired_graphs = params['rewired_graphs']
    random_walk_steps = params.get('random_walk_steps', 50)

    start_node = GraphAnalyzer.find_common_start_node(rewired_graphs)
    if start_node is not None:
        # Perform random walks on each graph starting from the common start node
        # and build (subgraph, path) tuples expected by GraphAnalyzer.
        graph_paths = []
        for graph in rewired_graphs:
            path = UtilityFunctions.random_walk_path(graph, random_walk_steps, start_node)
            subgraph = UtilityFunctions.subgraph_from_path(graph, path)
            graph_paths.append((subgraph, path))
    else:
        return []

    return GraphAnalyzer.report_graph_metrics_simpler(rewired_graphs, graph_paths)


def run_experimentB(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    random_walk_steps: int = 50,
    eval_repeats: int = 30,
    output_dir: str | None = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
) -> tuple[dict, list[str]]:
    """Run Experiment B.

    Returns:
        (results_dict, plot_paths)
    """
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()

    # Default outputs go under a centralized _outputs directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "_outputs", "modularity_to_creativity")

    plots_dir = os.path.join(output_dir, "plots")
    results_dir = os.path.join(output_dir, "results")
    _ensure_dir(plots_dir)
    _ensure_dir(results_dir)

    config_suffix = f"N{num_nodes}_K{average_degree}_RW{random_walk_steps}_REPEATS{eval_repeats}_seed{seed}"
    results_path = os.path.join(results_dir, f"expB_results_{config_suffix}.json")
    plot_file_1 = os.path.join(plots_dir, f"expB_modularity_vs_num_concepts_{config_suffix}.pdf")
    plot_file_2 = os.path.join(plots_dir, f"expB_modularity_vs_diameter_{config_suffix}.pdf")

    # Resume if possible
    cached = load_progress(results_path) if resume else None
    if cached is not None and isinstance(cached, dict) and "results" in cached:
        report_df = pd.DataFrame(cached["results"])
        output_data = cached
    else:
        start_time = time.time()

        file_pattern = os.path.join(root_dir, "data", "saved_cog_nets", f"G_n_{num_nodes}_k_{average_degree}_*.edgelist.gz")
        # If you generated a separate dataset, you can switch to it here:
        # file_pattern = os.path.join(root_dir, "data", "new_saved_cog_nets", f"G_n_{num_nodes}_k_{average_degree}_*.edgelist.gz")
        rewired_graphs = UtilityFunctions.load_graphs(file_pattern)
        if not rewired_graphs:
            raise FileNotFoundError(f"No graphs found matching the pattern: {file_pattern}")

        params_list = [{"rewired_graphs": rewired_graphs, "random_walk_steps": random_walk_steps}] * eval_repeats
        num_processes = multiprocessing.cpu_count()

        results: list[dict] = []
        with multiprocessing.Pool(processes=num_processes) as pool:
            for batch in tqdm(pool.imap_unordered(evaluation_helper, params_list), total=len(params_list)):
                results.extend(batch)

        report_df = pd.DataFrame(results)
        report_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        report_df.fillna(report_df.median(numeric_only=True), inplace=True)
        report_df = report_df.groupby('Index', as_index=False).mean(numeric_only=True)

        elapsed_time = time.time() - start_time

        output_data = {
            "experiment": "experimentB",
            "config": {
                "random_walk_steps": random_walk_steps,
                "eval_repeats": eval_repeats,
                "N": num_nodes,
                "K": average_degree,
                "seed": seed,
                "machine_id": machine_id,
            },
            "results_path": results_path,
            "time_taken": elapsed_time,
            "results": report_df.to_dict(orient="records"),
        }

        if save_json:
            save_progress(output_data, results_path)

    plot_paths: list[str] = []
    if make_plot:
        # Correlation analysis is informational; keep it in the returned dict
        corr1 = UtilityFunctions.pearson_correlation_with_p_value(
            report_df, 'Modularity', 'Num Concepts Accessed'
        )
        corr2 = UtilityFunctions.pearson_correlation_with_p_value(
            report_df, 'Modularity', 'Diameter'
        )
        output_data["correlations"] = {
            "Modularity_vs_NumConcepts": {
                "pearson_r": corr1[0],
                "p_value": corr1[1],
                "ci_lower": corr1[2],
                "ci_upper": corr1[3],
            },
            "Modularity_vs_Diameter": {
                "pearson_r": corr2[0],
                "p_value": corr2[1],
                "ci_lower": corr2[2],
                "ci_upper": corr2[3],
            },
        }

        plot_paths.append(_plot_regression(
            report_df,
            x='Modularity',
            y='Num Concepts Accessed',
            out_path=plot_file_1,
            scatter_alpha=0.3,
        ))
        plot_paths.append(_plot_regression(
            report_df,
            x='Modularity',
            y='Diameter',
            out_path=plot_file_2,
            scatter_alpha=0.1,
        ))

        if save_json:
            save_progress(output_data, results_path)

    return output_data, plot_paths


# Preferred usage (from repo root):
#   python -m experiments.experimentB.experimentB
# Direct execution also works via the import fallback above.
def main():
    args = parse_arguments()
    results, plot_paths = run_experimentB(
        num_nodes=args.num_nodes,
        average_degree=args.average_degree,
        seed=args.seed,
        random_walk_steps=args.random_walk_steps,
        eval_repeats=args.eval_repeats,
        output_dir=args.output_dir,
        resume=not args.no_resume,
        make_plot=not args.no_plot,
        save_json=True,
    )

    print("Done.")
    print(f"Results JSON: {results.get('results_path')}")
    if plot_paths:
        for p in plot_paths:
            print(f"Plot: {p}")


if __name__ == '__main__':
    main()
