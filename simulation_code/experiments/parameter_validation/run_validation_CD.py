"""Experiment CD: Random-walk parameter validation.

This wrapper runs two complementary validations for random-walk length:

Side D (Baseline Sampling):
  - Measures how random-walk sampling alone (no inspiration) affects observed subgraph properties.

Side C (Inspiration Sensitivity):
  - Measures how inspiration/integration changes sampled subgraph properties (after–before deltas)
    as a function of random-walk length.

Outputs are written under:
  outputs/parameter_validation/
    baseline_D/...
    inspiration_C/...
  plus a manifest.json describing exactly what was run.
"""

from __future__ import annotations

import os
import argparse
import json
from datetime import datetime

from experiments.parameter_validation.baseline_sampling_D import run_experimentD
from experiments.parameter_validation.inspiration_sensitivity_C import run_experimentC

def parse_args():
    parser = argparse.ArgumentParser(description="Run combined parameter validation (Experiments D + C).")

    parser.add_argument("--seed", type=int, default=42, help="Global seed (passed into C and D).")
    parser.add_argument("--iterations", type=int, default=5, help="Iterations per starting node.")
    parser.add_argument("--starting_nodes", type=int, default=5, help="Starting nodes per graph/pair.")

    parser.add_argument(
        "--rw_list",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 30, 40],
        help="Random walk lengths to evaluate.",
    )

    parser.add_argument(
        "--run_baseline",
        action="store_true",
        help="If set, run baseline sampling (Experiment D).",
    )
    parser.add_argument(
        "--run_inspiration",
        action="store_true",
        help="If set, run inspiration sensitivity (Experiment C).",
    )

    # Graph config (kept explicit to avoid hidden assumptions)
    parser.add_argument("--num_nodes", type=int, default=100, help="N for the saved cognitive graphs.")
    parser.add_argument("--average_degree", type=int, default=4, help="k for the saved cognitive graphs.")

    return parser.parse_args()


def main():
    args = parse_args()

    # If neither flag is provided, run both sides by default.
    run_baseline = bool(args.run_baseline)
    run_inspiration = bool(args.run_inspiration)
    if not run_baseline and not run_inspiration:
        run_baseline = True
        run_inspiration = True

    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_root = os.path.join(REPO_ROOT, "outputs", "parameter_validation")

    baseline_dir = os.path.join(results_root, "baseline_D")
    inspiration_dir = os.path.join(results_root, "inspiration_C")

    os.makedirs(baseline_dir, exist_ok=True)
    os.makedirs(inspiration_dir, exist_ok=True)

    manifest = {
        "experiment": "Experiment CD: Random-walk parameter validation",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": args.seed,
        "iterations": args.iterations,
        "starting_nodes": args.starting_nodes,
        "random_walk_lengths": args.rw_list,
        "graph_config": {"N": args.num_nodes, "K": args.average_degree},
        "sides": {
            "baseline_D": {"enabled": run_baseline, "results": []},
            "inspiration_C": {"enabled": run_inspiration, "results": []},
        },
    }

    for rw in args.rw_list:
        print(f"\n=== Random walk length = {rw} ===")

        if run_baseline:
            print("Running baseline sampling (D)...")
            results_D, plots_D = run_experimentD(
                num_nodes=args.num_nodes,
                average_degree=args.average_degree,
                seed=args.seed,
                random_walk_steps=rw,
                iterations=args.iterations,
                starting_nodes=args.starting_nodes,
                output_dir=baseline_dir,
                resume=True,
                make_plot=True,
                plot_metric="all",
                plot_style="box",
            )

            manifest["sides"]["baseline_D"]["results"].append(
                {
                    "random_walk_steps": rw,
                    "config_results_dir": results_D.get("config_results_dir"),
                    "db_path": results_D.get("db_path"),
                    "plot_paths": plots_D,
                    "did_compute": results_D.get("did_compute"),
                }
            )

        if run_inspiration:
            print("Running inspiration sensitivity (C)...")
            results_C, plots_C = run_experimentC(
                num_nodes=args.num_nodes,
                average_degree=args.average_degree,
                seed=args.seed,
                random_walk_steps=rw,
                iterations=args.iterations,
                starting_nodes=args.starting_nodes,
                output_dir=inspiration_dir,
                resume=True,
                make_plot=True,
                plot_metric="all",
                plot_style="box",
            )

            manifest["sides"]["inspiration_C"]["results"].append(
                {
                    "random_walk_steps": rw,
                    "config_results_dir": results_C.get("config_results_dir"),
                    "db_path": results_C.get("db_path"),
                    "plot_paths": plots_C,
                    "did_compute": results_C.get("did_compute"),
                }
            )

    manifest_path = os.path.join(results_root, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
