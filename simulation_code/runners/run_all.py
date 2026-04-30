"""Run all experiments from one place.

Usage (from repo root):
  python -m runners.run_all

Direct execution also works:
  python runners/run_all.py

This file is designed to be the single entry point that runs the main experiments (mechanism + applications).
A/B live under `experiments/mechanism/` and E/F live under `experiments/applications/`.
Parameter validation experiments (C, D) are separate and not run by default.
Edit the PARAMETERS section below once, and it will apply to all experiments.

Each experiment is expected to expose a function named `run_experimentX(...)` that
returns: (results_dict, plot_path_or_paths)

- plot_path_or_paths may be a string (single plot), a list/tuple of strings (multiple plots), or None.
"""


from __future__ import annotations

import sys
import os
# Ensure repo root is on sys.path so `import experiments...` works when running:
#   python runners/run_all.py
# as well as:
#   python -m runners.run_all
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import inspect
import json
import time
from dataclasses import dataclass, asdict
from typing import Any

# =====================
# PARAMETERS (edit once)
# =====================

# Global/shared parameters
SEED: int = 42
OUTPUT_ROOT: str = os.path.join(REPO_ROOT, "outputs")
RESUME: bool = True
SAVE_JSON: bool = True
MAKE_PLOTS: bool = True

# Paper-facing names (used for output folders and summary keys)
EXP_SLUGS: dict[str, str] = {
    "A": "modularity_control",
    "B": "modularity_to_creativity",
    "C": "inspiration_sensitivity",
    "D": "baseline_sampling",
    "E": "social_inspiration",
    "F": "creativity_landscape",
}

EXP_TITLES: dict[str, str] = {
    "A": "Modularity control",
    "B": "Modularity → creativity",
    "C": "Inspiration sensitivity (RW length)",
    "D": "Baseline sampling (RW length)",
    "E": "Social inspiration",
    "F": "Creativity landscape",
}

# Default graph/model parameters (used by experiments that accept these)
NUM_NODES: int = 100
AVERAGE_DEGREE: int = 4


NUM_GRAPHS_PER_P: int = 15 # only needed in exp A

# If True, also run parameter-validation experiments (C and D).
RUN_PARAMETER_VALIDATION: bool = False

# Per-experiment optional overrides (set any of these to override the shared values)
# Example:
#   EXP_OVERRIDES = {"A": {"num_nodes": 300, "average_degree": 10}}
EXP_OVERRIDES: dict[str, dict[str, Any]] = {}


@dataclass
class ExperimentRunSummary:
    slug: str
    legacy_key: str
    status: str  # "ok" | "skipped" | "error"
    output_dir: str
    plot_path: Any = None
    results_path: str | None = None
    time_taken_sec: float | None = None
    error: str | None = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, data: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _base_kwargs_for_all(output_dir: str) -> dict[str, Any]:
    """Shared kwargs passed to experiments when supported."""
    return {
        "seed": SEED,
        "output_dir": output_dir,
        "resume": RESUME,
        "save_json": SAVE_JSON,
        "make_plot": MAKE_PLOTS,
    }


def _shared_graph_kwargs() -> dict[str, Any]:
    return {
        "num_nodes": NUM_NODES,
        "average_degree": AVERAGE_DEGREE,
        "num_graphs_per_p": NUM_GRAPHS_PER_P,
    }


def _apply_overrides(exp_key: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    overrides = EXP_OVERRIDES.get(exp_key, {})
    if overrides:
        out = dict(kwargs)
        out.update(overrides)
        return out
    return kwargs


def _call_with_supported_kwargs(func, kwargs: dict[str, Any]):
    """Call func with only the kwargs it accepts (avoids breaking experiments with different signatures)."""
    sig = inspect.signature(func)
    supported = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return func(**supported)


def _run_one(exp_key: str) -> tuple[ExperimentRunSummary, dict | None]:
    """Run one experiment by key (A–F). Returns (summary, results_dict_or_none)."""
    slug = EXP_SLUGS.get(exp_key, f"experiment{exp_key}")
    output_dir = os.path.join(OUTPUT_ROOT, slug)
    _ensure_dir(output_dir)

    started = time.time()

    try:
        if exp_key == "A":
            # Mechanism: Modularity control (formerly experimentA)
            from experiments.mechanism.modularity_control import run_experimentA as runner
        elif exp_key == "B":
            # Mechanism: Modularity -> creativity (formerly experimentB)
            from experiments.mechanism.modularity_to_creativity import run_experimentB as runner
        elif exp_key == "C":
            if not RUN_PARAMETER_VALIDATION:
                raise ImportError("Parameter validation is disabled (set RUN_PARAMETER_VALIDATION=True to run C/D)")
            from experiments.parameter_validation.inspiration_sensitivity_C import run_experimentC as runner
        elif exp_key == "D":
            if not RUN_PARAMETER_VALIDATION:
                raise ImportError("Parameter validation is disabled (set RUN_PARAMETER_VALIDATION=True to run C/D)")
            from experiments.parameter_validation.baseline_sampling_D import run_experimentD as runner
        elif exp_key == "E":
            # Applications: Social inspiration via overlap (formerly experimentE)
            from experiments.applications.social_inspiration import run_experimentE as runner
        elif exp_key == "F":
            # Applications: Shared vs separate sources (formerly experimentF)
            from experiments.applications.creativity_landscape import run_experimentF as runner
        else:
            raise ValueError(f"Unknown experiment key: {exp_key}")

        kwargs = {
            **_shared_graph_kwargs(),
            **_base_kwargs_for_all(output_dir),
        }
        kwargs = _apply_overrides(exp_key, kwargs)

        results, plot_path = _call_with_supported_kwargs(runner, kwargs)

        elapsed = time.time() - started

        results_path = None
        if isinstance(results, dict):
            results_path = results.get("results_path") or results.get("results_filename")

        summary = ExperimentRunSummary(
            slug=slug,
            legacy_key=exp_key,
            status="ok",
            output_dir=output_dir,
            plot_path=plot_path,
            results_path=results_path,
            time_taken_sec=float(results.get("time_taken", elapsed)) if isinstance(results, dict) else float(elapsed),
        )
        return summary, results

    except (ImportError, ModuleNotFoundError) as e:
        elapsed = time.time() - started
        summary = ExperimentRunSummary(
            slug=slug,
            legacy_key=exp_key,
            status="skipped",
            output_dir=output_dir,
            time_taken_sec=float(elapsed),
            error=f"ImportError: {e}",
        )
        return summary, None

    except Exception as e:
        elapsed = time.time() - started
        summary = ExperimentRunSummary(
            slug=slug,
            legacy_key=exp_key,
            status="error",
            output_dir=output_dir,
            time_taken_sec=float(elapsed),
            error=str(e),
        )
        return summary, None


def run_all() -> dict[str, Any]:
    """Run experiments A–F and write a combined summary JSON."""
    _ensure_dir(OUTPUT_ROOT)

    all_started = time.time()

    summaries: list[ExperimentRunSummary] = []
    results_by_exp: dict[str, Any] = {}

    exp_keys = ["A", "B", "E", "F"]
    if RUN_PARAMETER_VALIDATION:
        exp_keys = ["A", "B", "C", "D", "E", "F"]

    for key in exp_keys:
        title = EXP_TITLES.get(key, key)
        print(f"\n=== Running {title} (legacy {key}) ===")
        summary, results = _run_one(key)
        summaries.append(summary)

        if summary.status == "ok":
            print(f"[OK] {title} (legacy {key}) finished in {summary.time_taken_sec:.2f}s")
            if summary.results_path:
                print(f"     Results: {summary.results_path}")
            if summary.plot_path:
                if isinstance(summary.plot_path, (list, tuple)):
                    for p in summary.plot_path:
                        print(f"     Plot:    {p}")
                else:
                    print(f"     Plot:    {summary.plot_path}")
            results_by_exp[summary.slug] = results
        elif summary.status == "skipped":
            print(f"[SKIP] {title} (legacy {key}) not available: {summary.error}")
        else:
            print(f"[ERR] {title} (legacy {key}) failed: {summary.error}")

    total_elapsed = time.time() - all_started

    payload = {
        "runner": "runners/run_all.py",
        "output_root": OUTPUT_ROOT,
        "parameters": {
            "seed": SEED,
            "resume": RESUME,
            "save_json": SAVE_JSON,
            "make_plots": MAKE_PLOTS,
            "run_parameter_validation": RUN_PARAMETER_VALIDATION,
            "num_nodes": NUM_NODES,
            "average_degree": AVERAGE_DEGREE,
            "num_graphs_per_p": NUM_GRAPHS_PER_P,
            "exp_overrides": EXP_OVERRIDES,
        },
        "total_time_sec": total_elapsed,
        "experiments": [asdict(s) for s in summaries],
        "results": results_by_exp,
    }

    summary_path = os.path.join(OUTPUT_ROOT, "all_experiments_summary.json")
    _write_json(summary_path, payload)

    print("\n=== All experiments finished ===")
    print(f"Summary JSON: {summary_path}")
    print(f"Total time: {total_elapsed:.2f}s")

    for s in summaries:
        print(f"  - {s.slug} (legacy {s.legacy_key}): {s.status}")

    return payload


# done 

if __name__ == "__main__":
    run_all()
    