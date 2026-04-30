

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Robustness runner

Reads a declarative plan (docs/robustness_plan.yaml), runs specified robustness sweeps,
and writes a machine-readable robustness table.

Typical usage (from repo root):
  python -m runners.robustness --plan docs/robustness_plan.yaml

Outputs (by default):
  <output_root>/robustness_table.csv
  <output_root>/robustness_table.jsonl
  <output_root>/runs/<exp>/... (experiment-native outputs)

Notes
-----
- This runner calls the *importable* experiment APIs:
    Experiment E: experiments.applications.social_inspiration.run_experimentE
    Experiment F: experiments.applications.creativity_landscape.run_experimentF
- Experiment E's main effect summary is computed from its SQLite DB:
    slope( overlap -> #new concepts ) with cluster bootstrap CI (cluster=pair_index).
- Experiment F's main effect summary is read from its monolithic JSON(.gz) output:
    mean diff CI and Cohen's dz + paired t p-value.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import json
import os
import sys
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.stats import linregress

# ---------- repo-root bootstrap ----------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ---------- experiment imports ----------
from experiments.applications.social_inspiration import run_experimentE, bootstrap_slope_cluster  # type: ignore
from experiments.applications.creativity_landscape import run_experimentF  # type: ignore


# ---------- YAML loading ----------
def _load_plan(path: str) -> dict:
    """
    Load YAML plan. Requires PyYAML.

    We intentionally keep YAML as the plan format so reviewers can edit without code changes.
    """
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: PyYAML is required to read robustness_plan.yaml.\n"
            "Install with: pip install pyyaml"
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Plan must be a YAML mapping/dict. Got: {type(data)}")
    return data


# ---------- small helpers ----------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _jsonl_append(path: str, row: dict) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: str, rows: List[dict]) -> None:
    _ensure_dir(os.path.dirname(path))
    if not rows:
        # write header-less empty file for clarity
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    # stable column ordering: union of keys, with common fields first
    preferred = [
        "timestamp",
        "check",
        "experiment",
        "N",
        "K",
        "seed",
        "random_walk_steps",
        "iterations",
        "starting_nodes",
        "start_nodes_sample",
        "pair_set_seed",
        "num_pairs",
        "pair_offset",
        "outcome",
        "effect_type",
        "direction",
        "estimate",
        "ci_low",
        "ci_high",
        "p_value",
        "n_effective",
        "results_path",
        "db_path",
        "config_dir",
    ]
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    rest = sorted([k for k in keys if k not in preferred])
    fieldnames = [k for k in preferred if k in keys] + rest

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _direction(x: float) -> str:
    if not np.isfinite(x):
        return "NA"
    return "+" if x > 0 else ("-" if x < 0 else "0")


# ---------- plan expansion ----------
def _expand_check(
    *,
    defaults: dict,
    grids: dict,
    check: dict,
) -> Iterable[Tuple[dict, dict, dict]]:
    """
    Yield (params, vary_vals, fixed_vals) for each configuration implied by the check.
    """
    if "experiment" not in check:
        raise ValueError(f"Check missing 'experiment': {check}")
    vary = check.get("vary", []) or []
    fixed = check.get("fixed", {}) or {}

    # Build list of values for each vary key from grids (or from fixed if explicit list was provided there).
    vary_keys = list(vary)
    vary_values = []
    for k in vary_keys:
        if k in fixed and isinstance(fixed[k], list):
            vals = fixed[k]
        else:
            vals = grids.get(k)
        if not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"Check '{check.get('name')}' varies '{k}' but no grid values provided.")
        vary_values.append(vals)

    if not vary_keys:
        # Single config
        params = dict(defaults)
        params.update(fixed)
        yield params, {}, fixed
        return

    for combo in product(*vary_values):
        vary_vals = dict(zip(vary_keys, combo))
        params = dict(defaults)
        params.update(fixed)
        params.update(vary_vals)
        yield params, vary_vals, fixed


# ---------- Experiment E summarization ----------
def _summarize_experimentE_from_db(db_path: str) -> Tuple[float, float, float, int]:
    """
    Compute slope(overlap -> # new concepts) with cluster bootstrap CI.

    Returns:
      (slope_point_estimate, ci_low, ci_high, n_pairs)
    """
    import sqlite3

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB not found: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT pair_index, overlap_vals, unique_concepts_gained FROM pair_results")
    rows = cur.fetchall()
    conn.close()

    xs: List[float] = []
    ys: List[float] = []
    groups: List[str] = []

    for pair_index, ov_json, uniq_json in rows:
        try:
            ov_list = json.loads(ov_json) if ov_json else []
            uq_list = json.loads(uniq_json) if uniq_json else []
        except Exception:
            continue
        L = min(len(ov_list), len(uq_list))
        if L <= 0:
            continue
        pid = str(pair_index)
        for i in range(L):
            ov = float(ov_list[i])
            uq = float(uq_list[i])
            if np.isfinite(ov) and np.isfinite(uq):
                xs.append(ov)
                ys.append(uq)
                groups.append(pid)

    if len(xs) < 3:
        return float("nan"), float("nan"), float("nan"), 0

    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    # point estimate: simple OLS slope
    lr = linregress(x, y)
    slope = float(lr.slope)

    # CI: cluster bootstrap by pair_index (reuses experiment helper)
    mean_slope, (lo, hi) = bootstrap_slope_cluster(x, y, groups=groups, B=1000, seed=0)

    # number of unique clusters (pairs) is the effective group count
    n_pairs = len(set(groups))
    # Prefer the bootstrap mean if available; but keep point estimate as the reported estimate.
    _ = mean_slope
    return slope, float(lo), float(hi), int(n_pairs)


# ---------- Experiment F summarization ----------
def _load_json_maybe_gz(path: str) -> dict:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summarize_experimentF_from_results_path(results_path: str) -> Tuple[float, float, float, float, int]:
    """
    Returns:
      (mean_diff, ci_low, ci_high, cohen_dz, n_pairs)
    """
    data = _load_json_maybe_gz(results_path)
    eff = data.get("stats", {}).get("effect_size", {})
    ci = eff.get("mean_diff_ci95", [float("nan"), float("nan")])
    dz = eff.get("cohen_dz", float("nan"))

    # mean diff is the midpoint of CI if present, otherwise compute from arrays
    try:
        ci_low = float(ci[0])
        ci_high = float(ci[1])
        mean_diff = float((ci_low + ci_high) / 2.0)
    except Exception:
        ci_low = ci_high = float("nan")
        control = np.asarray(data.get("results", {}).get("control_means", []), float)
        triad = np.asarray(data.get("results", {}).get("triad_means", []), float)
        mean_diff = float(np.nanmean(triad - control)) if control.size else float("nan")

    n_pairs = int(data.get("config", {}).get("valid_pairs", 0)) or int(
        len(data.get("results", {}).get("control_means", []))
    )
    return mean_diff, ci_low, ci_high, float(dz), n_pairs


# ---------- main execution ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run robustness sweeps from a YAML plan and write a summary table.")
    ap.add_argument("--plan", type=str, required=True, help="Path to robustness_plan.yaml")
    ap.add_argument(
        "--checks",
        nargs="*",
        default=None,
        help="Optional list of check names to run (must match check.name). If omitted, runs all checks.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print planned runs, do not execute.")
    ap.add_argument("--make-plots", action="store_true", help="Allow experiments to generate plots (slower).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    plan = _load_plan(args.plan)

    meta = plan.get("meta", {}) or {}
    output_root = meta.get("output_root", "outputs/robustness")
    output_root = os.path.abspath(output_root)
    _ensure_dir(output_root)

    defaults = plan.get("defaults", {}) or {}
    grids = plan.get("grids", {}) or {}
    checks = plan.get("checks", []) or []
    if not isinstance(checks, list) or not checks:
        raise ValueError("Plan must contain a non-empty 'checks' list.")

    # Force these defaults for robustness runs (keeps runs fast + reviewer-friendly).
    defaults = dict(defaults)
    defaults.setdefault("resume", True)
    defaults.setdefault("save_json", True)
    defaults.setdefault("make_plot", False)

    if args.make_plots:
        defaults["make_plot"] = True

    # output dirs per experiment, under output_root/runs/<exp>
    runs_root = os.path.join(output_root, "runs")
    _ensure_dir(runs_root)

    rows_out: List[dict] = []
    jsonl_path = os.path.join(output_root, "robustness_table.jsonl")

    for chk in checks:
        name = chk.get("name", "(unnamed)")
        if args.checks and name not in set(args.checks):
            continue

        exp = chk.get("experiment")
        if exp not in ("E", "F"):
            raise ValueError(f"Unsupported experiment '{exp}' in check '{name}'. Expected 'E' or 'F'.")

        for params, vary_vals, fixed_vals in _expand_check(defaults=defaults, grids=grids, check=chk):
            # Mandatory shared params
            if "num_nodes" not in params or "average_degree" not in params:
                raise ValueError(
                    f"Check '{name}' must provide num_nodes and average_degree via defaults/fixed."
                )

            # Route outputs under output_root/runs/<experiment>/
            exp_dir = os.path.join(runs_root, f"experiment{exp}")
            _ensure_dir(exp_dir)
            params = dict(params)
            params["output_dir"] = exp_dir

            # Human-readable label for this run (for table + logging)
            label_bits = []
            for k in chk.get("vary", []) or []:
                label_bits.append(f"{k}={params.get(k)}")
            label = ", ".join(label_bits) if label_bits else "default"

            if args.dry_run:
                print(f"[DRY] {name} | exp={exp} | {label}")
                continue

            ts = _dt.datetime.now().isoformat(timespec="seconds")
            common_row = {
                "timestamp": ts,
                "check": name,
                "experiment": exp,
                "N": int(params["num_nodes"]),
                "K": int(params["average_degree"]),
                "seed": int(params.get("seed", 42)),
            }

            if exp == "E":
                # Execute
                results, _plots = run_experimentE(
                    num_nodes=int(params["num_nodes"]),
                    average_degree=int(params["average_degree"]),
                    seed=int(params.get("seed", 42)),
                    random_walk_steps=int(params.get("random_walk_steps", 20)),
                    iterations=int(params.get("iterations", 10)),
                    starting_nodes=int(params.get("starting_nodes", 10)),
                    pair_set_seed=int(params.get("pair_set_seed", 42)),
                    num_pairs=int(params.get("num_pairs", 500)),
                    pair_offset=int(params.get("pair_offset", 0)),
                    output_dir=exp_dir,
                    resume=bool(params.get("resume", True)),
                    save_json=bool(params.get("save_json", True)),
                    make_plot=bool(params.get("make_plot", False)),
                    plot_out=params.get("plot_out", None),
                )

                slope, lo, hi, n_pairs = _summarize_experimentE_from_db(results["db_path"])

                row = {
                    **common_row,
                    "random_walk_steps": int(params.get("random_walk_steps", 20)),
                    "iterations": int(params.get("iterations", 10)),
                    "starting_nodes": int(params.get("starting_nodes", 10)),
                    "pair_set_seed": int(params.get("pair_set_seed", 42)),
                    "num_pairs": int(params.get("num_pairs", 500)),
                    "pair_offset": int(params.get("pair_offset", 0)),
                    "outcome": "unique_concepts_gained",
                    "effect_type": "slope(overlap -> new_concepts)",
                    "direction": _direction(slope),
                    "estimate": float(slope),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p_value": float("nan"),
                    "n_effective": int(n_pairs),
                    "db_path": results.get("db_path"),
                    "config_dir": results.get("config_dir"),
                }

            else:  # exp == "F"
                results, _plots = run_experimentF(
                    num_nodes=int(params["num_nodes"]),
                    average_degree=int(params["average_degree"]),
                    seed=int(params.get("seed", 42)),
                    random_walk_steps=int(params.get("random_walk_steps", 20)),
                    iterations=int(params.get("iterations", 5)),
                    start_nodes_sample=int(params.get("start_nodes_sample", 10)),
                    output_dir=exp_dir,
                    resume=bool(params.get("resume", True)),
                    save_json=bool(params.get("save_json", True)),
                    make_plot=bool(params.get("make_plot", False)),
                )

                # Summarize from the monolithic JSON(.gz)
                mean_diff, lo, hi, dz, n_pairs = _summarize_experimentF_from_results_path(results["results_path"])

                # p-value is in the payload JSON, but we can also read it here
                payload = _load_json_maybe_gz(results["results_path"])
                p_val = payload.get("stats", {}).get("paired_t", {}).get("p", float("nan"))

                row = {
                    **common_row,
                    "random_walk_steps": int(params.get("random_walk_steps", 20)),
                    "iterations": int(params.get("iterations", 5)),
                    "start_nodes_sample": int(params.get("start_nodes_sample", 10)),
                    "outcome": "overlap(triad-control)",
                    "effect_type": "mean_diff(triad - control)",
                    "direction": _direction(mean_diff),
                    "estimate": float(mean_diff),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "p_value": float(p_val) if p_val is not None else float("nan"),
                    "n_effective": int(n_pairs),
                    "results_path": results.get("results_path"),
                }
                # include effect size explicitly (nice for paper)
                row["cohen_dz"] = float(dz)

            rows_out.append(row)
            _jsonl_append(jsonl_path, row)

            # Compact console log
            print(
                f"[OK] {name} | exp={exp} | {label} | estimate={row.get('estimate'):.4g} "
                f"CI=[{row.get('ci_low'):.4g},{row.get('ci_high'):.4g}]"
            )

    # Write CSV from all rows we accumulated this run.
    csv_path = os.path.join(output_root, "robustness_table.csv")
    _write_csv(csv_path, rows_out)
    print(f"\nWrote robustness table:")
    print(f"  CSV : {csv_path}")
    print(f"  JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()