#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Experiment F — Shared vs Separate Inspiration Sources (Triad vs Control)

Goal:
- Show that two low-creativity (high-modularity) networks become more similar when
  they both take inspiration from the SAME high-creativity (low-modularity) source,
  compared to inspiration from DIFFERENT sources.

What we do:
- Bin 500 graphs by modularity (ascending):
    * bottom 20%  (low modularity)  -> "high creativity"  -> SOURCES
    * top   20%   (high modularity) -> "low creativity"   -> TARGETS
- For every combo:
    Control:  S1 → T1,  S2 → T2
    Triad:    S1 → T1,  S1 → T2   (shared source; two independent walks)
- For each combo, sample up to K common start nodes, run N random walks per start,
  induce subgraphs on targets, and measure node-Jaccard overlap. Average to 1 value
  per condition. Do a paired t-test over matched pairs.

CLI (kept small):
  run      -> do everything in one go, write a single JSON(.gz)
  shard    -> do a disjoint subset, write JSONL shard to result/shards/<RUN_ID>/
  reduce   -> combine shards for a given RUN_ID, compute stats, optional plot
  plot     -> plot a run JSON (from `run`) or a reduce summary

Defaults:
  random_walk_steps=20, iterations=5, start_nodes_sample=10, seed=42

Outputs:
- Monolithic run: experiments/experimentF/result/experimentF_results.json.gz
- Shards:        experiments/experimentF/result/shards/<RUN_ID>/*.jsonl(.gz)
- Reduce summary: experiments/experimentF/result/experimentF_summary.json
- Plots:         experiments/experimentF/plots/
"""


from __future__ import annotations

# ----------------- imports & repo-root bootstrap -----------------

import os, sys, json, time, math, gzip, glob, random, argparse, itertools, hashlib
from typing import List, Tuple, Optional, Iterator


# Prefer package-style imports. If executed directly, fall back to adding
# the project root to sys.path.
try:
    from core.utils import UtilityFunctions
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
except ModuleNotFoundError:
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    from core.utils import UtilityFunctions

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, wilcoxon


# --- project-local output dirs (live beside this script) ---
BASE_DIR   = os.path.dirname(__file__)
RESULT_DIR = os.path.join(BASE_DIR, "result")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)




# ----------------- simple constants -----------------

TOP_PCT = 0.20   # top 20% modularity -> targets (low creativity)
BOT_PCT = 0.20   # bottom 20% modularity -> sources (high creativity)
SEED    = 42     # global run seed


# ----------------- tiny helpers -------------------

def stable_seed(*parts: int) -> int:
    """Deterministic 32-bit seed from ints."""
    h = hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:8], 16)

def choose_one(candidates: List[int], *key: int) -> int:
    rng = random.Random(stable_seed(*key))
    return rng.choice(candidates)

def sample_nodes(nodes, k: int, *key: int):
    nodes = list(nodes)
    if len(nodes) <= k: return nodes
    rng = random.Random(stable_seed(*key))
    return rng.sample(nodes, k)

def cohen_dz(control: np.ndarray, triad: np.ndarray) -> float:
    diff = triad - control
    sd = diff.std(ddof=1)
    return 0.0 if sd == 0 else diff.mean() / sd


# ----------------- worker globals (intentionally simple) -----------------

SOURCE_GRAPHS: List = []
TARGET_GRAPHS: List = []

def _worker_init(source_graphs, target_graphs):
    global SOURCE_GRAPHS, TARGET_GRAPHS
    SOURCE_GRAPHS = source_graphs
    TARGET_GRAPHS = target_graphs


# ----------------- core worker -----------------

def process_combo(args) -> Optional[Tuple[int, int, float, float]]:
    """
    One combo:
      s1_idx over sources
      (t1_idx, t2_idx) over targets
      s2_idx over sources (s2 != s1)
    Returns: (s1_idx, target_pair_idx, control_mean, triad_mean) or None if no shared starts.
    """
    (s1_idx, tp_idx, t1_idx, t2_idx, s2_idx,
     walk_steps, iters, K, seed) = args

    s1 = SOURCE_GRAPHS[s1_idx]
    s2 = SOURCE_GRAPHS[s2_idx]
    t1 = TARGET_GRAPHS[t1_idx]
    t2 = TARGET_GRAPHS[t2_idx]

    common = set(t1.nodes) & set(t2.nodes) & set(s1.nodes) & set(s2.nodes)
    if not common:
        return None

    starts = sample_nodes(common, K, seed, 1001, s1_idx, tp_idx, t1_idx, t2_idx, s2_idx)

    control_vals, triad_vals = [], []
    base = stable_seed(seed, 2003, s1_idx, tp_idx, t1_idx, t2_idx, s2_idx)

    for k, start in enumerate(starts):
        for it in range(iters):
            # CONTROL: S1→T1, S2→T2
            UtilityFunctions.set_global_seed(stable_seed(base, 1, k, it))
            p_s1 = UtilityFunctions.random_walk_path(s1, walk_steps, start)
            UtilityFunctions.set_global_seed(stable_seed(base, 2, k, it))
            p_s2 = UtilityFunctions.random_walk_path(s2, walk_steps, start)
            c1 = UtilityFunctions.subgraph_from_path(t1, p_s1)
            c2 = UtilityFunctions.subgraph_from_path(t2, p_s2)
            control_vals.append(UtilityFunctions.compute_overlap(c1, c2))

            # TRIAD: S1→T1 and S1→T2 (independent walks on same source)
            UtilityFunctions.set_global_seed(stable_seed(base, 3, k, it))
            p_t1 = UtilityFunctions.random_walk_path(s1, walk_steps, start)
            UtilityFunctions.set_global_seed(stable_seed(base, 4, k, it))
            p_t2 = UtilityFunctions.random_walk_path(s1, walk_steps, start)
            t1g = UtilityFunctions.subgraph_from_path(t1, p_t1)
            t2g = UtilityFunctions.subgraph_from_path(t2, p_t2)
            triad_vals.append(UtilityFunctions.compute_overlap(t1g, t2g))

    return (s1_idx, tp_idx, float(np.mean(control_vals)), float(np.mean(triad_vals)))


# ----------------- shared prep -----------------


def load_binned_graphs(num_nodes: int = 100, average_degree: int = 4, seed: int = 42):
    """Load 500 graphs, compute modularity, split into sources (low modularity) and targets (high modularity)."""
    UtilityFunctions.set_global_seed(seed)

    file_pattern = os.path.join(
        root_dir,
        "data",
        "saved_cog_nets",
        f"G_n_{num_nodes}_k_{average_degree}_*.edgelist.gz",
    )
    graphs = UtilityFunctions.load_graphs(file_pattern)
    if len(graphs) != 500:
        raise ValueError(f"Expected 500 graphs, found {len(graphs)} (pattern: {file_pattern})")

    vals = [(i, UtilityFunctions.compute_modularity(g)) for i, g in enumerate(graphs)]
    vals.sort(key=lambda x: x[1])  # ascending by modularity

    n = len(vals)
    nbin = int(BOT_PCT * n)

    src_idx = [i for (i, _) in vals[:nbin]]
    tgt_idx = [i for (i, _) in vals[-nbin:]]

    sources = [graphs[i] for i in src_idx]
    targets = [graphs[i] for i in tgt_idx]
    return sources, targets


def target_pairs(targets) -> List[Tuple[int, int]]:
    """All target pairs (100C2 = 4950)."""
    return list(itertools.combinations(range(len(targets)), 2))



def run_id(walk_steps: int, iters: int, K: int, total_shards: int, seed: int = 42) -> str:
    """Deterministic id for shard folder naming."""
    return f"rw{walk_steps}_it{iters}_k{K}_S{total_shards}_seed{seed}"


# ----------------- modes -----------------


def run_all(walk_steps: int, iters: int, K: int, output: str, num_nodes: int = 100, average_degree: int = 4, seed: int = 42):
    """Monolithic run that processes all combos and writes one JSON(.gz)."""
    src, tgt = load_binned_graphs(num_nodes=num_nodes, average_degree=average_degree, seed=seed)
    pairs = target_pairs(tgt)

    def gen_tasks() -> Iterator[Tuple[int,int,int,int,int,int,int,int,int]]:
        for s1_idx in range(len(src)):
            for tp_idx, (t1_idx, t2_idx) in enumerate(pairs):
                cands = [i for i in range(len(src)) if i != s1_idx]
                s2_idx = choose_one(cands, seed, 777, s1_idx, tp_idx, t1_idx, t2_idx)
                yield (s1_idx, tp_idx, t1_idx, t2_idx, s2_idx,
                       walk_steps, iters, K, seed)

    t0 = time.time()
    with Pool(initializer=_worker_init, initargs=(src, tgt)) as pool:
        res = []
        for out in tqdm(pool.imap_unordered(process_combo, gen_tasks(), chunksize=64),
                        total=len(src) * len(pairs), desc="Combos"):
            if out is not None:
                res.append(out)

    control = np.array([x[2] for x in res], dtype=float)
    triad   = np.array([x[3] for x in res], dtype=float)
    n = control.size

    t_stat, p_val = ttest_rel(triad, control, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(triad, control, method="approx")
    except Exception:
        w_stat, w_p = (float("nan"), float("nan"))
    dz = cohen_dz(control, triad)
    diff = (triad - control)
    ci_half = 1.96 * diff.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    ci = [float(diff.mean() - ci_half), float(diff.mean() + ci_half)]

    payload = {
        "config": {
            "random_walk_steps": walk_steps,
            "iterations": iters,
            "start_nodes_sample": K,
            "seed": seed,
            "n_sources": len(src),
            "n_targets": len(tgt),
            "n_target_pairs": len(pairs),
            "valid_pairs": int(n)
        },
        "results": {
            "control_means": control.tolist(),
            "triad_means": triad.tolist()
        },
        "stats": {
            "paired_t": {"t": float(t_stat), "df": int(n-1), "p": float(p_val)},
            "wilcoxon": {"W": float(w_stat), "p": float(w_p)},
            "effect_size": {"cohen_dz": float(dz), "mean_diff_ci95": ci}
        },
        "time_taken_seconds": float(time.time() - t0)
    }

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    if output.endswith(".gz"):
        with gzip.open(output, "wt", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    else:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"\nDone. wrote {n} pairs to {output}")
    print(f"t={t_stat:.4f}, p={p_val:.3e} | W={w_stat:.1f}, p={w_p:.3e} | dz={dz:.3f}")
    return payload


def shard_one(walk_steps: int, iters: int, K: int, total_shards: int, shard_id: int, gzip_out: bool):
    """Process a disjoint subset and write a JSONL shard (resume-safe filename)."""
    assert 0 <= shard_id < total_shards
    src, tgt = load_binned_graphs()
    pairs = target_pairs(tgt)

    rid = run_id(walk_steps, iters, K, total_shards, seed=SEED)
    # out_dir = os.path.join("results", "shards", rid)
    out_dir = os.path.join(RESULT_DIR, "shards", rid)

    
    os.makedirs(out_dir, exist_ok=True)
    ext = ".jsonl.gz" if gzip_out else ".jsonl"
    shard_path = os.path.join(out_dir, f"{rid}_shard{shard_id}{ext}")

    def shard_tasks():
        for s1_idx in range(len(src)):
            for tp_idx, (t1_idx, t2_idx) in enumerate(pairs):
                # deterministic sharding by (s1_idx, tp_idx)
                if (stable_seed(s1_idx, tp_idx) % total_shards) != shard_id:
                    continue
                cands = [i for i in range(len(src)) if i != s1_idx]
                s2_idx = choose_one(cands, SEED, 777, s1_idx, tp_idx, t1_idx, t2_idx)
                yield (s1_idx, tp_idx, t1_idx, t2_idx, s2_idx,
                       walk_steps, iters, K, SEED)

    total = sum(1 for _ in shard_tasks())

    def write_rows(rows: List[dict]):
        if gzip_out:
            with gzip.open(shard_path, "ab") as f:
                for r in rows:
                    f.write((json.dumps(r) + "\n").encode("utf-8"))
        else:
            with open(shard_path, "a", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")

    buf, B = [], 512
    t0 = time.time()
    with Pool(initializer=_worker_init, initargs=(src, tgt)) as pool:
        for out in tqdm(pool.imap_unordered(process_combo, shard_tasks(), chunksize=64),
                        total=total, desc=f"Shard {shard_id}/{total_shards-1}"):
            if out is None: 
                continue
            s1_idx, tp_idx, c, t = out
            buf.append({"s1": int(s1_idx), "tp": int(tp_idx), "control": float(c), "triad": float(t)})
            if len(buf) >= B:
                write_rows(buf); buf.clear()
    if buf:
        write_rows(buf)

    print(f"Shard written: {shard_path} (elapsed {time.time()-t0:.1f}s)")


def reduce_run(walk_steps: int, iters: int, K: int, total_shards: int, plot_out: Optional[str], output_summary: str):
    """Combine all shards for a RUN_ID, dedupe, compute stats, optional plot."""
    rid = run_id(walk_steps, iters, K, total_shards, seed=SEED)
    # shard_dir = os.path.join("results", "shards", rid)
    shard_dir = os.path.join(RESULT_DIR, "shards", rid)

    if not os.path.isdir(shard_dir):
        raise ValueError(f"No shard dir: {shard_dir}")

    paths = sorted([os.path.join(shard_dir, f) for f in os.listdir(shard_dir)
                    if f.startswith(rid + "_shard") and (f.endswith(".jsonl") or f.endswith(".jsonl.gz"))])
    if not paths:
        raise ValueError(f"No shard files in {shard_dir}")

    # read + dedupe by (s1,tp)
    rows = []
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        mode = "rt" if p.endswith(".gz") else "r"
        with opener(p, mode, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    df = pd.DataFrame(rows).drop_duplicates(subset=["s1", "tp"], keep="first")
    control = df["control"].to_numpy(float)
    triad   = df["triad"].to_numpy(float)
    n = control.size

    t_stat, p_val = ttest_rel(triad, control, nan_policy="omit")
    try:
        w_stat, w_p = wilcoxon(triad, control, method="approx")
    except Exception:
        w_stat, w_p = (float("nan"), float("nan"))
    dz = cohen_dz(control, triad)
    diff = triad - control
    ci_half = 1.96 * diff.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
    ci = [float(diff.mean() - ci_half), float(diff.mean() + ci_half)]

    summary = {
        "run_id": rid,
        "num_pairs": int(n),
        "stats": {
            "paired_t": {"t": float(t_stat), "df": int(n-1), "p": float(p_val)},
            "wilcoxon": {"W": float(w_stat), "p": float(w_p)},
            "effect_size": {"cohen_dz": float(dz), "mean_diff_ci95": ci}
        },
        "source_files": paths
    }

    os.makedirs(os.path.dirname(output_summary) or ".", exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote summary: {output_summary}")
    print(f"n={n} | t={t_stat:.4f}, p={p_val:.3e} | W={w_stat:.1f}, p={w_p:.3e} | dz={dz:.3f}")

    if plot_out:
        long = pd.DataFrame({
            "pair_id": np.arange(n).tolist() * 2,
            "Condition": ["Control"]*n + ["Triad Network"]*n,
            "Overlap": control.tolist() + triad.tolist()
        })
        plt.figure(figsize=(7,6))
        sns.boxplot(x="Condition", y="Overlap", data=long)
        sns.stripplot(x="Condition", y="Overlap", data=long, alpha=0.4, jitter=0.25)
        plt.ylabel("Overlap (node Jaccard)")
        plt.tight_layout()
        fig = plt.gcf()
        final_path = UtilityFunctions.save_figure_pdf(fig, plot_out, dpi=300, tight=True)
        print(f"Saved plot: {final_path}")
        plt.close(fig)



# --- New helper: mean ± 95% CI plot ---
def plot_means_ci(control: np.ndarray, triad: np.ndarray, out_path: str) -> str:
    """Plot mean±95% CI with a faint distribution backdrop.

    This keeps the "clean" summary (mean + CI) but adds context via:
      - a low-opacity violin (distribution silhouette)
      - optional low-opacity jittered points (downsampled)

    The intent is to avoid clutter while still conveying the spread / tails.
    """

    # Ensure 1D float arrays
    control = np.asarray(control, dtype=float).ravel()
    triad   = np.asarray(triad, dtype=float).ravel()

    data = [control, triad]
    labels = ["Control", "Triad Network"]

    # 95% CI of the mean (normal approx)
    def ci95_mean(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if x.size <= 1:
            return 0.0
        return 1.96 * x.std(ddof=1) / math.sqrt(x.size)

    means = [float(np.mean(control)), float(np.mean(triad))]
    cis   = [float(ci95_mean(control)), float(ci95_mean(triad))]

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(5.5, 5.0))

    # 1) Faint distribution silhouette
    viol = ax.violinplot(
        data,
        positions=[0, 1],
        widths=0.65,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in viol["bodies"]:
        body.set_alpha(0.18)
        body.set_linewidth(0.8)

    # 2) Very low-opacity points (downsampled) for extra texture
    #    (keeps the plot informative without the full clutter)
    max_points = 3000
    point_alpha = 0.03
    jitter_sd = 0.06
    rng = np.random.default_rng(12345)

    for i, x in enumerate(data):
        if x.size == 0:
            continue
        if x.size > max_points:
            idx = rng.choice(x.size, size=max_points, replace=False)
            xs = x[idx]
        else:
            xs = x

        jitter = rng.normal(loc=0.0, scale=jitter_sd, size=xs.size)
        ax.scatter(
            np.full(xs.size, i) + jitter,
            xs,
            s=10,
            alpha=point_alpha,
            linewidths=0,
            zorder=1,
        )

    # 3) Mean ± 95% CI overlay (the "headline" signal)
    ax.errorbar(
        [0, 1],
        means,
        yerr=cis,
        fmt="o",
        capsize=6,
        zorder=3,
    )

    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Overlap (node Jaccard)")
    ax.set_title("Mean Overlap ± 95% CI (with distribution backdrop)")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    print(f"Saved mean±CI (with backdrop) plot to {final_path}")
    plt.close(fig)
    return final_path

    # print(f"Saved plot: {final_path}")
    # plt.close(fig)

    # print(f"Saved mean±CI (with backdrop) plot to {out_path}")
    # return out_path



def plot_run(json_paths: List[str], out_path: Optional[str] = None) -> str:
    """Plot Control vs Triad from one or more monolithic run JSONs. Returns the plot path."""
    num = len(json_paths)
    fig, axes = plt.subplots(1, num, figsize=(8*num, 6), sharey=True)
    if num == 1: axes = [axes]
    for ax, p in zip(axes, json_paths):
        loader = gzip.open if p.endswith(".gz") else open
        mode = "rt" if p.endswith(".gz") else "r"
        with loader(p, mode, encoding="utf-8") as f:
            data = json.load(f)
        control = data["results"]["control_means"]
        triad   = data["results"]["triad_means"]
        long = pd.DataFrame({
            "pair_id": np.arange(len(control)).tolist() * 2,
            "Condition": ["Control"]*len(control) + ["Triad Network"]*len(triad),
            "Overlap": control + triad
        })
        sns.boxplot(x="Condition", y="Overlap", data=long, ax=ax)
        sns.stripplot(x="Condition", y="Overlap", data=long, ax=ax, alpha=0.4, jitter=0.25)
        ax.set_title(os.path.basename(p))
        ax.set_ylabel("Overlap (node Jaccard)")
        ax.set_xlabel("Condition")
    plt.tight_layout()



    if out_path is None:
        out_path = os.path.join(PLOTS_DIR, "experimentF_plot.pdf")

    fig = plt.gcf()
    final_path = UtilityFunctions.save_figure_pdf(fig, out_path, dpi=300, tight=True)
    print(f"Saved plot to {final_path}")
    plt.close(fig)
    return final_path


    # if out_path is None:
    #     out_path = os.path.join(PLOTS_DIR, "experimentF_plot.png")
    # os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # plt.savefig(out_path, dpi=200, bbox_inches="tight")
    # print(f"Saved plot to {out_path}")
    # return out_path



    
# ==========================
# Central-runner entrypoint
# ==========================

def run_experimentF(
    *,
    num_nodes: int,
    average_degree: int,
    seed: int = 42,
    random_walk_steps: int = 20,
    iterations: int = 5,
    start_nodes_sample: int = 10,
    output_dir: Optional[str] = None,
    resume: bool = True,
    save_json: bool = True,
    make_plot: bool = True,
):
    """Run Experiment F (monolithic) from the all-experiments runner.

    Returns:
        (results_dict, plot_paths)

    Results are stored under:
      <output_dir>/result/
      <output_dir>/plots/
    """
    UtilityFunctions.set_global_seed(seed)
    machine_id = UtilityFunctions.get_machine_id()

    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    result_dir = os.path.join(output_dir, "result")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    suffix = f"N{num_nodes}_K{average_degree}_rw{random_walk_steps}_it{iterations}_k{start_nodes_sample}_seed{seed}"
    results_path = os.path.join(result_dir, f"experimentF_results_{suffix}.json.gz")
    plot_path = os.path.join(plots_dir, f"experimentF_plot_{suffix}.pdf")
    mean_ci_plot_path = os.path.join(
        plots_dir,
        f"experimentF_meanCI_{suffix}.pdf"
    )

    did_compute = False

    if not (resume and os.path.exists(results_path)):
        did_compute = True
        payload = run_all(
            random_walk_steps,
            iterations,
            start_nodes_sample,
            results_path,
            num_nodes=num_nodes,
            average_degree=average_degree,
            seed=seed,
        )
    else:
        with gzip.open(results_path, "rt", encoding="utf-8") as f:
            payload = json.load(f)

    # Extract arrays for plotting mean±CI
    control = np.array(payload["results"]["control_means"], dtype=float)
    triad   = np.array(payload["results"]["triad_means"], dtype=float)

    results = {
        "experiment": "experimentF",
        "config": {
            "N": num_nodes,
            "K": average_degree,
            "seed": seed,
            "random_walk_steps": random_walk_steps,
            "iterations": iterations,
            "start_nodes_sample": start_nodes_sample,
        },
        "results_path": results_path,
        "time_taken": float(payload.get("time_taken_seconds", 0.0)) if isinstance(payload, dict) else 0.0,
        "did_compute": did_compute,
        "machine_id": machine_id,
    }

    plot_paths: List[str] = []
    if make_plot:
        out = plot_run([results_path], out_path=plot_path)
        plot_paths.append(out)
        # Also generate mean±CI plot
        out_ci = plot_means_ci(control, triad, mean_ci_plot_path)
        plot_paths.append(out_ci)

    return results, plot_paths




# ----------------- CLI -----------------

# Preferred usage (from repo root):
#   python -m experiments.experimentF.experimentF run
# Direct execution also works via the import fallback above.
def main():
    ap = argparse.ArgumentParser(description="Experiment F (lean)")
    sub = ap.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("run", help="Run all combos, write one JSON(.gz)")
    pr.add_argument("--random_walk_steps", type=int, default=20)
    pr.add_argument("--iterations", type=int, default=5)
    pr.add_argument("--start_nodes_sample", type=int, default=10)
    pr.add_argument("--output", type=str, default=os.path.join(RESULT_DIR, "experimentF_results.json.gz"))



    ps = sub.add_parser("shard", help="Process one shard, write JSONL to results/shards/<RUN_ID>/")
    ps.add_argument("--random_walk_steps", type=int, default=20)
    ps.add_argument("--iterations", type=int, default=5)
    ps.add_argument("--start_nodes_sample", type=int, default=10)
    ps.add_argument("--total_shards", type=int, required=True)
    ps.add_argument("--shard_id", type=int, required=True)
    ps.add_argument("--gzip", action="store_true")

    prd = sub.add_parser("reduce", help="Combine shards for a run id (recomputed from params)")
    prd.add_argument("--random_walk_steps", type=int, required=True)
    prd.add_argument("--iterations", type=int, required=True)
    prd.add_argument("--start_nodes_sample", type=int, required=True)
    prd.add_argument("--total_shards", type=int, required=True)
    prd.add_argument("--plot_out", type=str, default=os.path.join(PLOTS_DIR, "experimentF_summary_plot.pdf"))
    prd.add_argument("--output_summary", type=str, default=os.path.join(RESULT_DIR, "experimentF_summary.json"))


    pp = sub.add_parser("plot", help="Plot monolithic run JSON files")
    pp.add_argument("json_files", nargs="+")

    psp = sub.add_parser("simplified_plot", help="Plot mean ± 95% CI from a saved run JSON(.gz)")
    psp.add_argument("json_file", help="Path to experimentF_results*.json or .json.gz")
    psp.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: experiments/experimentF/plots/experimentF_meanCI_<input_basename>.pdf)",
    )

    args = ap.parse_args()

    if args.mode == "run":
        run_all(args.random_walk_steps, args.iterations, args.start_nodes_sample, args.output, num_nodes=100, average_degree=4, seed=SEED)
    elif args.mode == "shard":
        shard_one(args.random_walk_steps, args.iterations, args.start_nodes_sample,
                  args.total_shards, args.shard_id, args.gzip)
    elif args.mode == "reduce":
        reduce_run(args.random_walk_steps, args.iterations, args.start_nodes_sample,
                   args.total_shards, args.plot_out, args.output_summary)
    elif args.mode == "plot":
        out = plot_run(args.json_files)
        print(f"Plot: {out}")

    elif args.mode == "simplified_plot":
        in_path = args.json_file
        loader = gzip.open if in_path.endswith(".gz") else open
        mode = "rt" if in_path.endswith(".gz") else "r"
        with loader(in_path, mode, encoding="utf-8") as f:
            data = json.load(f)

        control = np.array(data["results"]["control_means"], dtype=float)
        triad   = np.array(data["results"]["triad_means"], dtype=float)

        if args.out is None:
            base = os.path.basename(in_path)
            # strip extensions like .json or .json.gz
            if base.endswith(".json.gz"):
                base = base[:-8]
            elif base.endswith(".gz"):
                base = base[:-3]
            elif base.endswith(".json"):
                base = base[:-5]
            out_path = os.path.join(PLOTS_DIR, f"experimentF_meanCI_{base}.pdf")
        else:
            out_path = args.out

        out = plot_means_ci(control, triad, out_path)
        print(f"Simplified plot: {out}")

if __name__ == "__main__":
    main()
