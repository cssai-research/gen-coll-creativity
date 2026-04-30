
import json


import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)


import networkx as nx
from framework import UtilityFunctions
from itertools import combinations
import glob

def process_one(N, k, data_dir):
    file_pattern = os.path.join(data_dir, f"G_n_{N}_k_{k}_*.edgelist.gz")
    graphs = UtilityFunctions.load_graphs(file_pattern)
    graph_files = sorted([os.path.basename(f) for f in glob.glob(file_pattern)])
    assert len(graphs) == len(graph_files), f"Mismatch for N={N}, k={k}!"

    graph_metadata = []
    id_to_modularity = {}

    for i, (graph, fname) in enumerate(zip(graphs, graph_files)):
        graph_id = f"g{i:03d}"
        modularity = UtilityFunctions.compute_modularity(graph)
        graph_metadata.append({
            "graph_id": graph_id,
            "filename": fname,
            "modularity": modularity
        })
        id_to_modularity[graph_id] = modularity
        graph.graph["id"] = graph_id

    with open(os.path.join(data_dir, f"graph_metadata_master_n{N}_k{k}.json"), "w") as f:
        json.dump(graph_metadata, f, indent=4)

    pairs = []
    graph_ids = [g["graph_id"] for g in graph_metadata]
    pair_index = 0
    for source, target in combinations(graph_ids, 2):
        for s, t in [(source, target), (target, source)]:
            pairs.append({
                "pair_index": pair_index,
                "source_graph_id": s,
                "target_graph_id": t,
                "source_modularity": id_to_modularity[s],
                "target_modularity": id_to_modularity[t]
            })
            pair_index += 1

    with open(os.path.join(data_dir, f"pair_index_master_n{N}_k{k}.json"), "w") as f:
        json.dump(pairs, f, indent=4)

    print(f"Wrote {len(graph_metadata)} graphs and {len(pairs)} pairs for N={N}, k={k}.")

def main():
    # data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "saved_cog_nets"))

    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "new_saved_cog_nets"))
    configs = [(100, 4), (300, 10)]
    for N, k in configs:
        process_one(N, k, data_dir)

if __name__ == "__main__":
    main()