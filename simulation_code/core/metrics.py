"""Graph-level metrics utilities used across experiments.

- No plotting
- No file I/O
- Pure computations on graphs/subgraphs/paths
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence

import networkx as nx

from core.utils import UtilityFunctions


class GraphAnalyzer:
    """Small helper utilities used by experiment scripts."""

    @staticmethod
    def find_common_start_node(graph_list: Sequence[nx.Graph]):
        """Return a node that exists in all graphs (or None)."""
        if not graph_list:
            return None
        common_nodes = set(graph_list[0].nodes())
        for g in graph_list[1:]:
            common_nodes.intersection_update(g.nodes())
        return random.choice(tuple(common_nodes)) if common_nodes else None

    @staticmethod
    def get_unique_elements(list_of_lists: Sequence[Sequence[Any]]):
        """Return per-list unique elements relative to the union of all others."""
        unique_elements: List[set] = []
        for i, l in enumerate(list_of_lists):
            other = set().union(*[set(list_of_lists[j]) for j in range(len(list_of_lists)) if j != i])
            unique_elements.append(set(l) - other)
        return unique_elements

    @staticmethod
    def compute_unique_items_length(
        path: Optional[Sequence[Any]],
        unique_elements_list: Sequence[set],
        list_of_paths: Sequence[Sequence[Any]],
    ) -> int:
        """Length of the precomputed unique-elements set for the given path."""
        if not path:
            return 0
        try:
            idx = list_of_paths.index(list(path))
        except ValueError:
            return 0
        if idx < 0 or idx >= len(unique_elements_list):
            return 0
        return len(unique_elements_list[idx])

    @staticmethod
    def compute_metrics(
        graph: nx.Graph,
        subgraph: nx.Graph,
        path: Sequence[Any],

        include_diameter: bool,
        include_unique_items: bool,
        unique_elements_list: Sequence[set],
        list_of_paths: Sequence[Sequence[Any]],
    ) -> Dict[str, Any]:
        """Compute a standard set of metrics for one graph/subgraph/path triple."""
        metrics: Dict[str, Any] = {
            "Modularity": UtilityFunctions.compute_modularity(graph),
            "Num_Concepts_Accessed": UtilityFunctions.compute_num_concepts_accessed(path),
        }

        if include_diameter:
            metrics["Diameter"] = UtilityFunctions.compute_diameter(subgraph)


        if include_unique_items:
            metrics["Unique Items List Length"] = GraphAnalyzer.compute_unique_items_length(
                path, unique_elements_list, list_of_paths
            )

        return metrics

    @staticmethod
    def report_graph_metrics_simpler(
        graphs: Sequence[nx.Graph],
        subgraphs: Sequence[tuple[Optional[nx.Graph], Optional[Sequence[Any]]]],
    ) -> List[Dict[str, Any]]:
        """Return per-graph metrics used in several experiments.

        Expected `subgraphs` format: [(subgraph, path), ...]
        where `path` is a list of visited nodes.

        Diameter is computed on the largest connected component if disconnected.
        """
        results: List[Dict[str, Any]] = []

        for i, (graph, (subgraph, path)) in enumerate(zip(graphs, subgraphs)):
            modularity = UtilityFunctions.compute_modularity(graph)
            num_concepts_accessed = len(set(path)) if path else 0

            if subgraph is None or subgraph.number_of_nodes() == 0:
                diameter = 0
            else:
                # Ensure we measure diameter on a connected component.
                if nx.is_connected(subgraph):
                    diameter = nx.diameter(subgraph)
                else:
                    largest_component = max(nx.connected_components(subgraph), key=len)
                    subgraph_lcc = subgraph.subgraph(largest_component).copy()
                    diameter = nx.diameter(subgraph_lcc) if subgraph_lcc.number_of_nodes() > 0 else 0

            results.append(
                {
                    "Index": i,
                    "Modularity": modularity,
                    "Num Concepts Accessed": num_concepts_accessed,
                    "Diameter": diameter,
                }
            )

        return results
