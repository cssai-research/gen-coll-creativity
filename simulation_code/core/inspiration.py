"""
Inspiration mechanism.

This module defines how inspiration is operationalized in the model:
edges discovered in a source subgraph are integrated into a target
cognitive graph, enabling access to previously unreachable concepts.
"""

from __future__ import annotations

import networkx as nx


def integrate_subgraph(
    target_graph: nx.Graph,
    source_subgraph: nx.Graph,
) -> nx.Graph:
    """
    Integrate inspiration from `source_subgraph` into `target_graph`.

    This operation copies the target graph and adds all edges from the
    source subgraph. Nodes are assumed to already exist in the shared
    conceptual space.

    Parameters
    ----------
    target_graph : nx.Graph
        The recipient cognitive graph.
    source_subgraph : nx.Graph
        The source subgraph providing inspirational connections.

    Returns
    -------
    nx.Graph
        A new graph representing the inspired target graph.
    """
    inspired_graph = target_graph.copy()

    if source_subgraph is None:
        return inspired_graph

    for u, v in source_subgraph.edges():
        inspired_graph.add_edge(u, v)

    return inspired_graph