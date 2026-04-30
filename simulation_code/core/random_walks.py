# NOTE: We are using the random walk option present in utils instead of this file.


# """Random-walk sampling utilities for cognitive graphs.

# This module defines how agents traverse cognitive graphs to sample
# accessible conceptual subgraphs. It contains *only* walk and subgraph
# construction logic (no experiments, no plotting, no I/O).
# """

# from __future__ import annotations

# import random
# from typing import Any, List, Sequence

# import networkx as nx


# class RandomWalkSampler:
#     """Random-walk sampler for cognitive graphs.

#     Note: randomness is controlled externally by seeding Python's `random` module
#     (e.g., via `UtilityFunctions.set_global_seed`).
#     """

#     @staticmethod
#     def random_walk_path(
#         graph: nx.Graph,
#         steps: int,
#         start_node: Any,
#     ) -> List[Any]:
#         """Generate a random-walk path of length `steps`.

#         Behavior is intentionally aligned with `UtilityFunctions.random_walk_path`:
#           - If the graph is empty or None: return [].
#           - If start_node is None: pick a random node from the graph.
#           - If start_node is not in the graph: return [].
#           - If a dead-end is reached (no neighbors): stop early and return the path.

#         Returns a list of visited nodes (including the start node when available).
#         """
#         if graph is None or graph.number_of_nodes() == 0:
#             return []

#         if start_node is None:
#             start_node = random.choice(list(graph.nodes()))

#         if start_node not in graph:
#             return []

#         steps = int(steps)
#         if steps <= 0:
#             return [start_node]

#         current = start_node
#         path: List[Any] = [current]

#         for _ in range(steps):
#             neighbors = list(graph.neighbors(current))
#             if not neighbors:
#                 break
#             current = random.choice(neighbors)
#             path.append(current)

#         return path

#     @staticmethod
#     def subgraph_from_path(
#         graph: nx.Graph,
#         path: Sequence[Any] | None,
#     ) -> nx.Graph:
#         """Return the induced subgraph corresponding to nodes in `path`.

#         Behavior is aligned with `UtilityFunctions.subgraph_from_path`:
#           - If graph is None or path is empty/None: return an empty nx.Graph().
#           - Otherwise: return the induced subgraph over the visited nodes.
#         """
#         if graph is None or not path:
#             return nx.Graph()

#         nodes = set(path)
#         if not nodes:
#             return nx.Graph()

#         return graph.subgraph(nodes).copy()