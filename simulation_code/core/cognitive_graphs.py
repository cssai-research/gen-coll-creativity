from __future__ import annotations

import concurrent.futures
import logging
import random

from core.utils import UtilityFunctions

logger = logging.getLogger(__name__)

class RewiringModule:
    """Generate rewired copies of a cognitive graph.

    This module performs edge-rewiring attempts and returns rewired copies.
    It does not guarantee specific target modularity values.
    """
    
    def __init__(self, modularity_distribution=None, iterations: int = 5000):
        # `modularity_distribution` is kept for backward compatibility but is not
        # currently used by this rewiring routine.
        self.modularity_distribution = modularity_distribution
        self.iterations = int(iterations)

    def rewire_and_compute_modularity(self, G, edge1_start, edge1_end, possible_nodes, current_modularity):
        if not possible_nodes:
            return None, None
        edge2_end = random.choice(list(possible_nodes))

        # Modify the graph
        G_rewired = G.copy()
        G_rewired.remove_edge(edge1_start, edge1_end)
        G_rewired.add_edge(edge1_start, edge2_end)

        new_modularity = UtilityFunctions.compute_modularity(G_rewired)

        # Accept modest modularity changes; too-strict thresholds can collapse
        # diversity and lead to many fallbacks. [todo: review logic] 
        if abs(new_modularity - current_modularity) > 0.001:
            return G_rewired, new_modularity
        
        # Otherwise, revert changes and return None
        G_rewired.remove_edge(edge1_start, edge2_end)
        G_rewired.add_edge(edge1_start, edge1_end)
        return None, None

    def parallel_rewire_and_compute(self, G, edges, current_modularity):
        graphs_list = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.rewire_and_compute_modularity,
                    G,
                    edge1_start,
                    edge1_end,
                    possible_nodes,
                    current_modularity,
                )
                for edge1_start, edge1_end, possible_nodes in edges
            ]

            for future in concurrent.futures.as_completed(futures):
                G_rewired, _new_modularity = future.result()
                if G_rewired is not None and G_rewired.number_of_edges() == G.number_of_edges():
                    graphs_list.append(G_rewired)
        return graphs_list

    def rewire_graphs(self, G, num_graphs):
        """Generate `num_graphs` rewired copies of `G`.

        If insufficient valid rewires are found within bounded attempts,
        the remainder are filled with unchanged copies of `G`.
        """
        if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
            return []  # No edges to rewire, return empty list

        graphs_list = []
        max_attempts = 5  # Limit retries to prevent infinite looping
        attempt = 0

        while len(graphs_list) < num_graphs and attempt < max_attempts:
            G_rewired = G.copy()
            edges = []

            # Compute once per attempt; the previous version updated this inside the
            # loop and then passed only the *last* value to all workers.
            current_modularity = UtilityFunctions.compute_modularity(G_rewired)

            for _ in range(self.iterations):
                if G_rewired.number_of_edges() == 0:
                    continue  # No edges to rewire

                edge1_start, edge1_end = random.choice(list(G_rewired.edges()))
                possible_nodes = set(G_rewired.nodes()) - {edge1_start, edge1_end} - set(G_rewired.neighbors(edge1_start))
                logger.debug(f"Rewiring edge ({edge1_start}, {edge1_end}) with possible nodes: {possible_nodes}")

                if possible_nodes:
                    edges.append((edge1_start, edge1_end, possible_nodes))

            if not edges:
                logger.warning("No valid rewiring candidates found for this attempt.")
                attempt += 1
                continue

            # Avoid submitting an excessive number of futures.
            max_candidates = max(100, num_graphs * 20)
            if len(edges) > max_candidates:
                edges = random.sample(edges, k=max_candidates)

            rewired_graphs = self.parallel_rewire_and_compute(G_rewired, edges, current_modularity)
            graphs_list.extend(rewired_graphs)

            attempt += 1
            if len(graphs_list) < num_graphs:
                logger.warning(f"Retrying rewiring (Attempt {attempt}/{max_attempts}) due to insufficient graphs.")

        # Final fallback: Ensure output consistency without adding extra edges
        while len(graphs_list) < num_graphs:
            G_fallback = G.copy()
            graphs_list.append(G_fallback)
            logger.debug("Fallback: Returning an unchanged copy of the graph to meet output requirements.")

        return graphs_list[:num_graphs]
