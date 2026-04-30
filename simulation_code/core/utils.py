import random
import socket
import getpass
import hashlib
import os

import numpy as np
import networkx as nx
from scipy.stats import pearsonr


class UtilityFunctions:
    """General-purpose helper functions, including graph computations."""

    @staticmethod
    def compute_modularity(G):
        """Compute modularity of a graph using community detection."""
        import networkx.algorithms.community as nx_comm
        communities = list(nx_comm.greedy_modularity_communities(G))
        return nx_comm.modularity(G, communities)

    @staticmethod
    def random_walk_path(graph, num_steps, start_node=None):
        """Performs a (vanilla) random walk on the given graph and returns a list of visited nodes."""
        if graph is None or not graph.nodes:
            return []

        if start_node is None or start_node not in graph:
            start_node = random.choice(list(graph.nodes))

        current_node = start_node
        path = [current_node]

        for _ in range(num_steps):
            neighbors = list(graph.neighbors(current_node))
            if neighbors:
                current_node = random.choice(neighbors)
                path.append(current_node)
            else:
                break

        return path

    # @staticmethod
    # def random_walk_path(graph, num_steps, start_node=None):
    #     """Performs a (vanilla) random walk on the given graph and returns a list of visited nodes."""
    #     if not graph.nodes:
    #         return []

    #     if start_node is None:
    #         start_node = random.choice(list(graph.nodes))

    #     current_node = start_node
    #     path = [current_node]

    #     for _ in range(num_steps):
    #         neighbors = list(graph.neighbors(current_node))
    #         if neighbors:
    #             current_node = random.choice(neighbors)
    #             path.append(current_node)
    #         else:
    #             break

    #     return path

    @staticmethod
    def subgraph_from_path(graph, path):
        """Forms a subgraph from a given path of nodes."""
        return graph.subgraph(set(path)).copy() if path else nx.Graph()

    @staticmethod
    def pearson_correlation_with_p_value(df, column1, column2, confidence_level=0.95):
        """Calculates Pearson correlation, p-value, and confidence interval between two columns in a DataFrame."""
        correlation, p_value = pearsonr(df[column1], df[column2])
        n = len(df)
        standard_error = np.sqrt((1 - correlation**2) / (n - 2))

        from scipy.stats import t

        critical_value = t.ppf((1 + confidence_level) / 2, n - 2)
        margin_of_error = critical_value * standard_error

        confidence_interval_lower = correlation - margin_of_error
        confidence_interval_upper = correlation + margin_of_error

        return correlation, p_value, confidence_interval_lower, confidence_interval_upper

    @staticmethod
    def compute_num_concepts_accessed(path):
        """Computes the number of unique concepts accessed in a path."""
        return len(set(path))

    @staticmethod
    def compute_diameter(subgraph):
        """Computes the diameter of a subgraph."""
        if subgraph.number_of_nodes() == 0:
            return 0

        if nx.is_connected(subgraph):
            return nx.diameter(subgraph)

        largest_component = max(nx.connected_components(subgraph), key=len)
        subgraph_lcc = subgraph.subgraph(largest_component).copy()
        return nx.diameter(subgraph_lcc)

    @staticmethod
    def compute_overlap(subg1, subg2):
        """Calculates the overlap between two subgraphs."""
        nodes1 = set(subg1.nodes)
        nodes2 = set(subg2.nodes)
        union_size = len(nodes1.union(nodes2))
        if union_size > 0:
            return len(nodes1.intersection(nodes2)) / union_size
        print("Warning: Union size is zero, returning overlap as 0.")
        return 0

    @staticmethod
    def load_graphs(file_pattern):
        """Loads graphs from files matching the given pattern."""
        import glob

        graph_files = sorted(glob.glob(file_pattern))
        graphs = []
        for file in graph_files:
            graph = nx.read_edgelist(file, create_using=nx.Graph(), nodetype=int)
            graphs.append(graph)
        return graphs

    @staticmethod
    def get_machine_id():
        unique_str = socket.gethostname() + getpass.getuser()
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    @staticmethod
    def set_global_seed(seed=42):
        """Sets the random seed for reproducibility across random and numpy."""
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def save_figure_pdf(fig, filepath, dpi=300, tight=True):
        """Save a matplotlib figure as a PDF with consistent settings.

        - Forces PDF output
        - Uses 300 DPI by default
        - Optionally applies tight bounding box

        Returns:
            str: The final path written (always ends with .pdf).
        """
        if filepath is None or str(filepath).strip() == "":
            raise ValueError("save_figure_pdf: filepath must be a non-empty path")

        # Ensure .pdf extension
        root, ext = os.path.splitext(filepath)
        if ext.lower() != ".pdf":
            filepath = root + ".pdf"

        # Ensure parent dir exists (if any)
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)

        save_kwargs = {"dpi": dpi}
        if tight:
            save_kwargs["bbox_inches"] = "tight"

        fig.savefig(filepath, format="pdf", **save_kwargs)
        return filepath