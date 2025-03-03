import numpy as np
import networkx as nx


def skeleton_to_graph(skeleton):
    """
    Converts a skeleton image into a graph.
    Each skeleton pixel is a node, and edges connect neighboring pixels.
    """
    G = nx.Graph()
    rows, cols = np.where(skeleton)
    for y, x in zip(rows, cols):
        G.add_node((x, y))
        # 8-connectivity
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < skeleton.shape[1] and 0 <= ny_ < skeleton.shape[0]:
                    if skeleton[ny_, nx_]:
                        G.add_edge((x, y), (nx_, ny_), weight=1)
    return G


if __name__ == "__main__":
    # Write some tests here.
    pass