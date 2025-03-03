import networkx as nx


def find_main_branch(G):
    """
    Finds the longest path in the skeleton graph using Dijkstra's algorithm.
    """
    # Find endpoints (nodes with degree 1)
    endpoints = [node for node, degree in G.degree() if degree == 1]

    # Find longest shortest path among endpoints
    longest_path = []
    max_length = 0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j], weight='weight')
                if len(path) > max_length:
                    max_length = len(path)
                    longest_path = path
            except nx.NetworkXNoPath:
                continue
    return longest_path