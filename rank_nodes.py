import networkx as nx
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union

def rank_nodes(G, S):
    """
    Ranks nodes based on the available area of their inscribed circles,
    considering their intersection with the union of previously selected circles.
    
    Parameters:
        G (networkx.Graph): The graph representation of the medial axis.
        S (set): Set of selected nodes.
    
    Returns:
        best_node (tuple): The best-ranked node.
        best_value (float): The score of the best node.
        ranked_nodes (list): Sorted list of nodes with their scores.
    """
    ranked_nodes = []
    
    # Compute the union of inscribed circles for selected nodes
    selected_circles = [Point(v).buffer(G.nodes[v]['radius']) for v in S if v in G.nodes]
    selected_union = unary_union(selected_circles) if selected_circles else None
    
    for v, data in G.nodes(data=True):
        if v in S:
            continue  # Skip already selected nodes
        
        # Area of the current node's inscribed circle
        A_v = Point(v).buffer(data['radius']).area
        
        # Compute intersection with the selected union
        if selected_union:
            intersection_area = Point(v).buffer(data['radius']).intersection(selected_union).area
        else:
            intersection_area = 0  # No previous selections, so no intersection
        
        # Compute ranking score
        f_v = A_v - intersection_area
        
        ranked_nodes.append((v, f_v))
    
    # Sort nodes by their computed score (descending)
    ranked_nodes.sort(key=lambda x: x[1], reverse=True)
    
    # Return best node and sorted ranking list
    best_node, best_value = ranked_nodes[0] if ranked_nodes else (None, None)
    return best_node, best_value, ranked_nodes


if __name__ == "__main__":
    # Create a sample graph
    G = nx.Graph()

    # Add nodes where the key itself is the position (x, y), and only radius is stored
    G.add_nodes_from([
        ((0, 0), {'radius': 1}),
        ((2, 0), {'radius': 1}),
        ((4, 0), {'radius': 1}),
        ((1, 2), {'radius': 1}),
        ((3, 2), {'radius': 1}),
    ])

    # Start with one pre-selected node
    S = {(0, 0)}  

    print(f"Testing ranking function with initial selected node: {S}\n")

    # Run the ranking function
    best_node, best_value, ranked_nodes = rank_nodes(G, S)

    print("Ranking of Nodes after selecting one:")
    for node, score in ranked_nodes:
        print(f"Node {node} - Score: {score:.4f}")

    print(f"\nBest Next Node: {best_node} with Score: {best_value:.4f}")
