import networkx as nx
import numpy as np
from shapely.geometry import MultiLineString, LineString, Polygon, Point
import matplotlib.pyplot as plt
from get_medial_axis_voronoi import get_medial_axis
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph
from medial_axis_to_graph import medial_axis_to_graph



def find_valid_cluster_partitions(N, max_clusters, min_cluster=2):
    """
    Generate valid cluster partitions ensuring that the sum equals N
    and the number of clusters does not exceed max_clusters.
    """
    all_partitions = []
    
    def find_partitions(current_partition=[], remaining=N, min_size=min_cluster):
        if remaining == 0:
            if len(current_partition) <= max_clusters:
                all_partitions.append(list(current_partition))
            return

        for size in range(min_size, remaining + 1):
            current_partition.append(size)
            find_partitions(current_partition, remaining - size, size)
            current_partition.pop()  # Backtrack

    find_partitions()

    

    # Step 2: Select the partition with the most degree-3 clusters
    def get_best_partition(partitions):
        best_partition = None
        max_degree_3_count = -1

        for partition in partitions:
            degree_3_count = sum(1 for c in partition if c == 3)  # Count degree-3 clusters

            if degree_3_count > max_degree_3_count:
                max_degree_3_count = degree_3_count
                best_partition = partition  # Store the best partition

        return [best_partition]  # Return the one with the most degree-3s

    return get_best_partition(all_partitions)



def compute_overlap_area(node1, node2, G):
    """
    Computes the actual overlap area between two circles centered at node1 and node2
    using Shapely's geometry operations.
    """
    p1, p2 = Point(node1), Point(node2)
    r1, r2 = G.nodes[node1]['radius'], G.nodes[node2]['radius']

    # Create two circles using buffer (radius expansion)
    circle1 = p1.buffer(r1)
    circle2 = p2.buffer(r2)

    # Compute intersection
    intersection = circle1.intersection(circle2)

    return intersection.area  # Extract the overlapping area

def select_optimal_nodes(G, clusters, alpha=1.0):
    """
    Selects the best nodes in G to maximize coverage, considering
    both large radius values and minimizing degree mismatch.
    Ensures higher-degree clusters are placed first.
    """
    sorted_clusters = sorted(clusters, reverse=True)  # Place higher-degree clusters first
    selected_nodes = []
    assignments = {}

    for cluster in sorted_clusters:
        best_node = None
        best_score = -float('inf')

        for node in G.nodes:
            if node in selected_nodes:
                continue

            degree = G.degree[node]
            radius = G.nodes[node]['radius']

            # Compute coverage and overlap using exact intersection
            coverage = np.pi * radius**2
            overlap = sum(compute_overlap_area(node, v, G) for v in selected_nodes)
            penalty = alpha * coverage * abs(degree - cluster)

            score = coverage - overlap

            if score > best_score:
                best_score = score
                best_node = node

        if best_node is not None:
            selected_nodes.append(best_node)
            assignments[best_node] = cluster  # Assign the cluster to the selected node

    return selected_nodes, assignments

def find_best_partition_and_nodes(G, N, alpha=1.0):
    """
    Finds the best cluster partition and selects nodes maximizing coverage,
    while considering degree mismatch directly in node selection.
    """
    valid_partitions = find_valid_cluster_partitions(N, max_clusters=len(G.nodes))

    best_partition = None
    best_nodes = None
    best_assignments = None
    best_score = -float('inf')

    for partition in valid_partitions:
        selected_nodes, assignments = select_optimal_nodes(G, partition, alpha)

        coverage = sum(np.pi * G.nodes[v]['radius']**2 for v in selected_nodes)
        overlap = sum(compute_overlap_area(v_i, v_j, G) for v_i in selected_nodes for v_j in selected_nodes if v_i != v_j)
        penalty = sum(alpha * abs(G.degree[node] - assignments[node]) for node in assignments)
        net_coverage = coverage - overlap - penalty
        
        if net_coverage > best_score:
            best_score = net_coverage
            best_partition = partition
            best_nodes = selected_nodes
            best_assignments = assignments

    return best_partition, best_nodes, best_assignments

if __name__ == "__main__":
    # Create a toy medial axis graph with radius attributes
    # polygon_coords = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]  # Square
    # polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)]  # Rectangle
    # polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)]  # Triangle
    # polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    polygon_coords = np.load('bunny_cross_section_scaled.npy')
    #polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    medial_axis, radius_function = get_medial_axis(polygon_coords)
    G = medial_axis_to_graph(medial_axis, radius_function)

    # Example test with N=6 robots
    N = 9
    best_partition, best_nodes, best_assignments = find_best_partition_and_nodes(G, N, alpha=1.0)
    
    print("Best Partition of Robots:", best_partition)
    print("Selected Nodes:", best_nodes)
    print("Cluster Assignments:", best_assignments)
    
    # Plot the graph and highlight selected nodes
    plt.figure(figsize=(8, 6))
    pos = {node: node for node in G.nodes}
    degrees = dict(G.degree())
    
    nx.draw(G, pos, with_labels=False, node_color='lightgray', edge_color='gray', node_size=100)
    nx.draw_networkx_nodes(G, pos, nodelist=best_nodes, node_color='red', node_size=100)
    
    # Annotate only the selected nodes with their degrees and assigned cluster size
    for node in best_nodes:
        x, y = pos[node]
        plt.text(x, y + 0.2, f"d={degrees[node]}, c={best_assignments[node]}", fontsize=6, ha='center', color='black')
    
    plt.title("Selected Nodes for Clustering")
    plt.show()