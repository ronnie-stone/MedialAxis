import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from get_medial_axis_voronoi import get_medial_axis
from medial_axis_to_graph import medial_axis_to_graph
from idetc_main_alg import find_best_partition_and_nodes
import random as rd

def compute_unit_vector(p1, p2):
    """Returns a unit vector pointing from p1 to p2."""
    vector = np.array(p2) - np.array(p1)
    return vector / np.linalg.norm(vector)

def compute_angle(v):
    """Angle in radians between vector and positive x-axis."""
    return np.arctan2(v[1], v[0])

def compute_angle_between(v1, v2):
    """Compute angle between two vectors."""
    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_angle)

def compute_perpendicular(v):
    """Compute a perpendicular vector to v."""
    return np.array([-v[1], v[0]])

import numpy as np

def compute_middle_vector(v1, v2, tol=1e-6):
    """
    Computes the unit vector in between two given unit vectors.
    Ensures that the resulting vector does not collapse to zero.
    
    Parameters:
        v1 (np.array): First unit vector.
        v2 (np.array): Second unit vector.
        tol (float): Tolerance for handling near-opposite vectors.

    Returns:
        np.array: The unit vector in between v1 and v2.
    """
    middle = v1 + v2
    norm = np.linalg.norm(middle)

    if norm < tol:  # Nearly opposite vectors case
        perpendicular = np.array([-v1[1], v1[0]])  # 90-degree rotation
        return perpendicular / np.linalg.norm(perpendicular)

    return middle / norm

# Example usage
v1 = np.array([1, 0])  # Unit vector along x-axis
v2 = np.array([0, 1])  # Unit vector along y-axis

middle_vector = compute_middle_vector(v1, v2)
print("Middle Vector:", middle_vector)

def compute_bisectors_at_branch_2(G, point, n_bisectors=3, order='cw', tol=1e-6):
    """
    Computes bisector vectors at a branching point in the medial axis graph,
    ensuring that the bisector lengths are bounded by the radius function value.

    Parameters:
        G (networkx.Graph): Medial axis graph.
        point (tuple): Coordinates of the branching point.
        n_bisectors (int): Number of bisectors to compute (either 2 or 3).
        order (str): 'cw' for clockwise, 'ccw' for counter-clockwise edge ordering.
        tol (float): Tolerance for collinear detection.

    Returns:
        list of np.array: List of bisector vectors (scaled by node radius).
    """
    neighbors = list(G.neighbors(point))
    degree = len(neighbors)

    if degree not in [2, 3]:
        raise ValueError(f"compute_bisectors_at_branch only supports degree-2 or degree-3 nodes, but node {point} has degree {degree}")

    # Get radius function value for this node
    radius = G.nodes[point]['radius']

    # Compute unit vectors to neighbors
    vectors = [compute_unit_vector(point, neighbor) for neighbor in neighbors]

    if degree == 2 and n_bisectors == 2:
        # Simple case: exactly two edges, directly compute bisector(s)
        v1, v2 = vectors

        avg = v1 + v2
        norm = np.linalg.norm(avg)

        if norm < tol:
            # Handle collinear case - directly perpendicular to either edge
            perpendicular = compute_perpendicular(v1)
            bisector1 = perpendicular
            bisector2 = -perpendicular
        else:
            bisector1 = avg / norm
            bisector2 = -bisector1

        # Scale bisectors by radius
        return [bisector1 * radius, bisector2 * radius]

    # Compute angles relative to +x axis (for degree-3 only)
    angles = np.array([compute_angle(v) for v in vectors])

    # Sort edges in rotational order
    order_indices = np.argsort(angles) if order == 'ccw' else np.argsort(-angles)
    ordered_vectors = [vectors[i] for i in order_indices]

    if degree == 3 and n_bisectors == 3:
        # Original degree-3, 3-bisector case
        bisectors = []
        for i in range(3):
            v1 = ordered_vectors[i]
            v2 = ordered_vectors[(i+1) % 3]

            avg = v1 + v2
            norm = np.linalg.norm(avg)

            if norm < tol:
                third_vector = ordered_vectors[(i+2) % 3]
                perpendicular = compute_perpendicular(v1)
                if np.dot(perpendicular, third_vector) > 0:
                    perpendicular *= -1
                bisector = perpendicular
            else:
                bisector = avg / norm

            # Scale bisector by radius
            bisectors.append(bisector * radius)

        return bisectors

    elif degree == 3 and n_bisectors == 2:
        # Degree-3 but only want 2 bisectors
        best_pair = None
        smallest_angle_diff = float('inf')
        largest_angle_diff = 0
        best_indices = None

        for i in range(3):
            v1 = vectors[i]
            v2 = vectors[(i+1) % 3]
            angle = compute_angle_between(v1, v2)
            angle_diff = np.abs(np.pi - angle)
            if angle_diff > largest_angle_diff:
                largest_angle_diff = angle_diff
                best_pair = (v1, v2)
                best_indices = (i, (i+1) % 3)

        # Step 2: Compute the bisector using the collinear-safe logic
        v1, v2 = best_pair
        avg = v1 + v2
        norm = np.linalg.norm(avg)

        if norm < tol:
            # Handle collinear edges case — compute perpendicular
            third_vector = vectors[3 - sum(best_indices)]  # The ignored third edge
            perpendicular = compute_perpendicular(v1)
            if np.dot(perpendicular, third_vector) > 0:
                perpendicular *= -1
            bisector1 = perpendicular
        else:
            bisector1 = avg / norm

        bisector2 = -bisector1

        # Scale bisectors by radius
        return [bisector1 * radius, bisector2 * radius]

    else:
        raise ValueError(f"Unexpected case: degree={degree}, n_bisectors={n_bisectors}")

def compute_bisectors_at_branch(G, point, n_bisectors=3):
    """
    Computes simplified bisector vectors at a branching point in the medial axis graph.
    
    Parameters:
        G (networkx.Graph): Medial axis graph.
        point (tuple): Coordinates of the branching point.
        n_bisectors (int): Number of bisectors to compute.

    Returns:
        list of np.array: List of bisector vectors (unit vectors).
    """
    neighbors = list(G.neighbors(point))
    degree = len(neighbors)

    if degree == 0:
        raise ValueError(f"Node {point} has no edges.")

    # Compute unit vectors to neighbors
    vectors = [compute_unit_vector(point, neighbor) for neighbor in neighbors]

    if n_bisectors == degree:
        # Standard case: Compute bisectors between edges
        bisectors = []
        for i in range(degree):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % degree]  # Wrap around for last pair
            
            bisector = compute_middle_vector(v1, v2)
            bisectors.append(bisector)
        
        return bisectors

    elif n_bisectors > degree:
        # More bisectors than edges → Use evenly spaced directions
        angles = np.linspace(0, 2 * np.pi, n_bisectors, endpoint=False)
        bisectors = [np.array([np.cos(a), np.sin(a)]) for a in angles]
        return bisectors

    else:
        # Fewer bisectors than edges → Compute and randomly pick n_bisectors
        all_bisectors = []
        for i in range(degree):
            v1 = vectors[i]
            v2 = vectors[(i + 1) % degree]  # Wrap around

            bisector = compute_middle_vector(v1, v2)
            all_bisectors.append(bisector)

        # Randomly select the required number of bisectors
        return rd.sample(all_bisectors, n_bisectors)


if __name__ == "__main__":
    
    # Input polygon (choose one)
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 
    buffer_size = 0.05

    # Step 1: Compute the Medial Axis and Transform to a Graph
    medial_axis, radius_map = get_medial_axis(input_polygon_coords, num_samples=1000)
    G = medial_axis_to_graph(medial_axis, radius_map)

    # Step 2: Define the number of robots and find the best partition
    N = 5  # Number of robots
    best_partition, best_nodes, best_assignments = find_best_partition_and_nodes(G, N, alpha=1.0)

    print("Best Partition:", best_partition)
    print("Best Nodes:", best_nodes)
    print("Best Assignments:", best_assignments)

    # Step 3: Compute bisectors for each selected node
    all_bisectors = []
    for node in best_nodes:
        cluster_size = best_assignments[node]  # Get the assigned cluster size
        bisectors = compute_bisectors_at_branch(G, node, n_bisectors=cluster_size)  # Compute bisectors

        # Store each bisector along with its originating node
        all_bisectors.extend([(node, bis) for bis in bisectors])

    # Step 5: Create subplots for visualization
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # ---- Figure 1: Graph Visualization ----
    pos = {node: node for node in G.nodes}
    degrees = dict(G.degree())

    # Draw the medial axis graph
    nx.draw(G, pos, with_labels=False, node_color='lightgray', edge_color='gray', node_size=25, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=best_nodes, node_color='red', node_size=100, ax=ax)

    # Annotate selected nodes with cluster size
    for node in best_nodes:
        x, y = pos[node]
        ax.text(x, y + 0.2, f"cluster={best_assignments[node]}", fontsize=10, ha='center', color='black')

    ax.set_title("Medial Axis Graph with Selected Nodes")

    # Plot the medial axis edges
    for u, v in G.edges():
        x, y = zip(*[u, v])
        ax.plot(x, y, 'k-', linewidth=2)

    # Plot the bisectors from branching points
    for i, (branching_point, bisector) in enumerate(all_bisectors):
        end = np.array(branching_point) + bisector  # Bisectors are now scaled by radius
        ax.plot([branching_point[0], end[0]], [branching_point[1], end[1]], 
                     linestyle='--', linewidth=2, label=f'Bisector {i+1}')

    # Show the plots
    plt.show()
