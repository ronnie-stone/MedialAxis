import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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

def compute_bisectors_at_branch(G, point, n_bisectors=3, order='cw', tol=1e-6):
    """
    Computes bisector vectors at a branching point in the medial axis graph.
    
    Parameters:
        G (networkx.Graph): Medial axis graph.
        point (tuple): Coordinates of the branching point.
        n_bisectors (int): Number of bisectors to compute (either 2 or 3).
        order (str): 'cw' for clockwise, 'ccw' for counter-clockwise edge ordering.
        tol (float): Tolerance for collinear detection.

    Returns:
        list of np.array: List of bisector vectors (unit vectors).
    """
    neighbors = list(G.neighbors(point))
    degree = len(neighbors)

    if degree not in [2, 3]:
        raise ValueError(f"compute_bisectors_at_branch only supports degree-2 or degree-3 nodes, but node {point} has degree {degree}")

    # Compute unit vectors to neighbors
    vectors = [compute_unit_vector(point, neighbor) for neighbor in neighbors]

    if degree == 2 and n_bisectors == 2:
        # Simple case: exactly two edges, no need to find best pair, just directly compute bisector(s)
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

        return [bisector1, bisector2]

    # Compute angles relative to +x axis (for degree-3 only)
    angles = np.array([compute_angle(v) for v in vectors])

    # Sort edges in rotational order
    order_indices = np.argsort(angles) if order == 'ccw' else np.argsort(-angles)
    ordered_vectors = [vectors[i] for i in order_indices]

    if degree == 3 and n_bisectors == 3:
        # Original degree-3, 3-bisector case (DO NOT TOUCH THIS)
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

            bisectors.append(bisector)

        return bisectors

    elif degree == 3 and n_bisectors == 2:
        # Degree-3 but only want 2 bisectors (same logic we fixed before)
        best_pair = None
        smallest_angle_diff = float('inf')
        best_indices = None

        for i in range(3):
            v1 = vectors[i]
            v2 = vectors[(i+1) % 3]
            angle = compute_angle_between(v1, v2)
            angle_diff = np.abs(np.pi - angle)
            if angle_diff < smallest_angle_diff:
                smallest_angle_diff = angle_diff
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

        return [bisector1, bisector2]

    else:
        raise ValueError(f"Unexpected case: degree={degree}, n_bisectors={n_bisectors}")


if __name__ == "__main__":
    
    branching_point = (1.5, 1.5)
    G = nx.Graph()

    # Example graph (a Y-shape)
    #G.add_edges_from([
    #    ((5, 5), (6, 6)),
    #    ((5, 5), (6, 5)),
    #    ((5, 5), (4, 5))
    #])

    G.add_edges_from([
        ((1.5, 1.5), (3, 1.5)),
        ((1.5, 1.5), (0, 2))])

    bisectors = compute_bisectors_at_branch(G, branching_point, n_bisectors=2)

    for i, b in enumerate(bisectors):
        print(f"Bisector {i+1}: {b}")

    # --- Plotting Section ---
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot edges in the graph (the Y-shape)
    for u, v in G.edges():
        x, y = zip(u, v)
        ax.plot(x, y, 'k-', linewidth=2)

    # Plot the branching point
    ax.scatter(branching_point[0], branching_point[1], c='red', s=100, label="Branching Point")

    # Plot the bisectors starting from the branching point
    length = 0.5  # length of bisector line for visualization
    for i, bisector in enumerate(bisectors):
        end = np.array(branching_point) + bisector * length
        ax.plot([branching_point[0], end[0]], [branching_point[1], end[1]], 
                linestyle='--', linewidth=2, label=f'Bisector {i+1}')

    ax.legend()
    ax.set_aspect('equal')
    ax.set_title("Branching Point with Bisectors")
    plt.show()
