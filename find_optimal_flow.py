import networkx as nx
from medial_axis_to_graph import medial_axis_to_graph
from connect_limbs_to_boundary import connect_limbs_to_boundary
from get_medial_axis_voronoi import get_medial_axis
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiLineString, LineString


def plot_partition_result(polygon_coords, medial_axis, G, paths):
    polygon = Polygon(polygon_coords)
    x, y = polygon.exterior.xy

    plt.figure(figsize=(10, 8))
    plt.plot(x, y, 'k-', linewidth=2, label='Polygon')

    if isinstance(medial_axis, MultiLineString):
        for line in medial_axis.geoms:
            x, y = line.xy
            plt.plot(x, y, 'r-', linewidth=1, label='Medial Axis' if 'Medial Axis' not in plt.gca().get_legend_handles_labels()[1] else None)
    elif isinstance(medial_axis, LineString):
        x, y = medial_axis.xy
        plt.plot(x, y, 'r-', linewidth=1, label='Medial Axis')

    for u, v in G.edges():
        x, y = zip(*[u, v])
        plt.plot(x, y, 'b--', linewidth=0.8)

    colors = ['green', 'orange', 'purple', 'cyan', 'magenta']
    for i, path in enumerate(paths):
        x, y = zip(*path)
        plt.plot(x, y, color=colors[i % len(colors)], linewidth=3, label=f'Partition Path {i+1}')

    plt.legend()
    plt.title("Medial Axis Partition Result")
    plt.axis('equal')
    plt.show()

def find_optimal_flow(G, boundary_nodes, branching_nodes, num_parts):
    """
    Finds a set of paths that partition the medial axis into `num_parts` subregions.

    Parameters:
        G (networkx.Graph): Medial axis graph (with boundary connections).
        boundary_nodes (list of nodes): Nodes directly connected to polygon boundary.
        branching_nodes (list of nodes): Nodes with degree > 2.
        num_parts (int): Desired number of partitions.

    Returns:
        list of paths (each path is a list of nodes).
    """
    if num_parts < 2:
        raise ValueError("At least 2 partitions required.")

    paths = []

    if num_parts == 2:
        # Simple: Find shortest path between two boundary nodes
        node1, node2 = boundary_nodes[:2]
        path = nx.shortest_path(G, source=node1, target=node2, weight='weight')
        paths.append(path)

    else:
        # More complex: Need to pick internal branching points
        num_branching_points = num_parts - 2

        if len(branching_nodes) < num_branching_points:
            raise ValueError(f"Not enough branching points (need {num_branching_points}, found {len(branching_nodes)})")

        # Select some central branching nodes (simple heuristic: just grab first few)
        chosen_branches = branching_nodes[:num_branching_points]

        # Now we need to route each boundary node to one of these branching points
        used_boundary_nodes = set()

        for branch in chosen_branches:
            # Find closest 2 boundary nodes (for 3-way split)
            closest_boundaries = sorted(boundary_nodes, key=lambda n: nx.shortest_path_length(G, source=n, target=branch, weight='weight'))
            for boundary in closest_boundaries:
                if boundary not in used_boundary_nodes:
                    path = nx.shortest_path(G, source=boundary, target=branch, weight='weight')
                    paths.append(path)
                    used_boundary_nodes.add(boundary)
                    break

        # For larger k, you could extend this to find "connecting paths" between branching points
        if num_parts > 3:
            # Example for k=4 (2 branching points)
            for i in range(len(chosen_branches) - 1):
                path = nx.shortest_path(G, source=chosen_branches[i], target=chosen_branches[i+1], weight='weight')
                paths.append(path)

        # After this, you should have num_parts - 1 paths (since each path creates a cut)

    return paths

if __name__ == "__main__":
    polygon_coords = [
        (0, 0), (6, 0), (6, 3), (3, 6), (0, 3), (0, 0)
    ]
    # polygon_coords = np.load('bunny_cross_section_scaled.npy')  # If you have your bunny file

    num_parts = 3  # Example: 3-way partition

    medial_axis = get_medial_axis(polygon_coords)
    G = medial_axis_to_graph(medial_axis)
    G = connect_limbs_to_boundary(G, polygon_coords)

    boundary_nodes = [node for node, degree in G.degree() if degree == 1]
    branching_nodes = [node for node, degree in G.degree() if degree > 2]

    paths = find_optimal_flow(G, boundary_nodes, branching_nodes, num_parts)

    plot_partition_result(polygon_coords, medial_axis, G, paths)
