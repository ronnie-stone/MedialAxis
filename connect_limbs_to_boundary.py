from shapely.geometry import Point, LineString


def connect_limbs_to_boundary(G, polygon_coords):
    """
    Connect all degree-1 nodes in the medial axis graph to the nearest point on the polygon boundary.

    Parameters:
        G (networkx.Graph): Graph representation of medial axis.
        polygon_coords (list of (x, y)): Polygon vertices (closed loop preferred).

    Returns:
        networkx.Graph: Modified graph with new edges connecting limbs to boundary.
    """

    boundary_line = LineString(polygon_coords)

    # Find all leaf nodes (degree = 1)
    leaf_nodes = [node for node, degree in G.degree() if degree == 1]

    for leaf in leaf_nodes:
        leaf_point = Point(leaf)

        # Find nearest point on polygon boundary
        nearest_point = boundary_line.interpolate(boundary_line.project(leaf_point))
        nearest_coords = (nearest_point.x, nearest_point.y)

        # Add edge to the graph
        G.add_edge(leaf, nearest_coords, weight=leaf_point.distance(nearest_point))

    return G

if __name__ == "__main__":
    pass