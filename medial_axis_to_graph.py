import networkx as nx
import numpy as np
from shapely.geometry import MultiLineString, LineString, Polygon
import matplotlib.pyplot as plt
from get_medial_axis_voronoi import get_medial_axis
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph


def medial_axis_to_graph(medial_axis):
    """
    Converts a Shapely medial axis (LineString or MultiLineString) into a NetworkX graph.

    Parameters:
        medial_axis (LineString or MultiLineString): The medial axis geometry.

    Returns:
        networkx.Graph: The graph representation of the medial axis.
    """
    G = nx.Graph()

    if isinstance(medial_axis, LineString):
        # A single line segment (degenerate case, e.g., triangle medial axis)
        coords = list(medial_axis.coords)
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            G.add_edge(p1, p2, weight=LineString([p1, p2]).length)

    elif isinstance(medial_axis, MultiLineString):
        # Multiple line segments (typical case)
        for line in medial_axis.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i + 1]
                G.add_edge(p1, p2, weight=LineString([p1, p2]).length)

    return G


if __name__ == "__main__":
    polygon_coords = [
        (0, 0), (6, 0), (6, 3), (3, 6), (0, 3), (0, 0)
    ]

    # Uncomment to load bunny (use your file)
    polygon_coords = np.load('bunny_cross_section_scaled.npy')

    medial_axis = get_medial_axis(polygon_coords)
    G = medial_axis_to_graph(medial_axis)

    plot_medial_axis_vs_graph(polygon_coords, medial_axis, G)
