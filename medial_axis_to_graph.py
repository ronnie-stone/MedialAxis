import networkx as nx
import numpy as np
from shapely.geometry import MultiLineString, LineString
import matplotlib.pyplot as plt
from get_medial_axis_voronoi import get_medial_axis
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph


def medial_axis_to_graph(medial_axis, radius_map):
    """
    Converts a Shapely medial axis (LineString or MultiLineString) into a NetworkX graph,
    incorporating radius information as node attributes.

    Parameters:
        medial_axis (LineString or MultiLineString): The medial axis geometry.
        radius_map (dict): Mapping from medial axis points to their distance to the boundary.

    Returns:
        networkx.Graph: The graph representation of the medial axis with radius information.
    """
    G = nx.Graph()

    if isinstance(medial_axis, LineString):
        coords = list(medial_axis.coords)
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            G.add_edge(p1, p2, weight=LineString([p1, p2]).length)
        
        # Store radius information as node attributes
        for p in coords:
            if tuple(p) in radius_map:
                G.nodes[p]['radius'] = radius_map[tuple(p)]

    elif isinstance(medial_axis, MultiLineString):
        for line in medial_axis.geoms:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = coords[i]
                p2 = coords[i + 1]
                G.add_edge(p1, p2, weight=LineString([p1, p2]).length)
            
            # Store radius information as node attributes
            for p in coords:
                if tuple(p) in radius_map:
                    G.nodes[p]['radius'] = radius_map[tuple(p)]
    
    return G

if __name__ == "__main__":
    # polygon_coords = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]  # Square
    # polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)]  # Rectangle
    # polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)]  # Triangle
    # polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    # polygon_coords = np.load('bunny_cross_section_scaled.npy')

    medial_axis, radius_function = get_medial_axis(polygon_coords)
    G = medial_axis_to_graph(medial_axis, radius_function)

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Print out radius values for each node
    print("Medial Axis Node Radii:")
    for node, data in G.nodes(data=True):
        print(f"Point: {node}, Radius: {data['radius']}")

    plot_medial_axis_vs_graph(polygon_coords, medial_axis, G, ax)
    plt.show()

