import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import geopandas as gpd
from shapely.geometry import MultiLineString, LineString, Point
from get_medial_axis_voronoi import get_medial_axis
from shapely.ops import unary_union
from medial_axis_to_graph import medial_axis_to_graph
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph


def enlarge_medial_axis(medial_axis, radius_map, alpha=0.2, resolution=10):
    """
    Enlarges the medial axis dynamically using a fraction of the local radius.

    Parameters:
        medial_axis (MultiLineString or LineString): The medial axis geometry.
        radius_map (dict): Mapping from medial axis points to their local radius.
        alpha (float): Fraction of the local radius to use for enlargement.
        resolution (int): Number of segments used for buffering (higher = smoother).

    Returns:
        Polygon: The enlarged medial axis as a smooth region.
    """
    if not isinstance(medial_axis, (LineString, MultiLineString)):
        raise ValueError("medial_axis must be a Shapely LineString or MultiLineString.")
    
    if alpha == 0:
        return medial_axis

    enlarged_shapes = []

    # Convert medial axis to list of individual segments
    if isinstance(medial_axis, LineString):
        medial_axis = [medial_axis]  # Make it iterable

    for line in medial_axis.geoms if isinstance(medial_axis, MultiLineString) else medial_axis:
        coords = list(line.coords)
        segment_buffers = []

        # Buffer each segment dynamically based on radius
        for i in range(len(coords) - 1):
            p1, p2 = coords[i], coords[i + 1]
            r1, r2 = radius_map.get(p1, 0), radius_map.get(p2, 0)

            # Compute local buffer size (smooth interpolation of local radius)
            buffer_size = alpha * ((r1 + r2) / 2.0)

            segment = LineString([p1, p2])
            segment_buffers.append(segment.buffer(buffer_size, resolution=resolution))

        # Union all segment buffers to avoid overlap artifacts
        enlarged_shape = segment_buffers[0]
        for buf in segment_buffers[1:]:
            enlarged_shape = enlarged_shape.union(buf)

        enlarged_shapes.append(enlarged_shape)

    # Merge all enlarged shapes
    return unary_union(enlarged_shapes)

if __name__ == "__main__":
    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    input_polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    #input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 

    # Get medial axis:

    medial_axis, radius_map = get_medial_axis(input_polygon_coords, num_samples=1000)

    # Apply the enlargement function
    alpha = 0.5  # Fraction of local radius to enlarge
    enlarged_medial_axis = enlarge_medial_axis(medial_axis, radius_map, alpha=alpha)
    boundary = LineString(enlarged_medial_axis.exterior)
    
    # Convert to graph:

    G = medial_axis_to_graph(boundary, radius_map)

    print(G.number_of_nodes())

    # Plot results

    fig, ax = plt.subplots(1, 1, figsize=(16, 6))

    # Print out radius values for each node

    plot_medial_axis_vs_graph(input_polygon_coords, medial_axis, G, ax)

    # Plot original medial axis
    #gpd.GeoSeries([medial_axis]).plot(ax=ax, color='blue', linewidth=2, label="Original Medial Axis")

    # Plot enlarged medial axis region
    gpd.GeoSeries([enlarged_medial_axis]).plot(ax=ax, color='red', alpha=alpha, label="Enlarged Region")

    plt.legend()
    plt.title("Medial Axis Enlargement Using Local Radius")
    plt.show()
