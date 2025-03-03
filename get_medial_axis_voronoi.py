import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
from scipy.spatial import Voronoi, voronoi_plot_2d


def sample_polygon_perimeter(polygon_coords, num_samples):
    """
    Sample approximately equidistant points along polygon perimeter,
    correctly rolling over across edges when needed.

    Parameters:
        polygon_coords (list of (x, y)): Polygon vertices in order.
        num_samples (int): Total number of samples.

    Returns:
        np.ndarray: Sampled (x, y) points.
    """
    polygon = Polygon(polygon_coords)
    perimeter = polygon.length

    # Desired spacing between points
    spacing = perimeter / num_samples

    sampled_points = []

    # Walk along the polygon perimeter, placing points at fixed intervals
    current_distance = 0  # Total perimeter distance walked so far
    edge_index = 0

    while len(sampled_points) < num_samples:
        p1 = np.array(polygon_coords[edge_index])
        p2 = np.array(polygon_coords[(edge_index + 1) % len(polygon_coords)])

        edge_length = np.linalg.norm(p2 - p1)

        if current_distance + spacing <= edge_length:
            # Sample point within this edge
            t = (current_distance + spacing) / edge_length
            point = (1 - t) * p1 + t * p2
            sampled_points.append(point)
            current_distance += spacing
        else:
            # Move to next edge and carry over the remaining distance
            leftover = edge_length - current_distance
            current_distance = -leftover
            edge_index = (edge_index + 1) % len(polygon_coords)

    return np.array(sampled_points)



def get_medial_axis(polygon_coords, num_samples=200):
    """
    Computes approximate medial axis of a polygon using Voronoi diagram
    of equidistant points sampled along its perimeter, keeping only edges
    that lie entirely within the polygon.

    Parameters:
        polygon_coords (list of (x, y)): Polygon vertices.
        num_samples (int): Number of total samples.

    Returns:
        Shapely geometry representing the medial axis (unary union of lines).
    """
    polygon = Polygon(polygon_coords)

    # Sample points along the perimeter (uniform sampling)
    sampled_points = sample_polygon_perimeter(polygon_coords, num_samples)

    # Compute Voronoi diagram
    vor = Voronoi(sampled_points)

    # Collect Voronoi edges that lie entirely inside the polygon
    lines = []
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue  # Skip infinite edges

        p1, p2 = vor.vertices[vpair]
        line = LineString([p1, p2])

        # Only keep this ridge if **both endpoints lie inside** the polygon
        if polygon.contains(Point(p1)) and polygon.contains(Point(p2)):
            lines.append(line)

    # Combine all internal segments into a single medial axis geometry
    medial_axis = unary_union(lines)

    return medial_axis


if __name__ == "__main__":
    # polygon_coords = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]  # Square
    # polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)]  # Rectangle
    # polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)]  # Triangle
    polygon_coords = np.load('bunny_cross_section_scaled.npy')

    num_samples = 1000  # Set number of samples along perimeter

    # Compute medial axis (polygon-internal Voronoi edges)
    medial_axis = get_medial_axis(polygon_coords, num_samples=num_samples)

    # Compute Voronoi diagram for the sampled points (the real sites used for medial axis)
    sampled_points = sample_polygon_perimeter(polygon_coords, num_samples)
    vor = Voronoi(sampled_points)

    # Plot everything
    polygon = Polygon(polygon_coords)
    x, y = polygon.exterior.xy

    plt.figure(figsize=(10, 8))

    # Plot polygon boundary
    plt.plot(x, y, 'k-', linewidth=2, label='Polygon')

    # Plot sampled points (the Voronoi sites)
    sampled_x, sampled_y = sampled_points[:, 0], sampled_points[:, 1]
    plt.scatter(sampled_x, sampled_y, c='blue', s=10, label='Sampled Points (Sites)')

    # Plot full Voronoi diagram (includes external edges)
    #voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, show_points=False, line_colors='gray', line_width=0.5)

    # Plot medial axis (filtered Voronoi edges inside polygon)
    if isinstance(medial_axis, MultiLineString):
        for line in medial_axis.geoms:
            x, y = line.xy
            plt.plot(x, y, 'r-', linewidth=2, label='Medial Axis' if 'Medial Axis' not in plt.gca().get_legend_handles_labels()[1] else None)
    elif isinstance(medial_axis, LineString):
        x, y = medial_axis.xy
        plt.plot(x, y, 'r-', linewidth=2, label='Medial Axis')

    plt.legend()
    plt.title("Polygon, Sampled Points, Voronoi Diagram, and Medial Axis")
    plt.show()