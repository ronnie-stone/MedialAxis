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

def get_exterior_medial_axis(polygon_coords, num_samples=200):
    """
    Computes the exterior medial axis of a polygon using the Voronoi diagram
    of equidistant points sampled along its perimeter, keeping only edges
    that are entirely outside the polygon.
    
    Parameters:
        polygon_coords (list of (x, y)): Polygon vertices.
        num_samples (int): Number of total samples.
    
    Returns:
        exterior_medial_axis (Shapely geometry): The exterior medial axis.
        radius_map (dict): Mapping from exterior medial axis points to their distance to the boundary.
    """
    polygon = Polygon(polygon_coords)
    
    # Sample points along the perimeter
    sampled_points = sample_polygon_perimeter(polygon_coords, num_samples)
    
    # Compute Voronoi diagram
    vor = Voronoi(sampled_points)
    
    # Collect Voronoi edges that are entirely outside the polygon
    lines = []
    radius_map = {}
    
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue  # Skip infinite edges

        p1, p2 = vor.vertices[vpair]
        line = LineString([p1, p2])

        # Keep this ridge only if both endpoints are outside the polygon
        if not polygon.contains(Point(p1)) and not polygon.contains(Point(p2)):
            lines.append(line)
            
            # Compute the distance of each medial axis vertex to the polygon boundary
            d1 = polygon.boundary.distance(Point(p1))
            d2 = polygon.boundary.distance(Point(p2))
            
            # Store these distances in the radius map
            radius_map[tuple(p1)] = d1
            radius_map[tuple(p2)] = d2

    # Combine all external segments into a single medial axis geometry
    exterior_medial_axis = unary_union(lines)
    
    return exterior_medial_axis, radius_map

if __name__ == "__main__":
    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    #input_polygon_coords = np.load('circle_scaled.npy') 
    #input_polygon_coords = np.load('bunny_cross_section_scaled.npy')
    #input_polygon_coords = np.load('BenchyPoints_scaled.npy') 
    #input_polygon_coords = np.load('HorsePoints_scaled.npy') 
    #input_polygon_coords = np.load('TeaPotPoints_scaled.npy') 

    num_samples = 100
    
    # Compute exterior medial axis
    exterior_medial_axis, radius_map = get_exterior_medial_axis(input_polygon_coords, num_samples=num_samples)
    
    # Plot results
    polygon = Polygon(input_polygon_coords)
    x, y = polygon.exterior.xy
    
    plt.figure(figsize=(10, 8))
    
    # Plot polygon boundary
    plt.plot(x, y, 'k-', linewidth=2, label='Polygon')

    # Compute Voronoi diagram for visualization
    sampled_points = sample_polygon_perimeter(input_polygon_coords, num_samples)
    vor = Voronoi(sampled_points)
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, show_points=False, line_colors='gray', line_width=0.5)
    
    # Plot exterior medial axis
    if isinstance(exterior_medial_axis, MultiLineString):
        for line in exterior_medial_axis.geoms:
            x, y = line.xy
            plt.plot(x, y, 'b-', linewidth=2, label='Exterior Medial Axis' if 'Exterior Medial Axis' not in plt.gca().get_legend_handles_labels()[1] else None)
    elif isinstance(exterior_medial_axis, LineString):
        x, y = exterior_medial_axis.xy
        plt.plot(x, y, 'b-', linewidth=2, label='Exterior Medial Axis')
    
    plt.legend()
    plt.xlim(-5, 8)
    plt.ylim(-5, 8)

    plt.show()
