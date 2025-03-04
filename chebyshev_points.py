from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from get_medial_axis_voronoi import get_medial_axis
from medial_axis_to_graph import medial_axis_to_graph


def compute_largest_inscribed_circle_radius(polygon, point):
    """
    Compute the largest inscribed circle radius centered at 'point' inside 'polygon'.
    
    Parameters:
        polygon (shapely.geometry.Polygon): The input polygon.
        point (tuple): (x, y) coordinates of the center.
    
    Returns:
        float: The maximum radius before the circle touches the boundary.
    """
    center = Point(point)
    nearest_boundary_point = nearest_points(polygon.boundary, center)[0]
    return center.distance(nearest_boundary_point)


def find_heaviest_branching_point(G, polygon):
    """
    Find the heaviest branching point in the medial axis graph.
    This is the point where the largest inscribed circle fits.
    In case of ties (multiple points with the same max radius), choose the one closest to the centroid.

    Parameters:
        G (networkx.Graph): The medial axis graph.
        polygon (shapely.geometry.Polygon): The input polygon.
    
    Returns:
        tuple: (heaviest_branching_point, max_radius, index_in_G)
    """
    branching_points = [node for node, degree in G.degree() if degree >= 3]
    
    # Compute polygon centroid
    centroid = np.array(polygon.centroid.coords[0])

    max_radius = 0
    candidates = []

    radii = {}

    for idx, branch in enumerate(branching_points):
        radius = compute_largest_inscribed_circle_radius(polygon, branch)
        radii[branch] = radius
        if radius > max_radius:
            max_radius = radius
            candidates = [(idx, branch)]
        elif np.isclose(radius, max_radius):
            candidates.append((idx, branch))

    if len(candidates) == 1:
        selected_idx, selected_branch = candidates[0]
        return selected_branch, max_radius, selected_idx

    # Tie-break by distance to centroid
    def distance_to_centroid(pt):
        return np.linalg.norm(np.array(pt) - centroid)

    selected_idx, selected_branch = min(candidates, key=lambda x: distance_to_centroid(x[1]))

    return selected_branch, max_radius, selected_idx



if __name__ == "__main__":

    # Define a polygon (for example, a square)
    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy')
    polygon = Polygon(input_polygon_coords)

    medial_axis = get_medial_axis(input_polygon_coords, num_samples=1000)
    G = medial_axis_to_graph(medial_axis)

    heaviest_branch, max_radius, heaviest_idx = find_heaviest_branching_point(G, polygon)
    print(f"Heaviest branching point: {heaviest_branch}, Radius: {max_radius:.3f}" , f"Index in G: {heaviest_idx}")

    # Plot the polygon and medial axis
    fig, ax = plt.subplots(figsize=(8, 8))

    # Polygon boundary
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label="Polygon")

    # Medial axis graph
    for u, v in G.edges():
        x, y = zip(u, v)
        ax.plot(x, y, 'b--', linewidth=1.5)

    # Plot branching points
    branching_points = [node for node, degree in G.degree() if degree >= 3]
    for bp in branching_points:
        ax.scatter(*bp, color='purple', s=100, edgecolors='black', label="Branching Points" if bp == branching_points[0] else "")

    # Plot the heaviest branching point
    ax.scatter(*heaviest_branch, color='red', s=150, edgecolors='black', label="Heaviest Branching Point")

    # Draw the largest inscribed circle
    circle = plt.Circle(heaviest_branch, max_radius, color='red', fill=False, linestyle='dashed', linewidth=2, label="Largest Inscribed Ball")
    ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.legend()
    ax.set_title("Medial Axis with Largest Inscribed Ball")
    plt.show()