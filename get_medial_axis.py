import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Voronoi


def get_medial_axis(polygon_coords):
    # Create Shapely polygon
    polygon = Polygon(polygon_coords)
    
    # Sample points densely along the edges
    sampled_points = []
    for i in range(len(polygon_coords)):
        p1 = np.array(polygon_coords[i])
        p2 = np.array(polygon_coords[(i+1) % len(polygon_coords)])
        # Sample 10 points per edge
        for t in np.linspace(0, 1, 500):
            sampled_points.append(p1 * (1 - t) + p2 * t)
    sampled_points = np.array(sampled_points)
    
    # Compute Voronoi diagram
    vor = Voronoi(sampled_points)
    
    # Collect Voronoi edges inside the polygon
    lines = []
    for vpair in vor.ridge_vertices:
        if -1 in vpair:
            continue  # Ignore infinite edges
        p1, p2 = vor.vertices[vpair]
        line = LineString([p1, p2])
        if polygon.contains(line):
            lines.append(line)

    # Union and polygonize for cleaner structure
    medial = unary_union(lines)

    return medial

# Example usage
polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
polygon_coords = np.load('bunny_cross_section_scaled.npy')
medial_axis = get_medial_axis(polygon_coords)

# Plot
polygon = Polygon(polygon_coords)
x, y = polygon.exterior.xy
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'b-', label='Polygon')
if isinstance(medial_axis, MultiLineString):
    for line in medial_axis.geoms:
        x, y = line.xy
        plt.plot(x, y, 'r-')
elif isinstance(medial_axis, LineString):
    x, y = medial_axis.xy
    plt.plot(x, y, 'r-')
plt.legend(['Polygon', 'Medial Axis'])
plt.title("Medial Axis of Polygon")
plt.show()