from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import matplotlib.pyplot as plt

def create_polygon(coordinates):
    # If coordinates are not nested (no holes), create a simple polygon
    if isinstance(coordinates[0][0], (int, float)):  
        return Polygon(shell=coordinates)
    
    # If coordinates are nested, assume the first list is the outer boundary,
    # and the remaining lists are holes
    elif isinstance(coordinates[0][0], (list, tuple)):
        outer_boundary = coordinates[0]  # First element is the outer boundary
        holes = coordinates[1:]  # Remaining elements are holes
        return Polygon(shell=outer_boundary, holes=holes)
    
    else:
        raise ValueError("Invalid polygon coordinates format")

def plot_polygon(polygon):
    # Plot the outer boundary
    outer_x, outer_y = polygon.exterior.xy
    plt.plot(outer_x, outer_y, color='blue')

    # Plot the holes, if any
    for interior in polygon.interiors:
        hole_x, hole_y = interior.xy
        plt.plot(hole_x, hole_y, color='red')

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    input_polygon = [
        [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)],  # Outer boundary
        [(1, 1), (1.5, 1), (1.5, 1.5), (1, 1.5), (1, 1)]  # Hole
    ]

    # Create the polygon with the function
    polygon = create_polygon(input_polygon)

    # Plot the polygon for confirmation
    plot_polygon(polygon)
    print(polygon.area)