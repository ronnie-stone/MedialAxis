import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Voronoi
from shapely.ops import unary_union
from shapely import geometry, buffer
from shapely.geometry import Polygon
from create_polygon import create_polygon
from plot_custom_solution import plot_custom_solution


def tessellate_with_buffer(robot_positions, printing_part, buffer_size):

    minx = min(robot_positions[:,0])
    maxx = max(robot_positions[:,0])
    miny = min(robot_positions[:,1])
    maxy = max(robot_positions[:,1])
    
    helper_points = np.array([[minx - 100, miny - 100], [maxx + 100, miny - 100], [maxx + 100, maxy + 100], [minx - 100, maxy + 100]])
    
    points = np.vstack((robot_positions, helper_points))
    
    vor = Voronoi(points)
    vertices = vor.vertices
    regions = vor.regions
    regions_index = vor.point_region
    
    printing_polygon = create_polygon(printing_part)
    
    a_regions = []
    b_regions = []
    a_star_regions = []

    a_regions_areas = []
    b_regions_areas = []
    a_star_regions_areas = []

    enlarged_a_regions = []


    # Get A regions from tessellation:

    for i in range(len(robot_positions)):

        cell = vertices[regions[regions_index[i]], :]
        unbounded_a_region = geometry.Polygon(cell)
        bounded_a_region = unbounded_a_region.intersection(printing_polygon)
        a_regions.append(bounded_a_region)
        a_regions_areas.append(bounded_a_region.area)
        enlarged_a_region = buffer(bounded_a_region, buffer_size, quad_segs=8)
        enlarged_a_regions.append(enlarged_a_region)

    # Get buffer regions by intersecting the bounded A regions with the union of the remaining enlarged regions

    for j in range(len(robot_positions)):
        other_enlarged_a_regions = [enlarged_a_regions[k] for k in range(len(enlarged_a_regions)) if k != j]
        union_of_other_enlarged_a_regions = unary_union(other_enlarged_a_regions)
        b_region = a_regions[j].intersection(union_of_other_enlarged_a_regions)
        b_regions.append(b_region)
        b_regions_areas.append(b_region.area)

        if a_regions[j].within(b_region):
            a_star_regions.append(Polygon())
            a_star_regions_areas.append(0)
        
        else:
            a_star_regions.append(a_regions[j].difference(b_region))
            a_star_regions_areas.append(a_star_regions[j].area)

    # return MainPolygons, BoundaryPolygons, MainAreas, BoundaryAreas
    return a_star_regions, b_regions, a_regions, a_star_regions_areas, b_regions_areas, a_regions_areas


if __name__ == "__main__":

    # Voronoi Sites:

    # solution = np.array([[1.31561465, 1.59776848], [1.01787483, 1.44344593], [1.13758911, 1.29757163], [1.13394131, 1.24222339]])
    solution = np.array([[0.25, 0.25], [0.25, 2.75], [2.75, 2.75], [2.75, 0.25]])
    # solution = np.array([[2.94481052,  3.16981612], [-0.23659766, -0.22294067], [ 2.68116008, -0.12173115], [0.98704001,  3.06411647]])
    solution = np.array([[-0.48313575, 2.29844539],  [2.83358478,  2.77404733],  [0.29216143,  0.08333269], [3.46014425,  0.83023325]])


    # Parameters:

    minimum_robot_distance = 0.5
    reachability_radius = 2.0
    buffer_size = 0.2

    # Printing Part Geometry:

    # printing_part = [(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)]
    # printing_part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    # printing_part = np.array([(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)])
    # priting_part = [[(0,0), (3,0), (3,3), (0,3), (0,0)], [(1, 1), (1.5, 1), (1.5,1.5), (1, 1.5), (1, 1)]]

    # Circle:
    radius = 1
    center = np.array([1.5, 1.5])
    num_points = 1000
    angles = np.linspace(0, 2 * np.pi, num_points)
    x_points = radius * np.cos(angles) + center[0]
    y_points = radius * np.sin(angles) + center[1]
    printing_part = np.column_stack((x_points, y_points)) 

    # Calling function:

    print(type(solution))
    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(solution, printing_part, buffer_size)

    # Printing Results:

    print(polygons_A_star)
    print(polygons_A_star_areas)
    print(polygons_B_areas)

    # Plotting:

    solution = solution.flatten().tolist()
    fig, ax = plt.subplots()
    plot_custom_solution(solution, polygons_A_star, polygons_B, minimum_robot_distance, reachability_radius, ax)
    plt.show()