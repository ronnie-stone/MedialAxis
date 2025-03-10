import numpy as np
from scipy.spatial import Voronoi
from shapely import geometry
from create_polygon import create_polygon


def get_A_regions(voronoi_points, input_polygon):

    minx = min(voronoi_points[:, 0])
    maxx = max(voronoi_points[:, 0])
    miny = min(voronoi_points[:, 1])
    maxy = max(voronoi_points[:, 1])
    
    boundary_points = np.array([[minx - 100, miny - 100], [maxx + 100, miny - 100], [maxx + 100, maxy + 100], [minx - 100, maxy + 100]])
    
    points = np.vstack((voronoi_points, boundary_points))
    
    vor = Voronoi(points)
    vertices = vor.vertices
    regions = vor.regions
    regions_index = vor.point_region

    
    
    printing_polygon = create_polygon(input_polygon)

    polygons = []
    polygons_areas = []
    
    for i in range(len(voronoi_points[:,1])):

        cell = vertices[regions[regions_index[i]], :]
        cell_polygon = geometry.Polygon(cell)
        polygon = cell_polygon.intersection(printing_polygon)
        polygons.append(polygon)
        polygons_areas.append(polygon.area)

    return polygons, polygons_areas, vor

if __name__ == "__main__":
    pass
