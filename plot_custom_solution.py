import matplotlib.pyplot as plt
import numpy as np 
from shapely import MultiPolygon
from scipy.spatial import Voronoi, voronoi_plot_2d
import os


def plot_custom_solution(solution, polygons_A_star, polygons_B, minimum_robot_distance, reachability_radius, ax, foldername=None):

    # Use the current directory if foldername is None
    if foldername is None:
        foldername = os.getcwd()  # Get the current working directory

    # Ensure the folder exists, if not, create it
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Plotting:
    cmap = plt.get_cmap('hsv')

    # Example color indices for polygons. Adjust based on your number of polygons.
    num_polygons_A_star = len(polygons_A_star)
    num_polygons_B = len(polygons_B)
    colors_A_star = [cmap(i / num_polygons_A_star) for i in range(num_polygons_A_star)]
    colors_B = [cmap(i / num_polygons_B) for i in range(num_polygons_B)]

    #plot_solution(solution, ax, num_generations)

    for i, MainPolygon in enumerate(polygons_A_star):
        color = colors_A_star[i]
        if isinstance(MainPolygon, MultiPolygon):
            for poly in MainPolygon.geoms:
                x, y = poly.exterior.xy
                # Plot the filled polygon with alpha for the fill
                ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
                # Plot the outline of the same polygon with no alpha (fully opaque)
                ax.plot(x, y, color='black', linewidth=1)
        else:
            x, y = MainPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.1, edgecolor='none')
            ax.plot(x, y, color='black', linewidth=1)

    # Plot polygons_B in the same way
    for i, BoundaryPolygon in enumerate(polygons_B):
        color = colors_B[i]
        if isinstance(BoundaryPolygon, MultiPolygon):
            label = False
            for poly in BoundaryPolygon.geoms:
                x, y = poly.exterior.xy
                if not label:
                    ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none', label=str(i))
                    label = True
                else:
                    ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none')
                ax.plot(x, y, color='black', linewidth=1)
        else:
            x, y = BoundaryPolygon.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.4, edgecolor='none', label=str(i))
            ax.plot(x, y, color='black', linewidth=1)

    voronoi_points = [[solution[i], solution[i+1]] for i in range(0, len(solution), 2)]
    helper_points = [[-100,-100], [-100,100], [100,-100], [100,100]]
    print(voronoi_points)
    points = voronoi_points + helper_points
    points = np.array(points)

    #fixed_points = np.array([[0.25, 2.75], [2.75, 2.75], [0.25, 0.25], [2.75, 0.25]])
    #ax.scatter(np.array(fixed_points)[:, 0],  np.array(fixed_points)[:, 1], s=20, color='red', edgecolor='black')
    vor = Voronoi(points)
    voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False, line_colors='black')
    ax.scatter(np.array(voronoi_points)[:, 0],  np.array(voronoi_points)[:, 1], s=20, color='red', edgecolor='black')
    ax.set_aspect('equal')
    plt.xlim([-0.15,3.15])
    plt.ylim([-0.15,3.15])
    #plt.legend()
    #filepath1 = os.path.join(foldername, "Fig1.pdf")
    #plt.savefig(filepath1, bbox_inches='tight', format='pdf')

if __name__ == "__main__":
    pass