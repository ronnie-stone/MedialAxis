import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiLineString, LineString
import numpy as np


def plot_medial_axis_vs_graph(polygon_coords, medial_axis, G, ax, selected_branches=None):
    """
    Plots the polygon, medial axis (direct from Shapely), and medial axis reconstructed as a graph.
    
    Parameters:
        polygon_coords (list of (x, y)): The polygon boundary coordinates.
        medial_axis (Shapely MultiLineString or LineString): The computed medial axis.
        G (networkx.Graph): The graph representation of the medial axis.
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        selected_branches (list of (x, y), optional): Coordinates of selected branching points to highlight.
    """
    polygon = Polygon(polygon_coords)
    x, y = polygon.exterior.xy

    ax.plot(x, y, 'k-', linewidth=2, label='Polygon')

    # Ground truth - direct medial axis (Shapely)
    if isinstance(medial_axis, MultiLineString):
        for line in medial_axis.geoms:
            x, y = line.xy
            ax.plot(x, y, 'r-', linewidth=2, label='Medial Axis (Direct)' if 'Medial Axis (Direct)' not in ax.get_legend_handles_labels()[1] else None)
    elif isinstance(medial_axis, LineString):
        x, y = medial_axis.xy
        ax.plot(x, y, 'r-', linewidth=2, label='Medial Axis (Direct)')

    ## Reconstructed medial axis - from graph
    #for u, v in G.edges():
    #    x, y = zip(*[u, v])
    #    ax.plot(x, y, 'b--', linewidth=1.5, label='Medial Axis (Graph)' if 'Medial Axis (Graph)' not in ax.get_legend_handles_labels()[1] else None)

    # Plot all graph nodes
    all_nodes = np.array(G.nodes)
    ax.scatter(all_nodes[:, 0], all_nodes[:, 1], c='gray', s=20, label='Graph Nodes', zorder=3)

    # Plot branching nodes (degree ≥ 3) in purple
    branching_nodes = [node for node, degree in G.degree() if degree >= 3]
    if branching_nodes:
        branching_nodes = np.array(branching_nodes)
        ax.scatter(branching_nodes[:, 0], branching_nodes[:, 1], c='purple', s=50, label='Branching Nodes (deg ≥ 3)', edgecolor='black', zorder=4)

    # Optional: Highlight selected branches (e.g., from optimizer)
    if selected_branches is not None and len(selected_branches) > 0:
        selected_branches = np.array(selected_branches)  # Ensure array for plotting
        ax.scatter(selected_branches[:, 0], selected_branches[:, 1], c='green', s=60, label='Selected Branches', edgecolor='black', zorder=5)

    ax.legend()
    ax.set_title("Medial Axis Comparison: Direct vs Graph")
    ax.set_aspect('equal')

