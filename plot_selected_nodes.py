import matplotlib.pyplot as plt
from shapely.geometry import Polygon, LineString, MultiLineString

def plot_selected_nodes(polygon_coords, medial_axis, selected_nodes, G, num_iterations=5):
    """
    Plots the polygon, medial axis, and inscribed circles of selected nodes, 
    with colors indicating their selection iteration based on HSV colormap.

    Parameters:
        polygon_coords (list of (x, y)): The polygon boundary coordinates.
        medial_axis (Shapely MultiLineString or LineString): The computed medial axis.
        selected_nodes (list of sets): A list of sets, each containing nodes selected in a particular iteration.
        G (networkx.Graph): The graph representation of the medial axis.
        num_iterations (int): The number of iterations used for selection (for coloring).
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot polygon boundary
    polygon = Polygon(polygon_coords)
    x, y = polygon.exterior.xy
    ax.plot(x, y, 'k-', linewidth=2, label='Polygon')

    # Plot medial axis
    if isinstance(medial_axis, MultiLineString):
        for line in medial_axis.geoms:
            x, y = line.xy
            ax.plot(x, y, 'r-', linewidth=2, label='Medial Axis' if 'Medial Axis' not in ax.get_legend_handles_labels()[1] else None)
    elif isinstance(medial_axis, LineString):
        x, y = medial_axis.xy
        ax.plot(x, y, 'r-', linewidth=2, label='Medial Axis')

    # HSV colormap
    cmap = plt.get_cmap('twilight')

    print(selected_nodes)

    # Plot selected nodes and their inscribed circles with iteration-based coloring
    for iteration, nodes_in_iteration in enumerate(selected_nodes):
        # Node coordinates are stored as the index (x, y)
        x, y = nodes_in_iteration  # Unpack coordinates directly from the node index (as it’s a tuple)

        # Retrieve the node's radius from the attributes
        radius = G.nodes[nodes_in_iteration]['radius']  # Radius is stored as an attribute

        # Generate a color based on the iteration using the colormap
        color = cmap(iteration / num_iterations)

        # Plot selected node with the assigned color (avoid repeating labels in legend)
        ax.scatter(x, y, color=color, s=50, zorder=2, label=f'Node {iteration + 1}')

        # Plot inscribed circle
        circle = plt.Circle((x, y), radius, color=color, fill=False, linestyle='dashed', alpha=1.0, zorder=1)
        ax.add_patch(circle)

    # Handle the legend for selected nodes and the polygon
    ax.legend(loc='upper left')

    ax.set_aspect('equal')
    ax.set_title("Selected Nodes and Inscribed Circles")

    plt.show()


