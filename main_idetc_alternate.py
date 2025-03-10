import matplotlib.pyplot as plt
import numpy as np
import pygad
from get_medial_axis_voronoi import get_medial_axis
from medial_axis_to_graph import medial_axis_to_graph
from connect_limbs_to_boundary import connect_limbs_to_boundary
from plot_medial_axis_vs_graph import plot_medial_axis_vs_graph
from tessellate_with_buffer import tessellate_with_buffer
from plot_custom_solution import plot_custom_solution
from create_polygon import create_polygon
from chebyshev_points import find_heaviest_branching_point  
from computer_bisectors_at_branch import compute_bisectors_at_branch
from idetc_main_alg import find_best_partition_and_nodes
import warnings
warnings.filterwarnings("ignore", module="pygad")
import networkx as nx


def run_partition_optimization(G, n_divisions, input_polygon, all_bisectors, objective_function, box_value=0.5):
    """
    Runs a Genetic Algorithm to optimize partitioning.
    Calls find_best_partition_and_nodes within the GA fitness function.
    """

    # Step 3: Genetic Algorithm setup
    num_genes = len(all_bisectors)
    gene_space = [{'low': 0.01, 'high': 1.0} for _ in range(num_genes)]

    def fitness_func(ga_instance, solution, solution_idx):
        voronoi_sites = []
        for i, (bp, bisector) in enumerate(all_bisectors):
            t = solution[i]
            site = np.array(bp) + t * bisector * box_value
            voronoi_sites.append(site)

        voronoi_sites = np.array(voronoi_sites)
        return objective_function(voronoi_sites, input_polygon)

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=20,
        fitness_func=fitness_func,
        sol_per_pop=50,
        num_genes=num_genes,
        gene_space=gene_space,
        mutation_type="random",
        mutation_num_genes=1,
        mutation_by_replacement=True,
        crossover_type="single_point",
        parent_selection_type="tournament"
    )

    ga_instance.run()

    best_solution, best_solution_fitness, _ = ga_instance.best_solution()
    print(f"Best normalized distances: {best_solution}, Score: {-best_solution_fitness:.4f}")

    # Step 4: Convert final normalized distances into Voronoi site coordinates
    best_sites = []
    for i, (bp, bisector) in enumerate(all_bisectors):
        t = best_solution[i]
        site = np.array(bp) + t * bisector * box_value
        best_sites.append(site)

    return np.array(best_sites)

# ----------------- Example Objective Function (Placeholder) -----------------

def calculate_area_imbalance_l2(selected_sites, input_polygon):
    """
    Given the polygon and the list of partition areas, compute normalized L2 norm of area imbalance.
    Parameters:
        selected_sites (np.array): Voronoi sites.
        input_polygon (list of (x, y)): Original polygon coordinates.
        G (networkx.Graph): Medial axis graph (not used here, but may be needed for more complex partitions).
    Returns:
        float: Negative L2 norm of area imbalance (lower imbalance is better, so this is a maximization problem in GA).
    """
    buffer_size = 0.1
    printing_polygon = create_polygon(input_polygon)
    total_area = printing_polygon.area

    _, _, _, _, _, areas = tessellate_with_buffer(selected_sites, input_polygon, buffer_size)

    areas = np.array(areas)
    n_parts = len(areas)

    normalized_areas = areas / total_area
    ideal_fraction = 1.0 / n_parts

    l2_norm = np.linalg.norm(normalized_areas - ideal_fraction)

    return -l2_norm  # Negative because PyGAD maximizes


if __name__ == "__main__":

    # Input polygon (choose one)
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 
    buffer_size = 0.05

    # Step 1: Compute the Medial Axis and Transform to a Graph
    medial_axis, radius_map = get_medial_axis(input_polygon_coords, num_samples=1000)
    G = medial_axis_to_graph(medial_axis, radius_map)

    # Step 2: Define the number of robots and find the best partition
    N = 4  # Number of robots
    best_partition, best_nodes, best_assignments = find_best_partition_and_nodes(G, N, alpha=1.0)

    print("Best Partition:", best_partition)
    print("Best Nodes:", best_nodes)
    print("Best Assignments:", best_assignments)

    # Step 3: Compute bisectors for each selected node
    all_bisectors = []
    for node in best_nodes:
        cluster_size = best_assignments[node]  # Get the assigned cluster size
        bisectors = compute_bisectors_at_branch(G, node, n_bisectors=cluster_size)  # Compute bisectors

        # Store each bisector along with its originating node
        all_bisectors.extend([(node, bis) for bis in bisectors])

    # Step 4: Run partition optimization
    best_solution = run_partition_optimization(G, N, input_polygon_coords, all_bisectors, calculate_area_imbalance_l2)

    # Step 5: Create subplots for visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- Figure 1: Graph Visualization ----
    pos = {node: node for node in G.nodes}
    degrees = dict(G.degree())

    # Draw the medial axis graph
    nx.draw(G, pos, with_labels=False, node_color='lightgray', edge_color='gray', node_size=25, ax=axes[0])
    nx.draw_networkx_nodes(G, pos, nodelist=best_nodes, node_color='red', node_size=100, ax=axes[0])

    # Annotate selected nodes with cluster size
    for node in best_nodes:
        x, y = pos[node]
        axes[0].text(x, y + 0.2, f"cluster={best_assignments[node]}", fontsize=10, ha='center', color='black')

    axes[0].set_title("Medial Axis Graph with Selected Nodes")

    # Plot the medial axis edges
    for u, v in G.edges():
        x, y = zip(*[u, v])
        axes[0].plot(x, y, 'k-', linewidth=2)

    # Plot the bisectors from branching points
    for i, (branching_point, bisector) in enumerate(all_bisectors):
        end = np.array(branching_point) + bisector  # Bisectors are now scaled by radius
        axes[0].plot([branching_point[0], end[0]], [branching_point[1], end[1]], 
                     linestyle='--', linewidth=2, label=f'Bisector {i+1}')

    # ---- Figure 2: Tessellation Visualization ----
    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(best_solution, input_polygon_coords, buffer_size)

    print("A Areas: " + str(polygons_A_areas))

    plot_custom_solution(best_solution.flatten().tolist(), polygons_A_star, polygons_B, 0, 0, ax=axes[1])
    axes[1].set_title("Tessellation Results")

    # Show the plots
    plt.show()
