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
import warnings

# Silence only PyGAD warnings
warnings.filterwarnings("ignore", module="pygad")

def run_partition_optimization(G, n_divisions, input_polygon, objective_function):
    branching_nodes = [node for node, degree in G.degree() if degree >= 3]
    branching_nodes = list(G.nodes)

    print(len(branching_nodes))

    if len(branching_nodes) < n_divisions - 1:
        raise ValueError(f"Not enough branching nodes for {n_divisions} partitions.")

    num_genes = n_divisions

    # Prepare the gene space (each gene can be any branching node)
    gene_space = list(range(len(branching_nodes)))  # Genes are just indices into branching_nodes

    def fitness_func(ga_instance, solution, solution_idx):
        return objective_function(solution, input_polygon, G)

    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        fitness_func=fitness_func,
        sol_per_pop=30,
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
    print(f"Best branching points: {best_solution}, Score: {-best_solution_fitness:.4f}")

    return tuple(best_solution)

# ----------------- Example Objective Function (Placeholder) -----------------

def calculate_area_imbalance_l2(selected_branches, input_polygon, G):
    """
    Given the polygon and the list of partition areas, compute L2 norm of area imbalance.
    Parameters:
        polygon_coords (list of (x, y)): Original polygon coordinates.
        areas (list of float): Areas of the resulting partitions.
    Returns:
        float: L2 norm of area imbalance (lower is better).
    """
    buffer_size = 0.1
    printing_polygon = create_polygon(input_polygon)
    total_area = printing_polygon.area

    all_nodes = list(G.nodes)
    selected_branches = np.array([all_nodes[int(idx)] for idx in selected_branches])

    _, _, _, _, _, areas = tessellate_with_buffer(selected_branches, input_polygon, buffer_size)
    n_parts = len(areas)
    ideal_area = total_area / n_parts
    areas = np.array(areas)
    l2_norm = np.linalg.norm(areas - ideal_area)
    return -l2_norm

if __name__ == "__main__":

    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 
    buffer_size = 0.05

    # Get medial axis:

    medial_axis = get_medial_axis(input_polygon_coords, num_samples=1000)

    # Transform the medial axis into a graph:

    G = medial_axis_to_graph(medial_axis)

    # Augment the graph to connect the one-degree vertices to the boundary:

    #G = connect_limbs_to_boundary(G, input_polygon_coords)

    best_solution = run_partition_optimization(G, 6, input_polygon_coords, calculate_area_imbalance_l2)

    all_nodes = list(G.nodes)
    selected_branches = np.array([all_nodes[int(idx)] for idx in best_solution])

    # Plotting:

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_medial_axis_vs_graph(input_polygon_coords, medial_axis, G, axes[0])

    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(selected_branches, input_polygon_coords, buffer_size)
    plot_custom_solution(selected_branches.flatten().tolist(), polygons_A_star, polygons_B, 0, 0, axes[1])
    plt.show()





