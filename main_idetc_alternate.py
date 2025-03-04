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
import warnings
warnings.filterwarnings("ignore", module="pygad")

def solve_partition_equation(n_divisions):
    """
    Solve 3a + 2b = n_divisions for non-negative integers a, b.
    The goal is to maximize a (maximize the number of 3-bisector branching points).

    Parameters:
        n_divisions (int): Desired number of partitions.

    Returns:
        (a, b): Number of degree-3 and degree-2 branching points.
    """
    max_a = n_divisions // 3  # Upper bound on number of 3-bisector branches.
    
    for a in range(max_a, -1, -1):
        remaining = n_divisions - 3 * a
        if remaining % 2 == 0:
            b = remaining // 2
            return a, b
    
    raise ValueError(f"No valid integer solution for 3a + 2b = {n_divisions}.")

def run_partition_optimization(G, n_divisions, input_polygon, objective_function, box_value=0.1):
    """
    Genetic algorithm to find optimal Voronoi site placements along bisectors of the heaviest branch(es).

    Parameters:
        G (networkx.Graph): The medial axis graph.
        n_divisions (int): Number of partitions.
        input_polygon (list): Polygon coordinates.
        objective_function (callable): Objective function taking (Voronoi sites, input polygon).

    Returns:
        np.ndarray: Array of Voronoi site coordinates.
    """
    if n_divisions < 3:
        raise ValueError("n_divisions must be at least 3.")

    polygon = create_polygon(input_polygon)

    # Step 1: Solve for (a, b) where 3a + 2b = n_divisions
    a, b = solve_partition_equation(n_divisions)

    print(f"n_divisions={n_divisions} => Using {a} degree-3 branches and {b} degree-2 branches.")

    # Step 2: Identify heaviest branches (we need exactly a + b branching points)
    selected_branches = []

    tempG = G.copy()  # Work on a temporary copy to avoid modifying the original graph

    for _ in range(a + b):
        heaviest_branch, max_radius, idx = find_heaviest_branching_point(tempG, polygon)

        selected_branches.append((heaviest_branch, max_radius))

        # Remove the selected branch from consideration for the next iteration
        tempG.remove_node(heaviest_branch)

    # Step 3: Compute bisectors based on degree-3 or degree-2 requirement
    all_bisectors = []
    for i, (branch, _) in enumerate(selected_branches):
        if i < a:  # First 'a' branches use 3 bisectors
            bisectors = compute_bisectors_at_branch(G, branch, n_bisectors=3)
        else:      # Remaining 'b' branches use 2 bisectors
            bisectors = compute_bisectors_at_branch(G, branch, n_bisectors=2)

        all_bisectors.extend([(branch, bis) for bis in bisectors])

    if len(all_bisectors) != n_divisions:
        raise ValueError(f"Expected exactly {n_divisions} bisectors, but found {len(all_bisectors)}")

    # Step 4: Genetic Algorithm setup — each gene is a normalized distance along its bisector
    num_genes = n_divisions
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
        sol_per_pop=100,
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

    # Step 5: Convert final normalized distances into Voronoi site coordinates
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

    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 
    buffer_size = 0.05

    # Get medial axis:

    medial_axis = get_medial_axis(input_polygon_coords, num_samples=1000)

    # Transform the medial axis into a graph:

    G = medial_axis_to_graph(medial_axis)

    # Augment the graph to connect the one-degree vertices to the boundary:

    #G = connect_limbs_to_boundary(G, input_polygon_coords)

    best_solution = run_partition_optimization(G, 5, input_polygon_coords, calculate_area_imbalance_l2)

    # Plotting:

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_medial_axis_vs_graph(input_polygon_coords, medial_axis, G, axes[0])

    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(best_solution, input_polygon_coords, buffer_size)

    print("A Areas: " + str(polygons_A_areas))
    plot_custom_solution(best_solution.flatten().tolist(), polygons_A_star, polygons_B, 0, 0, axes[1])
    plt.show()