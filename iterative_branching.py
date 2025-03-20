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
from shapely.geometry import MultiLineString, LineString, Point
from enlarge_medial_axis import enlarge_medial_axis
from adjacency_matrix_from_regions import adjacency_matrix_from_regions
from expand_adjacency_matrix import expand_adjacency_matrix
from adjacency_matrix_to_connected_tasks import adjacency_matrix_to_connected_tasks
from integer_lp import task_scheduling_ilp
from rank_nodes import rank_nodes
from plot_selected_nodes import plot_selected_nodes

def plot_ranking_histogram(ranked_nodes_history, benchmark_value):
    """
    Plots ranking scores of nodes across iterations using a bar chart with improved color contrast.
    
    Parameters:
        ranked_nodes_history (list of list of tuples): 
            A list where each element is a list of ranked nodes (node, score) for each iteration.
        benchmark_value (float): The score from the first iteration, used as the benchmark.
    """
    plt.figure(figsize=(10, 6))
    colors = plt.cm.Oranges(np.linspace(0.3, 1, len(ranked_nodes_history)))  # From light to dark orange
    
    for i, ranked_nodes in enumerate(ranked_nodes_history):
        # Sort nodes by ranking scores
        ranked_nodes_sorted = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)
        sorted_node_indices = list(range(len(ranked_nodes_sorted)))
        scores = [score for _, score in ranked_nodes_sorted]
        
        # Plot bars with no borders for better clarity
        plt.bar(sorted_node_indices, scores, color=colors[i], alpha=0.6, label=f'Iteration {i+1}')
    
    # Benchmark horizontal line
    plt.axhline(y=benchmark_value, color='red', linestyle='--', linewidth=2, label=fr'Benchmark Value: $f(v_1)$')
    
    # Labels and legend
    plt.xlim(0, 800)
    plt.title('Ranking Scores of Nodes Across Iterations')
    plt.xlabel('Ranked Node Index', fontsize=14)
    plt.ylabel('Ranking Score', fontsize=14)
    #plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

"""

def plot_ranking_histogram(ranked_nodes, benchmark_value):
    # Sort the ranked nodes in decreasing order of the ranking scores
    ranked_nodes_sorted = sorted(ranked_nodes, key=lambda x: x[1], reverse=True)
    
    # Extract sorted node indices and their corresponding ranking scores
    sorted_node_indices = list(range(len(ranked_nodes_sorted)))  # This will be the x-axis, sorted by rank
    scores = [score for _, score in ranked_nodes_sorted]  # Ranking scores
    
    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_node_indices, scores, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add horizontal line at the benchmark value (first iteration score)
    plt.axhline(y=benchmark_value, color='red', linestyle='--', label=f'Benchmark Value: {benchmark_value:.4f}')
    
    # Set title and labels
    plt.title('Ranking Scores of Nodes')
    plt.xlabel('Ranked Node Index')
    plt.ylabel('Ranking Score')
    
    # Add grid and legend
    plt.grid(True, axis='y')
    #plt.legend()
    
    # Display the plot
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

"""




def iterative_ranking(G, beta=0.1, num_iter=10):
    """
    Iteratively ranks and selects nodes based on the ranking function until
    the ratio of the selected node scores between two iterations falls below the threshold (beta),
    or the maximum number of iterations (num_iter) is exceeded.
    
    Parameters:
        G (networkx.Graph): The graph representation of the medial axis.
        beta (float): The threshold value to stop the iteration if the ratio of the scores falls below it.
        num_iter (int): The maximum number of iterations to perform before stopping.
    
    Returns:
        selected_nodes (list): List of selected nodes in order of selection.
        selection_history (list): History of selected nodes and their scores.
        final_ranked_nodes (list): List of all ranked nodes in the final iteration.
    """
    selected_nodes = []
    selection_history = []
    final_ranked_nodes = []
    ranked_nodes_history = []
    
    # Run the first iteration and initialize the benchmark value (first node score)
    best_node, best_value, ranked_nodes = rank_nodes(G, set(selected_nodes))
    final_ranked_nodes = ranked_nodes
    
    # Initialize benchmark value as the score of the first selected node
    benchmark_value = best_value
    
    # Append the first selection
    selected_nodes.append(best_node)
    selection_history.append((best_node, best_value))
    ranked_nodes_history.append(final_ranked_nodes)
    
    prev_best_value = best_value  # For comparison in next iterations
    
    num_iterations = 1  # Start counting from the first iteration
    
    while num_iterations <= num_iter:
        # Rank nodes based on current selection
        best_node, best_value, ranked_nodes = rank_nodes(G, set(selected_nodes))

        # Storing the ranked nodes of the current iteration
        final_ranked_nodes = ranked_nodes
        
        # Stopping criterion: check if the ratio of best_value to benchmark_value is below beta
        ratio = best_value / benchmark_value
        if ratio < beta:
            break
        
        # Append the best node and its score to history
        selected_nodes.append(best_node)
        selection_history.append((best_node, best_value))
        ranked_nodes_history.append(final_ranked_nodes)
        
        prev_best_value = best_value  # Update the previous best value for the next iteration
        
        num_iterations += 1  # Increment the iteration count
    
    return selected_nodes, selection_history, final_ranked_nodes, ranked_nodes_history

if __name__ == "__main__":

    # Iterative approach to the paper. 

    # Start from getting the medial axis and the graph representation:

    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (2, 0.5), (2, 2.5), (3, 2.5), (3, 3), (1, 3), (1, 0.5), (0, 0.5), (0,0)] # WEIRD-BEAM
    #input_polygon_coords = np.load('circle_scaled.npy') 
    #input_polygon_coords = np.load('bunny_cross_section_scaled.npy')
    input_polygon_coords = np.load('BenchyPoints_scaled.npy') 
    #input_polygon_coords = np.load('HorsePoints_scaled.npy') 
    #input_polygon_coords = np.load('TeaPotPoints_scaled.npy') 
    buffer_size = 0.001

    # Get medial axis:

    medial_axis, radius_map = get_medial_axis(input_polygon_coords, num_samples=1000)

    # Apply the enlargement function
    alpha = 1  # Fraction of local radius to enlarge
    enlarged_medial_axis = enlarge_medial_axis(medial_axis, radius_map, alpha=alpha)

    if alpha != 0:
        boundary = LineString(enlarged_medial_axis.exterior)
    else: 
        boundary = medial_axis

    # Transform the medial axis into a graph:

    G = medial_axis_to_graph(medial_axis, radius_map)

    for v, data in G.nodes(data=True):
        if 'radius' not in data:
            print(f"Warning: Node {v} is missing 'radius' attribute!")
        else:
            radius_v = data['radius']

    # Now run the iterative ranking algorithm
    beta_threshold = 0.2  # Stopping criterion
    selected_nodes, selection_history, final_ranked_nodes, ranked_nodes_history = iterative_ranking(G, beta=beta_threshold, num_iter=5)

    # Output results
    print("\nSelected Nodes after Iterative Ranking:", selected_nodes)
    print("Selection History:")
    for node, score in selection_history:
        print(f"Node {node} selected with score {score:.4f}")

    plot_selected_nodes(input_polygon_coords, medial_axis, selected_nodes, G)

    # Plot histogram of ranking scores for all evaluated nodes in the final iteration
    #print(final_ranked_nodes)
    #plot_ranking_histogram(ranked_nodes_history, selection_history[0][1])

    # Get the degree of each selected node and the sum of all degrees
    total_degree = 0
    for node in selected_nodes:
        degree = G.degree(node)  # Get degree of the node
        print(f"Node {node} has degree {degree}")
        total_degree += degree
    
    print(f"\nTotal sum of degrees for selected nodes: {total_degree}")

