import numpy as np


def expand_adjacency_matrix(adjacency_matrix):
    """
    Function to expand the adjacency matrix based on the rule: A_new = [0 I; I A_old].
    
    Parameters: 
    adjacency_matrix (np.ndarray): The original adjacency matrix to expand.

    Returns: 
    np.ndarray: The expanded adjacency matrix.
    """
    
    n = adjacency_matrix.shape[0]  # Number of original nodes
    new_size = 2 * n  # New size after expansion
    
    # Initialize the new matrix with zeros
    new_matrix = np.zeros((new_size, new_size), dtype=int)
    
    # Fill in the identity matrices in the top-right and bottom-left quadrants
    new_matrix[:n, n:] = np.eye(n, dtype=int)  # Top-right quadrant
    new_matrix[n:, :n] = np.eye(n, dtype=int)  # Bottom-left quadrant
    
    # Copy the original adjacency matrix into the bottom-right quadrant
    new_matrix[n:, n:] = adjacency_matrix
    
    return new_matrix

if __name__ == "__main__":

    adj_m = np.array([[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0], [1, 1, 0, 0]])
    expanded_adj_m = expand_adjacency_matrix(adj_m)
    print(expanded_adj_m)