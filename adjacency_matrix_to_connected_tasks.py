import numpy as np


def adjacency_matrix_to_connected_tasks(adjacency_matrix):
    """
    Find and return a list of connected tasks based on the adjacency matrix.
    
    Args: 
    adjacency_matrix (np.ndarray): A square adjacency matrix where a non-zero entry indicates a connection between tasks.

    Returns:
    list: A list of tuples, where each tuple (i, j) indicates that task i is connected to task j.
    """

    # Sanity check: ensure the matrix is square
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    
    # Sanity check: ensure the matrix is symmetric
    if not np.allclose(adjacency_matrix, adjacency_matrix.T):
        raise ValueError("Adjacency matrix must be symmetric.")
    
    # Sanity check: ensure the diagonal elements are all zero
    if not np.all(np.diag(adjacency_matrix) == 0):
        raise ValueError("Adjacency matrix diagonal elements must be zero (no self-connections).")
    
    n = adjacency_matrix.shape[0]  # Number of tasks (size of the matrix)
    connected_tasks = []

    # Loop over the upper triangle of the matrix to avoid duplicates
    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i, j] != 0 or adjacency_matrix[j, i] != 0:
                connected_tasks.append((i, j))

    return connected_tasks

if __name__ == "__main__":

    adj_matrix_1 = np.array([[0, 1], [1, 0]])
    connected_tasks_1 = adjacency_matrix_to_connected_tasks(adj_matrix_1)
    print(connected_tasks_1)  # Output: [(0, 1)]

    adj_matrix_2 = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    connected_tasks_2 = adjacency_matrix_to_connected_tasks(adj_matrix_2)
    print(connected_tasks_2)  # Output: [(0, 1), (0, 2)]

