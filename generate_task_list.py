import numpy as np


def generate_task_lists(T, n, k):
    lists = []
    for _ in range(k):
        # Generate n random values
        random_values = np.random.rand(int(n))
        
        # Normalize the values so that their sum is T
        normalized_values = (random_values / np.sum(random_values)) * T
        
        # Append the list of tasks to the output
        lists.append(normalized_values.tolist())
    
    return lists


if __name__ == "__main__":

    # Example usage
    T = 10.0  # total time
    n = 5     # number of tasks
    k = 3     # number of samples

    task_lists = generate_task_lists(T, n, k)
    for i, task_list in enumerate(task_lists, 1):
        print(f"Sample {i}: {task_list}, Sum: {sum(task_list)}")