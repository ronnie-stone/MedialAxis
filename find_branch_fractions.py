import numpy as np


def find_branch_fractions(branch, N, shift=0):
    """
    Finds points at 1/N, 2/N, ..., (N-1)/N along the given branch and computes their normal vectors.
    Optionally shifts the fraction points along the normal direction in an alternating fashion.

    Parameters:
        branch (list of tuples): List of (x, y) points representing the branch.
        N (int): The number of divisions.
        shift (float, optional): The distance to shift fraction points along their normal vectors.
                                 Points alternate directions. Default is 0 (no shift).

    Returns:
        tuple:
            - List of (x, y) points at the required fractions of the branch length.
            - List of (nx, ny) unit normal vectors at those points.
    """
    if len(branch) < 2:
        raise ValueError("Branch is too short to divide.")

    if N < 2:
        raise ValueError("N must be at least 2 to define fractions.")

    # Compute cumulative distances along the branch
    cumulative_distances = [0]
    for i in range(1, len(branch)):
        x0, y0 = branch[i - 1]
        x1, y1 = branch[i]
        dist = np.hypot(x1 - x0, y1 - y0)
        cumulative_distances.append(cumulative_distances[-1] + dist)

    total_length = cumulative_distances[-1]
    target_distances = [(i / N) * total_length for i in range(1, N)]

    # Find closest points to each fractional length
    fraction_points = []
    normals = []
    index = 0  # Start search from the beginning of the branch

    for target_dist in target_distances:
        while index < len(cumulative_distances) and cumulative_distances[index] < target_dist:
            index += 1

        # Ensure we don't go out of bounds
        if index >= len(branch):
            break

        # Selected fractional point
        fraction_point = branch[index]
        fraction_points.append(fraction_point)

        # Compute tangent vector
        if 0 < index < len(branch) - 1:
            # Use neighbors to estimate direction
            x_prev, y_prev = branch[index - 1]
            x_next, y_next = branch[index + 1]
        elif index == 0:
            # Use only forward difference for the first point
            x_prev, y_prev = fraction_point
            x_next, y_next = branch[index + 1]
        else:
            # Use only backward difference for the last point
            x_prev, y_prev = branch[index - 1]
            x_next, y_next = fraction_point

        dx = x_next - x_prev
        dy = y_next - y_prev
        magnitude = np.hypot(dx, dy)

        if magnitude == 0:
            normal_vector = (0, 0)  # Avoid division by zero
        else:
            # Compute normal as (-dy, dx) and normalize
            nx = -dy / magnitude
            ny = dx / magnitude
            normal_vector = (nx, ny)

        normals.append(normal_vector)

    # Apply alternating shift if shift is nonzero
    if shift != 0:
        print("here")
        for i in range(len(fraction_points)):
            x, y = fraction_points[i]
            nx, ny = normals[i]

            # Alternate shift direction
            direction = 1 if i % 2 == 0 else -1
            print(fraction_points[i])
            fraction_points[i] = (x + direction * shift * nx, y + direction * shift * ny)
            print(fraction_points[i])

    return fraction_points, normals