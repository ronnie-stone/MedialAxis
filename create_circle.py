import numpy as np

def generate_circle(num_points=100, center=(1.5, 1.5), radius=1.5):
    """
    Generates a circular shape centered at 'center' with given 'radius'.

    Parameters:
        num_points (int): Number of points in the circle.
        center (tuple): (x, y) coordinates of the circle's center.
        radius (float): Radius of the circle.

    Returns:
        np.ndarray: Array of shape (num_points, 2) containing (x, y) coordinates.
    """
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=True)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))

# Generate and save the circle
circle_coords = generate_circle(num_points=200)
np.save("circle_scaled.npy", circle_coords)

print("Circle saved as 'circle_scaled.npy'")
