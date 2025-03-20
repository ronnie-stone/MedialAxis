import numpy as np

def scale_to_range(polygon_coords, new_range=(0, 3)):
    """
    Scales the coordinates of a polygon to the new range [0, 3].
    
    Parameters:
        polygon_coords (numpy.ndarray): The array of shape (N, 2) with (x, y) coordinates.
        new_range (tuple): The desired range, by default (0, 3).
    
    Returns:
        scaled_coords (numpy.ndarray): Scaled (x, y) coordinates within the new range.
    """
    # Convert input to numpy array if not already
    polygon_coords = np.array(polygon_coords)
    
    # Find the min and max values for both x and y coordinates
    min_vals = polygon_coords.min(axis=0)
    max_vals = polygon_coords.max(axis=0)
    
    # Scale the coordinates to the new range [0, 3]
    scaled_coords = (polygon_coords - min_vals) / (max_vals - min_vals) * (new_range[1] - new_range[0])
    
    return scaled_coords

# Example Usage:
polygon_coords = np.load('BenchyPoints.npy')  # Load the original file

scaled_coords = scale_to_range(polygon_coords, new_range=(0, 3))

# Save the scaled coordinates to a new .npy file
np.save('BenchyPoints_scaled.npy', scaled_coords)

print("Scaled coordinates saved to 'scaled_HorsePoints.npy'.")