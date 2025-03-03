import numpy as np


def binary_to_polygon_coords(pixel_coords, original_polygon_coords, img_size=200, padding=5):
    """
    Converts pixel coordinates back to original floating-point polygon coordinates.

    Parameters:
        pixel_coords (list of tuples): List of (x, y) points in the binary image (pixel space).
        original_polygon_coords (list of tuples): Original polygon coordinates before transformation.
        img_size (int, optional): Size of the image used in polygon_to_binary_image. Default is 200.
        padding (int, optional): Padding used in polygon_to_binary_image. Default is 5.

    Returns:
        list of tuples: List of (x, y) points mapped back to the original floating-point space.
    """
    # Extract original min/max values
    x_orig, y_orig = zip(*original_polygon_coords)
    x_min, y_min = np.min(x_orig), np.min(y_orig)
    x_range = max(np.max(x_orig) - x_min, np.max(y_orig) - y_min)  # Maintain aspect ratio

    # Reverse the transformation
    scale = x_range / (img_size - 2 * padding)
    
    # Convert pixel coordinates back to float values
    original_coords = [
        ((x - padding) * scale + x_min, (y - padding) * scale + y_min) 
        for x, y in pixel_coords
    ]
    
    return original_coords

