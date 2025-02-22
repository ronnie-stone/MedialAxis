import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis

def polygon_to_binary_image(polygon_coords, img_size=200):
    """
    Converts a polygon given by a list of coordinates to a binary image.
    
    Args:
        polygon_coords: List of (x, y) tuples defining the polygon.
        img_size: Size of the square image.
        
    Returns:
        Binary image with the polygon filled in.
    """
    # Create an empty binary image
    binary_image = np.zeros((img_size, img_size), dtype=bool)

    # Convert polygon coordinates to NumPy arrays
    x, y = zip(*polygon_coords)
    
    # Scale coordinates to fit within the image size
    x = np.array(x) - min(x)
    y = np.array(y) - min(y)

    x = (x / max(x) * (img_size - 1)).astype(int)
    y = (y / max(y) * (img_size - 1)).astype(int)

    # Fill the polygon using rasterization
    rr, cc = polygon(y, x, shape=binary_image.shape)
    binary_image[rr, cc] = True

    return binary_image

def compute_medial_axis(binary_image):
    """
    Computes the medial axis of a binary image.
    
    Args:
        binary_image: 2D numpy array where True represents the polygon.
        
    Returns:
        Skeletonized image (medial axis).
    """
    skeleton, _ = medial_axis(binary_image, return_distance=True)
    return skeleton

def extract_skeleton_coordinates(skeleton_image):
    """
    Extracts the coordinates of the medial axis from the binary skeleton image.
    
    Args:
        skeleton_image: 2D numpy array representing the medial axis.
        
    Returns:
        List of (x, y) coordinates corresponding to the medial axis.
    """
    y, x = np.where(skeleton_image)  # Extract nonzero (skeleton) pixels
    return list(zip(x, y))

# Example Polygon (Star Shape)
polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
#polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
#polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
#polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
#polygon_coords = np.load('bunny_cross_section_scaled.npy')

# Convert polygon to binary image
binary_img = polygon_to_binary_image(polygon_coords, img_size=2000)

# Compute medial axis
medial_axis_img = compute_medial_axis(binary_img)

# Extract coordinates of the medial axis
skeleton_coords = extract_skeleton_coordinates(medial_axis_img)

# Plot the results
plt.figure(figsize=(8, 6))
plt.imshow(binary_img, cmap='gray', alpha=0.6)
plt.imshow(medial_axis_img, cmap='hot', alpha=0.8)
plt.scatter(*zip(*skeleton_coords), color='red', s=1)  # Show skeleton points
plt.title("Medial Axis of Polygon")
plt.gca().invert_yaxis()
plt.show()

# Print extracted medial axis coordinates
#print("Medial Axis Coordinates:", skeleton_coords)
