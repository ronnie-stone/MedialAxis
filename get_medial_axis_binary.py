import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis, binary_closing, disk


def polygon_to_binary_image(polygon_coords, img_size=200, padding=5):
    """
    Converts polygon coordinates into a binary image, ensuring boundaries are handled correctly.
    """
    # Convert to numpy array
    x, y = zip(*polygon_coords)
    x, y = np.array(x), np.array(y)

    # Normalize coordinates to fit within the image
    x -= np.min(x)
    y -= np.min(y)
    scale = (img_size - 2 * padding) / max(np.max(x), np.max(y))
    x = (x * scale + padding).astype(int)
    y = (y * scale + padding).astype(int)

    # Create binary image
    binary_image = np.zeros((img_size, img_size), dtype=bool)
    rr, cc = polygon(y, x, shape=binary_image.shape)
    binary_image[rr, cc] = True

    # Apply closing to ensure solid regions and fix thin boundaries
    binary_image = binary_closing(binary_image, disk(1))

    return binary_image

def compute_medial_axis(binary_image):
    """
    Computes the medial axis of a binary image.
    
    Args:
        binary_image: 2D numpy array where True represents the polygon.
        
    Returns:
        Skeletonized image (medial axis).
    """
    skeleton = medial_axis(binary_image)
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
#polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
#polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
#polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
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
