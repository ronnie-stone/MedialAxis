import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis, binary_closing, disk
from skimage.measure import label

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
    Computes the medial axis of a binary image with boundary refinement.
    """
    ## Apply closing again if needed to remove boundary artifacts
    #closed_image = binary_closing(binary_image, disk(1))

    # Compute medial axis
    skeleton, distance = medial_axis(binary_image, return_distance=True)

    return skeleton

def plot_binary_and_medial_axis(binary_image, medial_axis_image, title=""):
    plt.figure(figsize=(10, 5))

    # Binary Image
    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray', origin='lower')
    plt.title(f"{title} - Binary Image")
    plt.axis('off')

    # Medial Axis
    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray', alpha=0.4, origin='lower')
    plt.imshow(medial_axis_image, cmap='hot', alpha=0.8)
    plt.title(f"{title} - Medial Axis")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage with different polygons
if __name__ == "__main__":
    square_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    bunny_coords = np.load('bunny_cross_section_scaled.npy')

    # Star Example (More complex shape)
    star_coords = [(50, 90), (60, 60), (90, 50), (60, 40), (50, 10), (40, 40), (10, 50), (40, 60), (50, 90)]

    for shape_name, coords in [("Shape1", square_coords), ("Shape2", bunny_coords)]:
        binary_img = polygon_to_binary_image(coords, img_size=200)
        medial_axis_img = compute_medial_axis(binary_img)
        plot_binary_and_medial_axis(binary_img, medial_axis_img, title=shape_name)
