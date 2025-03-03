import numpy as np
from skimage.draw import polygon
from skimage.morphology import binary_closing, disk
import matplotlib.pyplot as plt 


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

if __name__ == "__main__":

    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 

    # Get binary image:

    binary_image = polygon_to_binary_image(input_polygon_coords, img_size=1000)

    plt.figure(figsize=(10, 5))
    plt.imshow(binary_image, cmap='gray', origin='lower')
    plt.title("Binary Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()