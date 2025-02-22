import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis
import networkx as nx

def polygon_to_binary_image(polygon_coords, img_size=200):
    """
    Converts a polygon given by a list of coordinates to a binary image.
    """
    binary_image = np.zeros((img_size, img_size), dtype=bool)
    x, y = zip(*polygon_coords)
    x = np.array(x) - min(x)
    y = np.array(y) - min(y)
    x = (x / max(x) * (img_size - 1)).astype(int)
    y = (y / max(y) * (img_size - 1)).astype(int)
    rr, cc = polygon(y, x, shape=binary_image.shape)
    binary_image[rr, cc] = True
    return binary_image

def compute_medial_axis(binary_image):
    """
    Computes the medial axis of a binary image.
    """
    skeleton, _ = medial_axis(binary_image, return_distance=True)
    return skeleton

def skeleton_to_graph(skeleton):
    """
    Converts a skeleton image into a graph.
    Each skeleton pixel is a node, and edges connect neighboring pixels.
    """
    G = nx.Graph()
    rows, cols = np.where(skeleton)
    for y, x in zip(rows, cols):
        G.add_node((x, y))
        # 8-connectivity
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < skeleton.shape[1] and 0 <= ny_ < skeleton.shape[0]:
                    if skeleton[ny_, nx_]:
                        G.add_edge((x, y), (nx_, ny_), weight=1)
    return G

def find_longest_path(G):
    """
    Finds the longest path in the skeleton graph using Dijkstra's algorithm.
    """
    # Find endpoints (nodes with degree 1)
    endpoints = [node for node, degree in G.degree() if degree == 1]

    # Find longest shortest path among endpoints
    longest_path = []
    max_length = 0
    for i in range(len(endpoints)):
        for j in range(i+1, len(endpoints)):
            try:
                path = nx.shortest_path(G, endpoints[i], endpoints[j], weight='weight')
                if len(path) > max_length:
                    max_length = len(path)
                    longest_path = path
            except nx.NetworkXNoPath:
                continue
    return longest_path

# Example Polygon (Star Shape)
polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
#polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
#polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
#polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
#polygon_coords = np.load('bunny_cross_section_scaled.npy')

# Step 1: Rasterize polygon
binary_img = polygon_to_binary_image(polygon_coords, img_size=1000)

# Step 2: Compute medial axis
medial_axis_img = compute_medial_axis(binary_img)

# Step 3: Convert skeleton to graph
G = skeleton_to_graph(medial_axis_img)

# Step 4: Find longest path
longest_path = find_longest_path(G)

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
plt.imshow(medial_axis_img, cmap='hot', alpha=0.8, origin='lower')

# Plot the longest path
if longest_path:
    path_x, path_y = zip(*longest_path)
    plt.plot(path_x, path_y, color='blue', linewidth=2, label='Longest Path')

plt.title("Medial Axis with Longest Subgraph")
plt.legend()
plt.show()

# Print the longest path length and coordinates
print(f"Longest Path Length: {len(longest_path)}")
print("Longest Path Coordinates:", longest_path)
