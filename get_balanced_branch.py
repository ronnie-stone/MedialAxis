import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis, binary_dilation, disk
from skimage.measure import label
import networkx as nx

def polygon_to_binary_image(polygon_coords, img_size=200):
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
    skeleton, _ = medial_axis(binary_image, return_distance=True)
    return skeleton

def skeleton_to_graph(skeleton):
    G = nx.Graph()
    rows, cols = np.where(skeleton)
    for y, x in zip(rows, cols):
        G.add_node((x, y))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx_, ny_ = x + dx, y + dy
                if 0 <= nx_ < skeleton.shape[1] and 0 <= ny_ < skeleton.shape[0]:
                    if skeleton[ny_, nx_]:
                        G.add_edge((x, y), (nx_, ny_), weight=1)
    return G

def find_boundary_nodes(binary_image, G):
    boundary_nodes = []
    for node in G.nodes:
        x, y = node
        # Check for adjacency to background pixels
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < binary_image.shape[1] and 0 <= ny_ < binary_image.shape[0]:
                if not binary_image[ny_, nx_]:
                    boundary_nodes.append(node)
                    break
    return boundary_nodes

def compute_area(binary_image):
    return np.sum(binary_image)

def evaluate_cut(binary_image, cut_path):
    """
    Evaluates the area difference after cutting along the given path.
    Uses dilation to ensure the cut fully separates the polygon.
    """
    # Copy binary image
    mask = np.copy(binary_image)

    # Apply the cut by setting cut pixels to False
    for x, y in cut_path:
        mask[y, x] = False

    # Dilate the cut to ensure full disconnection
    dilated_cut = np.zeros_like(mask)
    for x, y in cut_path:
        dilated_cut[y, x] = True
    dilated_cut = binary_dilation(dilated_cut, disk(2))  # Dilation with radius 2

    # Apply the dilated cut
    mask[dilated_cut] = False

    # Label connected components
    labeled_img = label(mask, connectivity=2)
    unique_labels = np.unique(labeled_img)

    # Expect two regions (excluding background)
    if len(unique_labels) < 3:  # Background + one region is invalid
        return float('inf'), None

    # Compute areas of the two regions
    areas = []
    for label_val in unique_labels[1:]:  # Skip background
        areas.append(np.sum(labeled_img == label_val))

    if len(areas) != 2:
        return float('inf'), None  # Invalid split

    area_diff = abs(areas[0] - areas[1])
    return area_diff, areas

def find_optimal_cut(binary_image, skeleton, G):
    boundary_nodes = find_boundary_nodes(binary_image, G)
    best_cut = None
    best_diff = float('inf')
    best_areas = None

    # Try all boundary-to-boundary paths
    for i in range(len(boundary_nodes)):
        for j in range(i+1, len(boundary_nodes)):
            try:
                path = nx.shortest_path(G, boundary_nodes[i], boundary_nodes[j], weight='weight')
                area_diff, areas = evaluate_cut(binary_image, path)
                if area_diff < best_diff:
                    best_diff = area_diff
                    best_cut = path
                    best_areas = areas
            except nx.NetworkXNoPath:
                continue

    return best_cut, best_areas, best_diff

# Example Polygon (Star Shape)
polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
polygon_coords = np.load('bunny_cross_section_scaled.npy')

# Rasterize Polygon
binary_img = polygon_to_binary_image(polygon_coords, img_size=200)

# Compute Medial Axis
medial_axis_img = compute_medial_axis(binary_img)

# Convert Skeleton to Graph
G = skeleton_to_graph(medial_axis_img)

# Find the Optimal Cut
optimal_cut, cut_areas, area_diff = find_optimal_cut(binary_img, medial_axis_img, G)

# Plot the Result
plt.figure(figsize=(8, 6))
plt.imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
plt.imshow(medial_axis_img, cmap='hot', alpha=0.8, origin='lower')

# Plot Optimal Cut
if optimal_cut:
    x_vals, y_vals = zip(*optimal_cut)
    plt.plot(x_vals, y_vals, color='blue', linewidth=3, label='Optimal Cut')

plt.title(f"Optimal Cut (Area Difference: {area_diff})")
plt.legend()
plt.show()

# Print the results
print(f"Optimal Cut Path: {optimal_cut}")
print(f"Areas After Cut: {cut_areas}")
print(f"Area Difference: {area_diff}")
