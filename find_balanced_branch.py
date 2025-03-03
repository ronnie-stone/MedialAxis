import networkx as nx
import numpy as np
from skimage.morphology import binary_dilation, disk
from skimage.measure import label


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

def find_balanced_branch(binary_image, G):
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


if __name__ == "__main__":
    pass