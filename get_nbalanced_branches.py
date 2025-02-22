import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.morphology import medial_axis, binary_dilation, binary_closing, disk
from skimage.measure import label
import networkx as nx

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
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx_, ny_ = x + dx, y + dy
            if 0 <= nx_ < binary_image.shape[1] and 0 <= ny_ < binary_image.shape[0]:
                if not binary_image[ny_, nx_]:
                    boundary_nodes.append(node)
                    break
    return boundary_nodes

def compute_area(binary_image):
    return np.sum(binary_image)

def evaluate_cut(binary_image, cut_path, target_ratio, total_area):
    """
    Evaluates the area difference after cutting along the given path,
    with respect to a target area split ratio.
    """
    mask = np.copy(binary_image)
    for x, y in cut_path:
        mask[y, x] = False

    dilated_cut = np.zeros_like(mask)
    for x, y in cut_path:
        dilated_cut[y, x] = True
    dilated_cut = binary_dilation(dilated_cut, disk(3))  # Stronger dilation
    mask[dilated_cut] = False

    labeled_img = label(mask, connectivity=2)
    unique_labels = np.unique(labeled_img)

    if len(unique_labels) < 3:
        return float('inf'), None

    areas = []
    for label_val in unique_labels[1:]:
        areas.append(np.sum(labeled_img == label_val))

    if len(areas) != 2:
        return float('inf'), None

    # Compute target areas
    target_area_1 = total_area * target_ratio
    target_area_2 = total_area * (1 - target_ratio)

    # Compute area difference from target
    area_diff = abs(areas[0] - target_area_1) + abs(areas[1] - target_area_2)
    return area_diff, areas

def find_optimal_cut(binary_image, skeleton, G, target_ratio, total_area):
    """
    Finds the cut that best matches the desired area split.
    """
    boundary_nodes = find_boundary_nodes(binary_image, G)
    best_cut = None
    best_diff = float('inf')
    best_areas = None

    for i in range(len(boundary_nodes)):
        for j in range(i+1, len(boundary_nodes)):
            try:
                path = nx.shortest_path(G, boundary_nodes[i], boundary_nodes[j], weight='weight')
                area_diff, areas = evaluate_cut(binary_image, path, target_ratio, total_area)
                if area_diff < best_diff:
                    best_diff = area_diff
                    best_cut = path
                    best_areas = areas
            except nx.NetworkXNoPath:
                continue

    return best_cut, best_areas, best_diff

def plot_cut_step(original_img, cut_path, regions, step):
    plt.figure(figsize=(8, 6))
    plt.imshow(original_img, cmap='gray', alpha=0.4, origin='lower')

    # Plot the cut
    if cut_path:
        cut_x, cut_y = zip(*cut_path)
        plt.plot(cut_x, cut_y, color='blue', linewidth=2, label='Cut Path')

    # Plot resulting regions
    colors = ['red', 'green']
    for idx, region in enumerate(regions):
        y, x = np.where(region)
        plt.scatter(x, y, color=colors[idx % len(colors)], s=1, label=f'Region {idx+1}')

    plt.title(f"Step {step}: After Cut")
    plt.legend()
    plt.show()

def sequential_partition(polygon_coords, num_partitions, img_size=200):
    """
    Sequentially partition the polygon into num_partitions, enforcing area ratios.
    """
    partitions = []
    cut_paths = []
    binary_img = polygon_to_binary_image(polygon_coords, img_size)
    medial_axis_img = compute_medial_axis(binary_img)
    G = skeleton_to_graph(medial_axis_img)
    total_area = compute_area(binary_img)

    queue = [(binary_img, G, total_area, num_partitions)]
    step = 1  # Step counter for plotting

    while queue:
        current_img, current_G, current_area, remaining_parts = queue.pop(0)

        if remaining_parts == 1:
            partitions.append(current_img)
            continue

        # Compute target ratio (e.g., for 3 parts: first cut 1/3 and 2/3)
        target_ratio = 1 / remaining_parts

        # Find optimal cut with the target ratio
        optimal_cut, cut_areas, area_diff = find_optimal_cut(
            current_img, compute_medial_axis(current_img), current_G, target_ratio, current_area
        )

        if not optimal_cut:
            partitions.append(current_img)
            continue

        # Apply the cut
        mask = np.copy(current_img)
        for x, y in optimal_cut:
            mask[y, x] = False

        dilated_cut = np.zeros_like(mask)
        for x, y in optimal_cut:
            dilated_cut[y, x] = True
        dilated_cut = binary_dilation(dilated_cut, disk(3))
        mask[dilated_cut] = False

        labeled_img = label(mask, connectivity=2)
        unique_labels = np.unique(labeled_img)

        if len(unique_labels) < 3:
            partitions.append(current_img)
            continue

        regions = [labeled_img == i for i in unique_labels if i != 0]
        regions = sorted(regions, key=lambda x: np.sum(x), reverse=True)

        # Plot current step with the cut and resulting regions
        plot_cut_step(current_img, optimal_cut, regions[:2], step)
        step += 1

        larger_region = regions[0]
        smaller_region = regions[1]

        partitions.append(smaller_region)
        cut_paths.append(optimal_cut)

        # Recompute medial axis for the remaining region and merge with the original
        new_medial_axis = compute_medial_axis(larger_region)
        new_G = skeleton_to_graph(new_medial_axis)
        merged_G = nx.compose(current_G, new_G)

        queue.append((larger_region, merged_G, np.sum(larger_region), remaining_parts - 1))

    return partitions, cut_paths

def plot_partitions(binary_img, partitions, cut_paths):
    plt.figure(figsize=(10, 8))
    plt.imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'yellow']
    for idx, (partition, cut) in enumerate(zip(partitions, cut_paths)):
        mask_y, mask_x = np.where(partition)
        plt.scatter(mask_x, mask_y, color=colors[idx % len(colors)], s=1, label=f'Partition {idx+1}')

        if cut:
            cut_x, cut_y = zip(*cut)
            plt.plot(cut_x, cut_y, color=colors[idx % len(colors)], linewidth=2)

    plt.title("Final Partitions with Cuts")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example Polygon (Star Shape)
    polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    #polygon_coords = np.load('bunny_cross_section_scaled.npy')

    # Number of partitions
    num_partitions = 3

    # Rasterize Polygon
    binary_img = polygon_to_binary_image(polygon_coords, img_size=100)

    # Perform Sequential Partitioning
    partitions, cut_paths = sequential_partition(polygon_coords, num_partitions, img_size=100)

    # Plot Final Result
    plot_partitions(binary_img, partitions, cut_paths)
