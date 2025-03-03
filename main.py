import matplotlib.pyplot as plt
import numpy as np
from skimage.morphology import medial_axis


from polygon_to_binary_image import polygon_to_binary_image
from skeleton_to_graph import skeleton_to_graph
from find_main_branch import find_main_branch
from find_balanced_branch import find_balanced_branch
from find_branch_fractions import find_branch_fractions
from binary_to_polygon_coords import binary_to_polygon_coords
from tessellate_with_buffer import tessellate_with_buffer
from plot_custom_solution import plot_custom_solution
from create_polygon import create_polygon


if __name__ == "__main__":

    #input_polygon_coords = [(0,0), (3,0), (3,3), (0,3), (0,0)] # Square
    #input_polygon_coords = [(0,0), (6,0), (6,3), (0,3), (0,0)] # Rectangle
    #input_polygon_coords = [(0,0), (3,0), (1.5,3), (0,0)] # Triangle
    #input_polygon_coords = [(0,0), (3,0), (3, 0.5), (0.5, 2.5), (3, 2.5), (3, 3), (0,3), (0,2.5), (2.5, 0.5), (0, 0.5), (0,0)] # Z-BEAM
    input_polygon_coords = np.load('bunny_cross_section_scaled.npy') 

    # Convert polygon to binary image:

    img_size = 500
    binary_img = polygon_to_binary_image(input_polygon_coords, img_size=img_size) 
    
    # Get medial axis:

    skeleton = medial_axis(binary_img)
    y, x = np.where(skeleton)  
    skeleton_coords = list(zip(x, y)) 

    # Transform the skeleton into a graph:

    G = skeleton_to_graph(skeleton)

    # Get the main AND balanced branches:

    main_branch = find_main_branch(G)
    balanced_branch, _, _ = find_balanced_branch(binary_img, G)

    # Find branch fractions:

    N = 20
    fraction_points, normal_vectors = find_branch_fractions(main_branch, N, shift=50)

    # Convert fractional points back to original polygon coordinates:

    original_polygon_coords = input_polygon_coords
    fraction_coords = binary_to_polygon_coords(fraction_points, original_polygon_coords, img_size=img_size, padding=5)
    fraction_coords_ar = np.array(fraction_coords)

    # Tessellate with buffer:

    buffer_size = 0.1
    polygons_A_star, polygons_B, polygons_A, polygons_A_star_areas, polygons_B_areas, polygons_A_areas = tessellate_with_buffer(fraction_coords_ar, input_polygon_coords, buffer_size)

    # Plot to visualize

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    if balanced_branch:
        path_x, path_y = zip(*main_branch)
        axes[0].plot(path_x, path_y, color='blue', linewidth=2)
        axes[0].set_title("Main Branch")

    # Plot the result on the second subplot
    axes[0].imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
    axes[0].imshow(skeleton, cmap='hot', alpha=0.8, origin='lower')

    #if balanced_branch:
    #    path_x, path_y = zip(*balanced_branch)
    #    axes[1].plot(path_x, path_y, color='blue', linewidth=2)
    #    axes[1].set_title("Balanced Branch")

    # Plot the result on the second subplot
    #axes[1].imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
    #axes[1].imshow(skeleton, cmap='hot', alpha=0.8, origin='lower')
    
    for i, point in enumerate(fraction_points, 1):
        axes[0].scatter(*point, s=50, color='red')

    #axes[0].set_title("Medial Axis with Longest Subgraph and Equidistant Points")
    #axes[0].legend()

    fraction_coords_ar = fraction_coords_ar.flatten().tolist()



    plot_custom_solution(fraction_coords_ar, polygons_A_star, polygons_B, 0.1, 0.1, axes[1])

    plt.show()

    """

    # Get optimal cut:
    
    balanced_branch, _, _ = find_balanced_branch(binary_img, G) 

    # Plot everything:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot the result on the first subplot
    axes[0].imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
    axes[0].imshow(skeleton, cmap='hot', alpha=0.8, origin='lower')

    # Plot the longest path
    if main_branch:
        path_x, path_y = zip(*main_branch)
        axes[0].plot(path_x, path_y, color='blue', linewidth=2, label='Longest Path')

    axes[0].set_title("Medial Axis with Longest Subgraph")
    axes[0].legend()

    # Plot the result on the second subplot
    axes[1].imshow(binary_img, cmap='gray', alpha=0.6, origin='lower')
    axes[1].imshow(skeleton, cmap='hot', alpha=0.8, origin='lower')

    # Plot the balanced branch
    if balanced_branch:
        balanced_x, balanced_y = zip(*balanced_branch)
        axes[1].plot(balanced_x, balanced_y, color='green', linewidth=2, label='Balanced Branch')

    axes[1].set_title("Medial Axis with Balanced Branch")
    axes[1].legend()

    plt.show()

    """
