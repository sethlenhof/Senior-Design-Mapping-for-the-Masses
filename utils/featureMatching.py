import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image

def read_xyz(file_path):
    """Read XYZ data from a file."""
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y, z = map(float, line.strip().split())
            points.append((x, y, z))
    return np.array(points)

def xyz_to_image(points, output_image_path, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), meter_interval=1, is_blueprint=False):
    """
    Converts an XYZ file to an image and optionally calculates pixels per unit if it's a blueprint.
    """

    # Create a copy of the points array to prevent modification of the original points
    points = points.copy()

    # Invert the x-axis for left-handed coordinate system
    points[:, 0] = -points[:, 0]  # Invert the x-axis

    # Sort points by y-values so that higher points are plotted last (on top)
    points = points[np.argsort(points[:, 1])]

    # Get x, z for 2D plotting, and y for the gradient color
    x_vals = points[:, 0]
    z_vals = points[:, 2]
    y_vals = points[:, 1]  # Use y for color gradient

    # Create a discrete color map where the color changes every 1 meter
    min_y = np.min(y_vals)
    max_y = np.max(y_vals)

    # Define color steps, one color per meter interval
    meter_bins = np.arange(min_y, max_y + meter_interval, meter_interval)
    
    # Use a fixed colormap (gist_rainbow or any other colormap) to assign colors for each interval
    cmap = plt.cm.gist_rainbow
    norm = plt.Normalize(vmin=min_y, vmax=max_y)
    
    # Find which bin each point's y value falls into (rounded down to the nearest meter)
    y_bin_indices = np.digitize(y_vals, meter_bins) - 1  # Subtract 1 to get the index for bins

    # Assign colors based on the bins
    colors = cmap(norm(meter_bins[y_bin_indices]))  # Map bin indices to colors

    # Calculate the bounding box using the blueprint's boundary
    min_x, max_x, min_z, max_z = boundary

    # Calculate the full range for both x and z
    range_x = max_x - min_x
    range_z = max_z - min_z

    # Determine the maximum range for symmetric limits around (0, 0)
    max_range = max(abs(min_x), abs(max_x), abs(min_z), abs(max_z))

    # Create the plot
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)

    # Plot the points in the xz plane, applying the color based on the meter intervals
    scatter = ax.scatter(x_vals, z_vals, c=colors, s=1) 

    # Convert the padding in pixels to data units (assuming square image for simplicity)
    fig_width_inch = image_size[0] / 100
    fig_height_inch = image_size[1] / 100
    data_padding_x = (padding_pixels / 100) * (range_x / fig_width_inch)
    data_padding_z = (padding_pixels / 100) * (range_z / fig_height_inch)

    # Set limits with (0, 0) in the center of the plot and padding
    ax.set_xlim(-max_range - data_padding_x, max_range + data_padding_x)
    ax.set_ylim(-max_range - data_padding_z, max_range + data_padding_z)

    # Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal', 'box')

    # Remove axes for a cleaner image
    ax.axis('off')

    # Save the figure as an image
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Calculate pixels per unit only if this image is marked as a blueprint
    if is_blueprint:
        pixels_per_unit = calculate_pixels_per_unit_from_image(output_image_path, range_x, range_z, error_pixels)
        return pixels_per_unit

    # Return None if not a blueprint
    return None

def error_pixels_from_image(points, output_image_path, boundary, padding_pixels=50, image_size=(500, 500)):
    # Find the point with the maximum y-value
    max_y_index = np.argmax(points[:, 1])
    max_y_point = points[max_y_index]  # Extract the single point with the max y-value

    # Get x, z for 2D plotting (max_y_point contains a single point)
    x_vals = [max_y_point[0]]
    z_vals = [max_y_point[2]]

    # Calculate the bounding box using the blueprint's boundary
    min_x, max_x, min_z, max_z = boundary

    # Calculate the full range for both x and z
    range_x = max_x - min_x
    range_z = max_z - min_z

    # Determine the maximum range for symmetric limits around (0, 0)
    max_range = max(abs(min_x), abs(max_x), abs(min_z), abs(max_z))

    # Create the plot
    fig, ax = plt.subplots(figsize=(image_size[0] / 100, image_size[1] / 100), dpi=100)

    colors = 'blue'

    # Plot the points in the xz plane, applying the color based on the meter intervals
    scatter = ax.scatter(x_vals, z_vals, c=colors, s=1) 

    # Convert the padding in pixels to data units (assuming square image for simplicity)
    fig_width_inch = image_size[0] / 100
    fig_height_inch = image_size[1] / 100
    data_padding_x = (padding_pixels / 100) * (range_x / fig_width_inch)
    data_padding_z = (padding_pixels / 100) * (range_z / fig_height_inch)

    # Set limits with (0, 0) in the center of the plot and padding
    ax.set_xlim(-max_range - data_padding_x, max_range + data_padding_x)
    ax.set_ylim(-max_range - data_padding_z, max_range + data_padding_z)

    # Set equal aspect ratio to prevent distortion
    ax.set_aspect('equal', 'box')

    # Remove axes for a cleaner image
    ax.axis('off')

    # Save the figure as an image
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Open the saved image to get its actual dimensions
    img = Image.open(output_image_path)
    img_array = np.array(img)
    
    # Convert the image to grayscale for easier analysis
    gray_img = np.mean(img_array, axis=2)

    # Find the first and last non-white pixels along the X and Z axes
    x_non_empty = np.where(np.any(gray_img != 255, axis=0))[0]
    z_non_empty = np.where(np.any(gray_img != 255, axis=1))[0]

    first_x_pixel, last_x_pixel = x_non_empty[0], x_non_empty[-1]
    first_z_pixel, last_z_pixel = z_non_empty[0], z_non_empty[-1]

    # Calculate the actual width and height of the plotted area in pixels
    effective_width = last_x_pixel - first_x_pixel
    effective_height = last_z_pixel - first_z_pixel

    # Calculate the error pixels
    error_pixels = max(effective_width, effective_height)

    return error_pixels

def calculate_pixels_per_unit_from_image(image_path, range_x, range_z, error_pixels):
    """Calculate pixels per unit using the image and XYZ range (range_x and range_z)."""
    # Open the saved image to get its actual dimensions
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Convert the image to grayscale for easier analysis
    gray_img = np.mean(img_array, axis=2)

    # Find the first and last non-white pixels along the X and Z axes
    x_non_empty = np.where(np.any(gray_img != 255, axis=0))[0]
    z_non_empty = np.where(np.any(gray_img != 255, axis=1))[0]

    first_x_pixel, last_x_pixel = x_non_empty[0], x_non_empty[-1]
    first_z_pixel, last_z_pixel = z_non_empty[0], z_non_empty[-1]

    # Calculate the actual width and height of the plotted area in pixels
    effective_width = last_x_pixel - first_x_pixel - error_pixels - 1
    effective_height = last_z_pixel - first_z_pixel - error_pixels - 1

    # Calculate pixels per unit using the effective width and height
    pixels_per_unit_x = effective_width / range_x
    pixels_per_unit_z = effective_height / range_z

    # Take the minimum of the two to avoid distortion
    final_pixels_per_unit = min(pixels_per_unit_x, pixels_per_unit_z)
    print(f"Final pixels per unit (min of X and Z): {final_pixels_per_unit}")

    return final_pixels_per_unit

def align_images_and_calculate_vector(output_image_path_scan, output_image_path_blueprint, transformation_matrix):
    """
    Align img1 with img2 using the given transformation matrix and calculate the vector difference between the centers.

    Parameters:
    - output_image_path_scan: Path to the first image (to be transformed).
    - output_image_path_blueprint: Path to the second image (reference image).
    - transformation_matrix: 2x3 transformation matrix for affine transformation.

    Returns:
    - aligned_img1: The transformed version of img1 aligned with img2.
    - img2: The reference image.
    - center_vector: The vector difference (dx, dy) between the center of img1 and img2 after alignment.
    """
    # Load the images
    img1 = cv2.imread(output_image_path_scan)
    img2 = cv2.imread(output_image_path_blueprint)

    # Ensure the images are loaded properly
    if img1 is None or img2 is None:
        raise ValueError("One or both of the image paths are invalid!")

    # Get the dimensions of img2 for warping (we want to align img1 to img2's size)
    rows, cols = img2.shape[:2]

    # Calculate the center of img1 and img2
    center_img1 = np.array([img1.shape[1] // 2, img1.shape[0] // 2, 1])  # (x, y, 1)
    center_img2 = np.array([img2.shape[1] // 2, img2.shape[0] // 2])      # (x, y)

    # Transform the center of img1 using the transformation matrix
    transformed_center_img1 = transformation_matrix @ center_img1

    # Calculate the vector difference between transformed center of img1 and center of img2
    dx = transformed_center_img1[0] - center_img2[0]
    dy = transformed_center_img1[1] - center_img2[1]
    center_vector = (dx, dy)

    # Apply the affine transformation to align img1 to img2
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

    return aligned_img1, img2, center_vector, transformed_center_img1, center_img2

def feature_matching_with_geometric_constraints(img1, img2):
    """
    Perform feature matching between two images while applying geometric constraints for rotation and translation.
    Assumes that the scale does not change.
    """
    # Step 1: Detect keypoints and descriptors using SIFT
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Step 2: Match descriptors using BFMatcher with cross-check to ensure mutual nearest matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Step 3: Sort the matches based on distance (lower distance is better)
    matches = sorted(matches, key=lambda x: x.distance)

    # Step 4: Extract matched keypoints in both images
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Step 5: Estimate a transformation matrix using RANSAC
    # We use cv2.estimateAffinePartial2D with RANSAC to get a rotation and translation transformation
    transformation_matrix, inliers = cv2.estimateAffinePartial2D(points1, points2, method=cv2.RANSAC)

    # Step 6: Apply the transformation to img1 to align it with img2
    rows, cols = img2.shape[:2]
    aligned_img1 = cv2.warpAffine(img1, transformation_matrix, (cols, rows))

    return aligned_img1, transformation_matrix, matches, keypoints1, keypoints2, points1, points2

def transform_point_cloud_without_x_inversion(points, transformation_matrix_3d):
    """
    Applies a 3D transformation matrix to a point cloud without X-axis inversion.
    :param points: An Nx3 NumPy array of (X, Y, Z) coordinates.
    :param transformation_matrix_3d: A 3x4 transformation matrix for 3D transformation.
    :return: Transformed Nx3 NumPy array of (X, Y, Z) coordinates.
    """
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))
    transformed_points = homogeneous_points @ transformation_matrix_3d.T
    return transformed_points[:, :3]

def visualize_point_clouds_with_grid(pcd1, pcd2, transformed_origin, direction_vector, output_path):
    """Visualize two point clouds and a rotated arrow at the transformed origin point in a bird's-eye (XZ) view using Matplotlib."""
    plt.figure(figsize=(10, 10))
    
    # Plot the first point cloud (blueprint) in blue
    plt.scatter(pcd1[:, 0], pcd1[:, 2], color='blue', s=5, label='Blueprint')
    
    # Plot the transformed scan point cloud in red
    plt.scatter(pcd2[:, 0], pcd2[:, 2], color='red', s=5, label='Transformed Scan')
    
    # Define arrowhead parameters for a minimal line segment
    arrow_length = 0.005 * (pcd1[:, 0].max() - pcd1[:, 0].min())  # Minimal arrow length just to display the head
    head_width = arrow_length * 5  # Large enough width to make the head noticeable
    head_length = arrow_length * 5  # Large enough length to make the head noticeable

    # Format the transformed origin coordinates to two decimal places
    transformed_origin_coords = f"({transformed_origin[0]:.2f}, {transformed_origin[2]:.2f})"

    # Plot only the arrowhead at the transformed origin with minimal extension
    plt.arrow(
        transformed_origin[0],  # Starting X position (transformed origin)
        transformed_origin[2],  # Starting Z position (transformed origin)
        direction_vector[0] * arrow_length,  # Minimal X component for the arrow direction
        direction_vector[2] * arrow_length,  # Minimal Z component for the arrow direction
        head_width=head_width, 
        head_length=head_length, 
        fc='magenta', 
        ec='magenta', 
        label=f'Current Location {transformed_origin_coords}'  # Display coordinates in the label
    )

    # Flip the X-axis so negative is on the right
    plt.gca().invert_xaxis()
    
    # Set equal scaling for X and Z axes and maintain a square aspect ratio
    plt.axis('equal')
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Add labels, grid, and title
    plt.xlabel("X")  # Flipped
    plt.ylabel("Z")
    plt.grid(True)

    # Move legend to the top-right corner, outside the plot area
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    # Save the plot as a PNG file
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

    # Display the plot
    plt.show()

def apply_transformation_and_visualize(blueprint_points, scan_points, aligned_image_file, scale_factor, transformation_matrix_2d):
    """Apply a 3D transformation to the scan point cloud and visualize both point clouds."""

    # Create the 3D transformation matrix based on the 2D matrix and scale factor
    transformation_matrix_3d = np.array([
        [transformation_matrix_2d[0][0], 0, transformation_matrix_2d[0][1], transformation_matrix_2d[0][2] / scale_factor],
        [0, 1, 0, 0],
        [transformation_matrix_2d[1][0], 0, transformation_matrix_2d[1][1], transformation_matrix_2d[1][2] / scale_factor]
    ])

    # Print the resulting 3D transformation matrix
    print("Resulting 3D Transformation Matrix:")
    print(transformation_matrix_3d)

    # Apply the transformation to the scan point cloud without X-axis inversion
    transformed_scan_points = transform_point_cloud_without_x_inversion(scan_points, transformation_matrix_3d)

    # Calculate the transformed origin point (0, 0, 0)
    origin = np.array([0, 0, 0, 1])
    transformed_origin = transformation_matrix_3d @ origin.T

    # Print the transformed origin point
    print(f"Transformed Origin Point: {transformed_origin[:3]}")

    # Extract the direction vector for the arrow from the transformation matrix
    direction_vector = -transformation_matrix_3d[:3, 2]  # -Z direction component of the rotation matrix

    # Visualize the point clouds in the XZ plane using Matplotlib with the rotated arrow
    visualize_point_clouds_with_grid(blueprint_points, transformed_scan_points, transformed_origin[:3], direction_vector, aligned_image_file)

# # How to use
# output_image_path_blueprint = '//Users/sethlenhof/Code/MFTM-Algo/image_matching/blueprint2f.png' # firstFloorSouth_even_output.png' # secondFloor_even_output.png' # room1(2)_even_output.png'
# output_image_path_scan = '/Users/sethlenhof/Code/MFTM-Algo/image_matching/sethRoom.png' # SameSpotWest_even_output.png' # KitchenNorth_even_output.png' # RoomSecondFloor_even_output.png' # room1_testcase5_even_output.png'
# output_image_path_error_pixels = '/Users/sethlenhof/Code/MFTM-Algo/image_matching/error_pixels.png'
# aligned_image_path = '/Users/sethlenhof/Code/MFTM-Algo/image_matching/aligned.png'

# xyz_file_path_blueprint = sys.argv[1]
# xyz_file_path_scan = sys.argv[2]

# # Read the blueprint data and get the bounding box / Read scan data
# blueprint_points = read_xyz(xyz_file_path_blueprint)
# scan_points = read_xyz(xyz_file_path_scan)

# # Calculate the boundary of the blueprint
# min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
# min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
# boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

# # Generate an image for the blueprint and calculate error pixels
# error_pixels = error_pixels_from_image(blueprint_points, output_image_path_error_pixels, boundary, padding_pixels=50, image_size=(500, 500))

# # Generate an image for the blueprint and calculate pixels per unit
# scale_factor = xyz_to_image(blueprint_points, output_image_path_blueprint, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=True)

# # Generate an image for a non-blueprint file (no pixels per unit calculation)
# xyz_to_image(scan_points, output_image_path_scan, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=False)

# # Load the two images to be matched
# img1 = cv2.imread(output_image_path_scan)
# img2 = cv2.imread(output_image_path_blueprint)

# # Perform feature matching with geometric constraints (translation and rotation only)
# aligned_img1, transformation_matrix, matches, keypoints1, keypoints2, points1, points2 = feature_matching_with_geometric_constraints(img1, img2)

# # Align img1 using the transformation matrix to align with img2 and calculate the vector difference
# aligned_img1_with_center, img2_with_center, center_vector, transformed_center_img1, center_img2 = align_images_and_calculate_vector(
#     output_image_path_scan, output_image_path_blueprint, transformation_matrix
# )

# # Create the transformation matrix for XYZ file with adjusted tx and ty
# xyz_transformation_matrix = transformation_matrix.copy()
# xyz_transformation_matrix[0, 2] = -center_vector[0]
# xyz_transformation_matrix[1, 2] = -center_vector[1]

# # Apply the transformation and visualize
# apply_transformation_and_visualize(blueprint_points, scan_points, aligned_image_path, scale_factor, xyz_transformation_matrix)
