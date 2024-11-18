# this is the controller for the endpoints

from flask import Flask, request, send_file
import os
from utils.conversion import usdz_to_xyz
from utils.file_utils import save_file, remove_file_if_exists, get_full_path
from utils.process_data import process_point_clouds
from utils.blueprintCreation import bp_read_xyz, bp_save_with_metadata, bp_error_pixels_from_image, bp_xyz_to_image
from utils.featureMatching import read_xyz, error_pixels_from_image, xyz_to_image, feature_matching_with_geometric_constraints, align_images_and_calculate_vector, apply_transformation_and_visualize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import open3d as o3d
import cv2
import numpy as np

app = Flask(__name__, static_url_path='/myflaskapp/static')
application = app


# Get the absolute path of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads')
SCRIPT_LOCATION = os.path.join(BASE_DIR, 'scripts')
IMAGE_MATCHING_FOLDER = os.path.join(BASE_DIR,'image_matching')

@app.route('/')
def hello_world():
    return "Hello, World!"


@app.route('/convertFile', methods=['POST'])
def convert_file():
    file = request.files.get('file')
    if not file:
        return "No file part"
    
    usdz_path = get_full_path(UPLOAD_FOLDER, 'userEnvironment.usdz')
    xyz_path = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')
    remove_file_if_exists(usdz_path)
    save_file(file, usdz_path)

    # Perform conversion
    usdz_to_xyz(usdz_path, xyz_path)

    return send_file(xyz_path, as_attachment=True)


@app.route('/uploadUserEnvironment', methods=['POST'])
def upload_user_environment():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    xyzPath = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')
    remove_file_if_exists(xyzPath)
    save_file(file, xyzPath)
    return "User Environment uploaded successfully"


@app.route('/uploadUserEnvironmentUSDZ', methods=['POST'])
def upload_user_environment_usdz():
    file = request.files.get('file')
    if not file:
        return "No file part"

    if file.filename == '':
        return "No selected file"

    usdz_path = get_full_path(UPLOAD_FOLDER, 'userEnvironment.usdz')
    xyz_path = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')

    # Remove any existing files before processing
    remove_file_if_exists(usdz_path)
    remove_file_if_exists(xyz_path)

    # Save the USDZ file
    save_file(file, usdz_path)

    usdz_to_xyz(usdz_path, xyz_path)

    return "User Environment uploaded and converted successfully"


@app.route('/uploadBlueprint', methods=['POST'])
def upload_blueprint():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = get_full_path(UPLOAD_FOLDER, 'blueprint.xyz')
    remove_file_if_exists(filename)
    save_file(file, filename)

    # process upload to png
    blueprint_png = get_full_path(IMAGE_MATCHING_FOLDER, 'blueprint.png')
    blueprint_points = bp_read_xyz(filename)
    blueprint_error_pixels = get_full_path(IMAGE_MATCHING_FOLDER, 'error_pixels.png')

    # calculate boundary
    min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
    min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
    boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

    # Generate an image for the blueprint and calculate error pixels
    error_pixels = bp_error_pixels_from_image(blueprint_points, blueprint_error_pixels, boundary, padding_pixels=50, image_size=(500, 500))

    # Generate an image for the blueprint and calculate pixels per unit
    scale_factor = bp_xyz_to_image(blueprint_points, blueprint_png, boundary, error_pixels, padding_pixels=50, image_size=(500, 500))

    # Save the output_image_path_blueprint with metadata
    bp_save_with_metadata(blueprint_png, boundary, error_pixels, scale_factor)
    return "Blueprint uploaded successfully"


@app.route('/getBackendpng', methods=['GET'])
def get_backendpng():
    # File paths
    png_filename = get_full_path(DOWNLOAD_FOLDER, 'export.png')
    remove_file_if_exists(png_filename)
    
    blueprint, little = process_point_clouds()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(little[0], little[2], little[1], color='y', s=5)
    ax.scatter(blueprint[0], blueprint[2], blueprint[1], color='b', s=5)

    # Birds-eye view
    ax.view_init(elev=90, azim=0)
    circle = plt.Circle((0, 0), 0.1, color='r')
    ax.add_patch(circle)
    art3d.pathpatch_2d_to_3d(circle, z=0, zdir='z')
    ax.axis("equal")
    
    plt.savefig(png_filename)

    return send_file(png_filename, as_attachment=True)

@app.route('/getNewPNG', methods=['GET'])
def get_backendImageMatching():

    # File paths generated dynamically using get_full_path
    xyz_file_path_blueprint = get_full_path(UPLOAD_FOLDER, 'blueprint.xyz')
    xyz_file_path_scan = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')
    output_image_path_blueprint = get_full_path(IMAGE_MATCHING_FOLDER, 'blueprint.png')
    output_image_path_scan = get_full_path(IMAGE_MATCHING_FOLDER, 'userScan.png')
    aligned_image_path = get_full_path(DOWNLOAD_FOLDER, 'aligned_image.png')
    output_image_path_error_pixels = get_full_path(IMAGE_MATCHING_FOLDER, 'error_pixels.png')

    remove_file_if_exists(output_image_path_blueprint)
    remove_file_if_exists(output_image_path_scan)
    remove_file_if_exists(aligned_image_path)
    remove_file_if_exists(output_image_path_error_pixels)

    # Read the blueprint data and get the bounding box / Read scan data
    blueprint_points = read_xyz(xyz_file_path_blueprint)
    scan_points = read_xyz(xyz_file_path_scan)

    # Calculate the boundary of the blueprint
    min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
    min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
    boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

    # Generate an image for the blueprint and calculate error pixels
    error_pixels = error_pixels_from_image(blueprint_points, output_image_path_error_pixels, boundary, padding_pixels=50, image_size=(500, 500))

    # Generate an image for the blueprint and calculate pixels per unit
    scale_factor = xyz_to_image(blueprint_points, output_image_path_blueprint, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=True)

    # Generate an image for a non-blueprint file (no pixels per unit calculation)
    xyz_to_image(scan_points, output_image_path_scan, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=False)

    # Load the two images to be matched
    img1 = cv2.imread(output_image_path_scan)
    img2 = cv2.imread(output_image_path_blueprint)

    # Perform feature matching with geometric constraints (translation and rotation only)
    aligned_img1, transformation_matrix, matches, keypoints1, keypoints2, points1, points2 = feature_matching_with_geometric_constraints(img1, img2)

    # Align img1 using the transformation matrix to align with img2 and calculate the vector difference
    aligned_img1_with_center, img2_with_center, center_vector, transformed_center_img1, center_img2 = align_images_and_calculate_vector(
        output_image_path_scan, output_image_path_blueprint, transformation_matrix
    )

    # Create the transformation matrix for XYZ file with adjusted tx and ty
    xyz_transformation_matrix = transformation_matrix.copy()
    xyz_transformation_matrix[0, 2] = -center_vector[0]
    xyz_transformation_matrix[1, 2] = -center_vector[1]

    # Apply the transformation and visualize
    apply_transformation_and_visualize(blueprint_points, scan_points, aligned_image_path, scale_factor, xyz_transformation_matrix)

    return send_file(aligned_image_path, as_attachment=True)

@app.route('/getBackendply', methods=['GET'])
def get_backendply():
    # File paths
    ply_filename = get_full_path(DOWNLOAD_FOLDER, 'export.ply')
    remove_file_if_exists(ply_filename)

    blueprint, _ = process_point_clouds()

    # Reverse the x values
    blueprint[0] = blueprint[0] * -1

    # Export the blueprint as a PLY file
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(blueprint[[0, 1, 2]].values)

    # Save the point cloud as a PLY file
    o3d.io.write_point_cloud(ply_filename, point_cloud, write_ascii=True)

    # Return the file as a download
    return send_file(ply_filename, as_attachment=True)

@app.route('/userXYZ', methods=['GET'])
def get_userxyz():
    # File paths
    user_xyz = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')
    return send_file(user_xyz, as_attachment=True)


if __name__ == '__main__':
    app.run()
