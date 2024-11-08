# this is the controller for the endpoints

from flask import Flask, request, send_file
import os
from utils.conversion import usdz_to_xyz
from utils.file_utils import save_file, remove_file_if_exists, get_full_path
from utils.process_data import process_point_clouds
from utils.imageAlignment import apply_transformation_and_visualize, read_xyz, error_pixels_from_image, xyz_to_image, feature_matching_with_geometric_constraints
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
def get_backendpngNew():
    # File paths generated dynamically using get_full_path
    xyz_file_path_blueprint = get_full_path(UPLOAD_FOLDER, 'blueprint.xyz')
    xyz_file_path_scan = get_full_path(UPLOAD_FOLDER, 'userEnvironment.xyz')
    output_image_path_blueprint = get_full_path(IMAGE_MATCHING_FOLDER, 'blueprint.png')
    output_image_path_scan = get_full_path(IMAGE_MATCHING_FOLDER, 'userScan.png')
    output_image_path_error_pixels = get_full_path(IMAGE_MATCHING_FOLDER, 'error_pixels.png')
    aligned_image_path = get_full_path(DOWNLOAD_FOLDER, 'aligned_image.png')

    # Remove existing images if they exist to prevent conflicts
    remove_file_if_exists(output_image_path_blueprint)
    remove_file_if_exists(output_image_path_scan)
    remove_file_if_exists(output_image_path_error_pixels)
    remove_file_if_exists(aligned_image_path)

    # Step 1: Read XYZ files and calculate boundaries
    blueprint_points = read_xyz(xyz_file_path_blueprint)
    min_x_blueprint, max_x_blueprint = np.min(blueprint_points[:, 0]), np.max(blueprint_points[:, 0])
    min_z_blueprint, max_z_blueprint = np.min(blueprint_points[:, 2]), np.max(blueprint_points[:, 2])
    boundary = (min_x_blueprint, max_x_blueprint, min_z_blueprint, max_z_blueprint)

    # Step 2: Calculate error pixels for the blueprint
    error_pixels = error_pixels_from_image(
        xyz_file_path_blueprint, output_image_path_error_pixels, boundary, padding_pixels=50, image_size=(500, 500)
    )

    # Step 3: Generate images for both the blueprint and scan point clouds
    # For the blueprint, calculate the scale factor
    scale_factor = xyz_to_image(
        xyz_file_path_blueprint, output_image_path_blueprint, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=True
    )

    # For the scan point cloud, generate the image without calculating scale factor
    xyz_to_image(
        xyz_file_path_scan, output_image_path_scan, boundary, error_pixels, padding_pixels=50, image_size=(500, 500), is_blueprint=False
    )

    # Step 4: Perform feature matching between the generated images to determine the transformation matrix
    # Load images using OpenCV
    img1 = cv2.imread(output_image_path_scan)
    img2 = cv2.imread(output_image_path_blueprint)

    # Make sure the images are loaded correctly
    if img1 is None or img2 is None:
        return "Error: One or both images failed to load!", 500

    # Get transformation matrix using feature matching
    _, transformation_matrix, _, _, _, _, _ = feature_matching_with_geometric_constraints(img1, img2)

    # Step 5: Apply the transformation to the XYZ point cloud and visualize
    apply_transformation_and_visualize(
        blueprint_file=xyz_file_path_blueprint,
        scan_file=xyz_file_path_scan,
        aligned_image_file=aligned_image_path,
        scale_factor=scale_factor,
        transformation_matrix_2d=transformation_matrix
    )

    # Send the saved file as a response
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

if __name__ == '__main__':
    app.run()
