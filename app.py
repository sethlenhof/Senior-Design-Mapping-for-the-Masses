# this is the controller for the endpoints

from flask import Flask, request, send_file
import os
from utils.conversion import UsdzToXyzConverter
from utils.file_utils import save_file, remove_file_if_exists, get_full_path
from utils.process_data import process_point_clouds
from utils.point_cloud_utils import generate_plot
from utils.createPng import createPng
from utils.createply import createPly
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.art3d as art3d
import open3d as o3d

app = Flask(__name__, static_url_path='/myflaskapp/static')
application = app


# Get the absolute path of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads')
SCRIPT_LOCATION = os.path.join(BASE_DIR, 'scripts')

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
    converter = UsdzToXyzConverter(usdz_path, xyz_path)
    converter.convert()

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

    # Convert USDZ to XYZ (reuse the converter logic you already have)
    converter = UsdzToXyzConverter(usdz_path, xyz_path)
    converter.convert()

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


# @app.route('/getBackendpng', methods=['GET'])
# def get_backendpng():
#     # File paths
#     png_filename = get_full_path(DOWNLOAD_FOLDER, 'export.png')
#     remove_file_if_exists(png_filename)
    
#     blueprint, little = process_point_clouds()
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(little[0], little[2], little[1], color='y', s=5)
#     ax.scatter(blueprint[0], blueprint[2], blueprint[1], color='b', s=5)

#     # Birds-eye view
#     ax.view_init(elev=90, azim=0)
#     circle = plt.Circle((0, 0), 0.1, color='r')
#     ax.add_patch(circle)
#     art3d.pathpatch_2d_to_3d(circle, z=0, zdir='z')
#     ax.axis("equal")
    
#     plt.savefig(png_filename)

#     return send_file(png_filename, as_attachment=True)

# @app.route('/getBackendply', methods=['GET'])
# def get_backendply():
#     # File paths
#     ply_filename = get_full_path(DOWNLOAD_FOLDER, 'export.ply')
#     remove_file_if_exists(ply_filename)

#     blueprint, _ = process_point_clouds()

#     # Reverse the x values
#     blueprint[0] = blueprint[0] * -1

#     # Export the blueprint as a PLY file
#     point_cloud = o3d.geometry.PointCloud()
#     point_cloud.points = o3d.utility.Vector3dVector(blueprint[[0, 1, 2]].values)

#     # Save the point cloud as a PLY file
#     o3d.io.write_point_cloud(ply_filename, point_cloud, write_ascii=True)

#     # Return the file as a download
#     return send_file(ply_filename, as_attachment=True)


@app.route('/getBackendpng', methods=['GET'])
def get_backendpng():
    png_filename = get_full_path(DOWNLOAD_FOLDER, 'export.png')
    remove_file_if_exists(png_filename)

    # Generate PNG image using the refactored function
    # createPng()
    generate_plot(upload_folder=UPLOAD_FOLDER, download_folder=DOWNLOAD_FOLDER)

    return send_file(png_filename, as_attachment=True)

@app.route('/getBackendply', methods=['GET'])
def get_backendply():
    ply_filename = get_full_path(DOWNLOAD_FOLDER, 'export.ply')
    remove_file_if_exists(ply_filename)

    # Generate PLY point cloud using the refactored function
    # generate_ply_point_cloud(UPLOAD_FOLDER, DOWNLOAD_FOLDER, 'blueprint.xyz', 'userEnvironment.xyz')
    createPly()

    return send_file(ply_filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
