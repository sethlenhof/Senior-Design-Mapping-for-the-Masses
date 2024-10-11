# this is the controller for the endpoints

from flask import Flask, request, send_file
import os
from utils.conversion import UsdzToXyzConverter
from utils.file_utils import save_file, remove_file_if_exists, get_full_path
from utils.scene_manager import SceneManager

app = Flask(__name__, static_url_path='/myflaskapp/static')
application = app


# Get the absolute path of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DOWNLOAD_FOLDER = os.path.join(BASE_DIR, 'downloads')
SCRIPT_LOCATION = os.path.join(BASE_DIR, 'scripts')

# Initialize the scene manager
scene_manager = SceneManager(UPLOAD_FOLDER, DOWNLOAD_FOLDER)

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


@app.route('/getBackendpng', methods=['GET'])
def get_backendpng():
    filename = get_full_path(DOWNLOAD_FOLDER, 'export.png')
    remove_file_if_exists(filename)
    os.system('python3 ' + SCRIPT_LOCATION + '/backend.py')
    return send_file(filename, as_attachment=True)


@app.route('/getBackendply', methods=['GET'])
def get_backendply():
    filename = get_full_path(DOWNLOAD_FOLDER, 'export.ply')
    remove_file_if_exists(filename)
    os.system('python3 ' + SCRIPT_LOCATION + '/backend2.py')
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
