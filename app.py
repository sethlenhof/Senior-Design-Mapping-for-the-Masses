from flask import Flask, request, send_file
import os
from convert_usdz_to_xyz import usdz_to_xyz

app = Flask(__name__, static_url_path='/myflaskapp/static')
UPLOAD_FOLDER = '/var/www/api/uploads'
DOWNLOAD_FOLDER = '/var/www/api/downloads'
SCRIPT_LOCATION = '/var/www/api/scripts'

def remove_if_exists(file_path):
    """Helper function to remove a file if it exists."""
    if os.path.isfile(file_path):
        os.remove(file_path)

@app.route('/convertFile', methods=['POST'])
def convert_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = os.path.join(UPLOAD_FOLDER, 'userScan.usdz')
    outputname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
    
    # Clean up any existing files
    remove_if_exists(filename)
    remove_if_exists(outputname)
    
    file.save(filename)

    # Convert USDZ to XYZ
    usdz_to_xyz(filename, outputname)

    # Direct download link for the converted file
    return send_file(outputname, as_attachment=True)

@app.route('/uploadUserEnvironment', methods=['POST'])
def upload_user_environment():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
    
    remove_if_exists(filename)
    
    file.save(filename)
    return "User Environment uploaded successfully"

@app.route('/uploadBlueprint', methods=['POST'])
def upload_blueprint():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    filename = os.path.join(UPLOAD_FOLDER, 'blueprint.xyz')
    
    remove_if_exists(filename)
    
    file.save(filename)
    return "Blueprint uploaded successfully"

@app.route('/getBackendpng', methods=['GET'])
def get_backendpng():
    filename = os.path.join(DOWNLOAD_FOLDER, 'export.png')
    remove_if_exists(filename)
    os.system(f'python3 {os.path.join(SCRIPT_LOCATION, "backend.py")}')
    return send_file(filename, as_attachment=True)

@app.route('/getBackendply', methods=['GET'])
def get_backendply():
    filename = os.path.join(DOWNLOAD_FOLDER, 'export.ply')
    remove_if_exists(filename)
    os.system(f'python3 {os.path.join(SCRIPT_LOCATION, "backend2.py")}')
    return send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run()
