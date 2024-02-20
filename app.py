from flask import Flask, request, send_file
import aspose.threed as a3d
import os
import subprocess
import sys

app = Flask(__name__, static_url_path='/myflaskapp/static')
UPLOAD_FOLDER = '/var/www/api/uploads'
DOWNLOAD_FOLDER = '/var/www/api/downloads'
SCRIPT_LOCATION = '/var/www/api/scripts'


@app.route('/convertFile', methods=['POST'])
def convert_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(UPLOAD_FOLDER, 'Room.usdz')
        roomname = os.path.join(SCRIPT_LOCATION, 'Room.glb')
        envname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
        if os.path.isfile(roomname):
            os.system('rm ' + roomname)
        if os.path.isfile(filename):
            os.system('rm ' + filename)
        if os.path.isfile(envname):
            os.system('rm' + envname)
        file.save(filename)
        # Call the conversion script

        license = a3d.License()
        license.set_license("/var/www/api/scripts/Aspose.3D.lic")
        convfilename = os.path.join(UPLOAD_FOLDER, 'Room.usdz')
        outputname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
        os.system('usd2gltf -i ' + convfilename + ' -o /var/www/api/scripts/Room.glb')
        scene = a3d.Scene.from_file("/var/www/api/scripts/Room.glb")
        scene.save(outputname)


        # Create a direct download link for the second file
        return send_file(envname, as_attachment=True)


@app.route('/uploadUserEnvironment', methods=['POST'])
def upload_user_environment():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
        if os.path.isfile(filename):
            os.system('rm ' + filename)
        file.save(filename)
        return "User Environment uploaded successfully"


@app.route('/uploadUserEnvironmentUSDZ', methods=['POST'])
def upload_user_environment_usdz():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(UPLOAD_FOLDER, 'Room.usdz')
        roomname = os.path.join(SCRIPT_LOCATION, 'Room.glb')
        envname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
        if os.path.isfile(envname):
            os.system('rm ' + envname)
        if os.path.isfile(roomname):
            os.system('rm ' + roomname)
        if os.path.isfile(filename):
            os.system('rm ' + filename)
        file.save(filename)
        
        license = a3d.License()
        license.set_license("/var/www/api/scripts/Aspose.3D.lic")
        convfilename = os.path.join(UPLOAD_FOLDER, 'Room.usdz')
        outputname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
        os.system('usd2gltf -i ' + convfilename + ' -o /var/www/api/scripts/Room.glb')
        scene = a3d.Scene.from_file("/var/www/api/scripts/Room.glb")
        scene.save(outputname)

        return "User Environment uploaded successfully"


@app.route('/uploadBlueprint', methods=['POST'])
def upload_blueprint():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(UPLOAD_FOLDER, 'blueprint.xyz')
        if os.path.isfile(filename):
            os.system("rm " + filename)
        file.save(filename)
        return "Blueprint uploaded successfully"


@app.route('/getBackendpng', methods=['GET'])
def get_backendpng():
    filename = os.path.join(DOWNLOAD_FOLDER, 'export.png')
    if os.path.isfile(filename):
        # subprocess.run(['rm', '/var/www/api/export.png'])
        os.system("rm " + filename)
    os.system('python3 ' + SCRIPT_LOCATION + '/backend.py')
    return send_file(filename, as_attachment=True)


@app.route('/getBackendply', methods=['GET'])
def get_backendply():
    filename = os.path.join(DOWNLOAD_FOLDER, 'export.ply')
    if os.path.isfile(filename):
        # subprocess.run(['rm', '/var/www/api/export.png'])
        os.system("rm " + filename)
    os.system('python3 ' + SCRIPT_LOCATION + '/backend2.py')
    return send_file(filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
