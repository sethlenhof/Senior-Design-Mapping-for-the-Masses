import aspose.threed as a3d
import os
UPLOAD_FOLDER = '/var/www/api/uploads'

license = a3d.License()
license.set_license("Aspose.3D.lic")
filename = os.path.join(UPLOAD_FOLDER, 'Room.usdz')
outputname = os.path.join(UPLOAD_FOLDER, 'userEnvironment.xyz')
os.system('usd2gltf -i ' + filename + ' -o /var/www/api/scripts/Room.glb')
scene = a3d.Scene.from_file("/var/www/api/scripts/Room.glb")
scene.save(outputname)
