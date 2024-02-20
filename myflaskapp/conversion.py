import aspose.threed as a3d
import os
os.system('usd2gltf -i Room.usdz -o Room.glb')
scene = a3d.Scene.from_file("Room.glb")
scene.save("Output.xyz")
