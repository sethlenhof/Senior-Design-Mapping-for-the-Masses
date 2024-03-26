import numpy as np
from pxr import Usd, UsdGeom, Gf
import os

def apply_transform(prim, point):
    """Apply the cumulative transformation from the prim to the point."""
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return world_transform.Transform(point)

def interpolate_points(start, end, num_points):
    """Linearly interpolate between two points."""
    return [start + (end - start) * (float(i) / (num_points - 1)) for i in range(num_points)]

def sample_surface_points(vertices, num_samples):
    """Sample points uniformly across the surface of a triangle using barycentric coordinates."""
    points = []
    for _ in range(num_samples):
        r1, r2 = np.random.random(2)
        sqrt_r1 = np.sqrt(r1)
        barycentric = (1 - sqrt_r1, sqrt_r1 * (1 - r2), r2 * sqrt_r1)
        point = Gf.Vec3f(0, 0, 0)
        for j in range(3):
            point += vertices[j] * barycentric[j]
        points.append(point)
    return points

def usdz_to_xyz(usdz_file_path, output_xyz_file_path, edge_samples=2, surface_samples=3):
    stage = Usd.Stage.Open(usdz_file_path)
    with open(output_xyz_file_path, 'w') as xyz_file:
        for prim in stage.Traverse():
            if prim.GetTypeName() == 'Mesh':
                mesh = UsdGeom.Mesh(prim)
                points_attr = mesh.GetPointsAttr()
                points = points_attr.Get()
                indices = mesh.GetFaceVertexIndicesAttr().Get()
                counts = mesh.GetFaceVertexCountsAttr().Get()

                index_offset = 0
                for count in counts:
                    face_vertices = [apply_transform(prim, points[indices[i + index_offset]]) for i in range(count)]
                    for i in range(count):
                        start = face_vertices[i]
                        end = face_vertices[(i + 1) % count]
                        for sample_point in interpolate_points(start, end, edge_samples):
                            xyz_file.write(f"{sample_point[0]} {sample_point[1]} {sample_point[2]}\n")

                    if count == 3:
                        surface_points = sample_surface_points(face_vertices, surface_samples)
                        for point in surface_points:
                            xyz_file.write(f"{point[0]} {point[1]} {point[2]}\n")

                    index_offset += count

    print(f"Conversion completed. Output saved to {output_xyz_file_path}")

# The file paths would be set dynamically in the Flask route function,
# so the below lines are just placeholders for standalone script testing.
# usdz_file_path = '/var/www/api/uploads/Room.usdz'
# output_xyz_file_path = '/var/www/api/uploads/userEnvironment.xyz'
# usdz_to_xyz(usdz_file_path, output_xyz_file_path, edge_samples=10, surface_samples=10)