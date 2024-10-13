# conversion.py

import numpy as np
from pxr import Usd, UsdGeom, Gf

class UsdzToXyzConverter:
    def __init__(self, usdz_file_path, output_xyz_file_path, edge_samples=2, surface_samples=3):
        self.usdz_file_path = usdz_file_path
        self.output_xyz_file_path = output_xyz_file_path
        self.edge_samples = edge_samples
        self.surface_samples = surface_samples

    def apply_transform(self, prim, point):
        """Apply the cumulative transformation from the prim to the point."""
        xformable = UsdGeom.Xformable(prim)
        world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        return world_transform.Transform(point)

    def interpolate_points(self, start, end, num_points):
        """Linearly interpolate between two points."""
        return [start + (end - start) * (float(i) / (num_points - 1)) for i in range(num_points)]

    def sample_surface_points(self, vertices, num_samples):
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

    def convert(self):
        """Perform the conversion from USDZ to XYZ."""
        stage = Usd.Stage.Open(self.usdz_file_path)
        with open(self.output_xyz_file_path, 'w') as xyz_file:
            for prim in stage.Traverse():
                if prim.GetTypeName() == 'Mesh':
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    points = points_attr.Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()
                    counts = mesh.GetFaceVertexCountsAttr().Get()

                    index_offset = 0
                    for count in counts:
                        face_vertices = [self.apply_transform(prim, points[indices[i + index_offset]]) for i in range(count)]
                        for i in range(count):
                            start = face_vertices[i]
                            end = face_vertices[(i + 1) % count]
                            for sample_point in self.interpolate_points(start, end, self.edge_samples):
                                xyz_file.write(f"{sample_point[0]} {sample_point[1]} {sample_point[2]}\n")

                        if count == 3:
                            surface_points = self.sample_surface_points(face_vertices, self.surface_samples)
                            for point in surface_points:
                                xyz_file.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"Conversion completed. Output saved to {self.output_xyz_file_path}")
