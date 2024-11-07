import os
from pxr import Usd, UsdGeom, Gf
import math

def apply_transform(prim, point):
    """Apply the cumulative transformation from the prim to the point."""
    xformable = UsdGeom.Xformable(prim)
    world_transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return world_transform.Transform(point)

def distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2 + (point2[2] - point1[2]) ** 2)

def distribute_points(start, end, step_size):
    """Distribute points along the edge at intervals of step_size."""
    dist = distance(start, end)
    
    if dist < step_size:
        return [start, end]
    
    num_steps = int(dist // step_size)
    points = [Gf.Lerp(float(i) / num_steps, start, end) for i in range(num_steps + 1)]
    return points

def usdz_to_xyz(usdz_file_path, output_xyz_file_path, step_size=0.1):
    """Perform the conversion from USDZ to XYZ."""
    try:
        print(f"Attempting to open USDZ file at: {usdz_file_path}")
        if not os.path.exists(usdz_file_path):
            raise FileNotFoundError(f"USDZ file not found: {usdz_file_path}")

        stage = Usd.Stage.Open(usdz_file_path)
        if not stage:
            raise ValueError(f"Failed to open the USDZ file at: {usdz_file_path}")
        
        # Open the output XYZ file
        with open(output_xyz_file_path, 'w') as xyz_file:
            # Iterate over all prims looking for Mesh prims
            for prim in stage.Traverse():
                if prim.GetTypeName() == 'Mesh':
                    # Get the Mesh geometry
                    mesh = UsdGeom.Mesh(prim)
                    points_attr = mesh.GetPointsAttr()
                    points = points_attr.Get()
                    indices = mesh.GetFaceVertexIndicesAttr().Get()
                    counts = mesh.GetFaceVertexCountsAttr().Get()

                    # Track visited vertices to avoid duplications
                    visited_vertices = set()
                    index_offset = 0

                    for count in counts:
                        face_points = [apply_transform(prim, points[indices[i + index_offset]]) for i in range(count)]

                        # Sample more points along the edges using step_size
                        for i in range(len(face_points)):
                            start = face_points[i]
                            end = face_points[(i + 1) % len(face_points)]

                            # Distribute points along the edge at intervals of step_size
                            for sample_point in distribute_points(start, end, step_size):
                                key = tuple(sample_point)
                                if key not in visited_vertices:
                                    xyz_file.write(f"{sample_point[0]} {sample_point[1]} {sample_point[2]}\n")
                                    visited_vertices.add(key)

                        index_offset += count

        print(f"Conversion completed. Output saved to {output_xyz_file_path}")

    except Exception as e:
        print(f"General error during conversion: {e}")
        raise
