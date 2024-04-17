import pycolmap
import open3d as o3d
from pathlib import Path

from hloc import visualization

# output = Path("datasets/naverlabs/HD_1F/sfm_superpoint+superglue_indoor_30/models/0")
output = Path("outputs/naverlabs/HD_1F/sfm_superpoint+superglue/")
reconstruction = pycolmap.Reconstruction(output)
print(reconstruction.summary())

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width = 600, height = 400)

sfm_cloud = o3d.geometry.PointCloud()
for point3D_id, point3D in reconstruction.points3D.items():
    sfm_cloud.points.append(point3D.xyz)
    sfm_cloud.colors.append(point3D.color / 255.0)

lidar = o3d.io.read_point_cloud("/home/seungjae/projects/vloc/vloc_ws/src/feat-your-cloud/data/naverlabs/HyundaiDepartmentStore_1F_static.pcd")
vis.add_geometry(lidar)

vis.add_geometry(sfm_cloud)
vis.run()
vis.destroy_window()
