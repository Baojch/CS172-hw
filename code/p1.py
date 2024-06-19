# MAC： open3d：pip install open3d==0.15.1
import open3d as o3d

# 读取第一个点云文件
pcd1 = o3d.io.read_point_cloud("data/bun000.ply")

# 读取第二个点云文件
pcd2 = o3d.io.read_point_cloud("data/bun045.ply")

# 读取第三个点云文件
pcd3 = o3d.io.read_point_cloud("data/bun090.ply")

# 可视化第一个点云
o3d.visualization.draw_geometries([pcd1])

# 可视化第二个点云
o3d.visualization.draw_geometries([pcd2])

# 可视化第三个点云
o3d.visualization.draw_geometries([pcd3])

# # 同时可视化三个点云
# o3d.visualization.draw_geometries([pcd1, pcd2, pcd3])