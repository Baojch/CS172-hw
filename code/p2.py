import numpy as np
import open3d as o3d
import copy
from tqdm import tqdm
from scipy.spatial import KDTree

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def nearest_neighbor(src, dst):
    src_points = np.asarray(src.points)
    dst_points = np.asarray(dst.points)
    tree = KDTree(dst_points)
    distances, indices = tree.query(src_points)
    return indices, distances

def compute_transformation(src, dst, indices):
    src_points = np.asarray(src.points)
    dst_points = np.asarray(dst.points)

    matched_src_points = src_points
    matched_dst_points = dst_points[indices]

    centroid_src = np.mean(matched_src_points, axis=0)
    centroid_dst = np.mean(matched_dst_points, axis=0)

    src_centered = matched_src_points - centroid_src
    dst_centered = matched_dst_points - centroid_dst

    H = np.dot(src_centered.T, dst_centered)

    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # To ensure R is valid: orthogonal(RTR=I) and det(R)=1[如果det<0，说明R是反射矩阵，不是旋转矩阵，需要乘以-1]
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    t = centroid_dst.T - np.dot(R, centroid_src.T)

    transformation = np.identity(4)
    transformation[:3, :3] = R
    transformation[:3, 3] = t

    return transformation

def icp(source, target, init_transformation, threshold, max_iterations, subset_size):
    source_temp = copy.deepcopy(source)
    transformation = init_transformation
    prev_error = float('inf')

    pbar = tqdm(range(max_iterations), desc="ICP Iteration")
    for i in pbar:
        # Randomly subset
        num_points = len(source_temp.points)
        subset_indices = np.random.choice(num_points, size=subset_size, replace=False)
        source_subset = source_temp.select_by_index(subset_indices)
        
        indices, distances = nearest_neighbor(source_subset, target)

        error = np.sum(distances ** 2)
        pbar.set_postfix(loss=error)
        
        new_transformation = compute_transformation(source_subset, target, indices)
        source_temp.transform(new_transformation)

        R = new_transformation[:3, :3]
        t = new_transformation[:3, 3]

        delta_R = np.linalg.norm(R - transformation[:3, :3])
        delta_t = np.linalg.norm(t - transformation[:3, 3])
        
        # 退出条件
        if delta_R < threshold and delta_t < threshold:
            break
        elif error < threshold:
            break

        transformation = np.dot(new_transformation, transformation)
        prev_error = error

    return transformation, error

source1 = o3d.io.read_point_cloud("data/bun000.ply")
target1 = o3d.io.read_point_cloud("data/bun315.ply")

source2 = o3d.io.read_point_cloud("data/bun000.ply")
target2 = o3d.io.read_point_cloud("data/bun045.ply")

source3 = o3d.io.read_point_cloud("data/bun270.ply")
target3 = o3d.io.read_point_cloud("data/bun315.ply")

# Initial transformation matrix
trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])

threshold = 1e-3
max_iterations = 800
subset_size = 1000

transformation1, error1 = icp(source1, target1, trans_init, threshold, max_iterations, subset_size)

transformation2, error2 = icp(source2, target2, trans_init, threshold, max_iterations, subset_size)

transformation3, error3 = icp(source3, target3, trans_init, threshold, max_iterations, subset_size)
print("Transformation for 000 and 315 is:")
print(transformation1)
draw_registration_result(source1, target1, transformation1)

print("Transformation for 000 and 045 is:")
print(transformation2)
draw_registration_result(source2, target2, transformation2)

print("Transformation for 270 and 315 is:")
print(transformation3)
draw_registration_result(source3, target3, transformation3)
