import open3d as o3d
import numpy as np
import os

# Constants
MAX_DISTANCE_INCHES = 250  
MAX_DISTANCE_METERS = MAX_DISTANCE_INCHES * 0.0254  
MIN_DISTANCE_INCHES = 200  
MIN_DISTANCE_METERS = MIN_DISTANCE_INCHES * 0.0254  



def compute_heights(lidar_pc, plane_model):
    """ Compute height of each point from the estimated road plane. """
    a, b, c, d = plane_model
    normal_mag = np.sqrt(a**2 + b**2 + c**2)

    heights = np.abs( d / normal_mag)
    
    return heights



def load_lidar_data(file_path):
    """ Load LiDAR point cloud from NPZ file. """
    data = np.load(file_path)
    return data['arr_0']

def filter_lidar_points(lidar_pc, max_distance, min_distance):
    """ Remove points beyond max_distance or closer than min_distance. """
    distances = np.linalg.norm(lidar_pc[:, :3], axis=1)
    filtered_pc = lidar_pc[(distances <= max_distance) & (distances >= min_distance)]
    return filtered_pc

def segment_ground_ransac(lidar_pc):
    """ Segment the ground using RANSAC plane fitting. """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pc[:, :3])

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.2, 
                                             ransac_n=3, 
                                             num_iterations=1000)

    ground_points = lidar_pc[inliers]  # Points belonging to the ground
    non_ground_points = np.delete(lidar_pc, inliers, axis=0)  # Rest of the points

    print("Ground plane equation: ax + by + cz + d = 0 ->", plane_model)
    return ground_points, non_ground_points, plane_model

def visualize_point_clouds(ground_pc, non_ground_pc):
    """ Visualize ground and non-ground points separately. """
    ground_pcd = o3d.geometry.PointCloud()
    ground_pcd.points = o3d.utility.Vector3dVector(ground_pc[:, :3])
    ground_pcd.paint_uniform_color([1, 0, 0])  # Red color for ground

    non_ground_pcd = o3d.geometry.PointCloud()
    non_ground_pcd.points = o3d.utility.Vector3dVector(non_ground_pc[:, :3])
    non_ground_pcd.paint_uniform_color([0, 1, 0])  # Green for non-ground

    o3d.visualization.draw_geometries([ground_pcd, non_ground_pcd])

def main():
    folder = "./scans"
    lidar_path = os.path.join(folder, 'lidar1.npz')

    print("Loading LiDAR data...")
    lidar_pc = load_lidar_data(lidar_path)

    print("Filtering LiDAR data...")
    filtered_pc = filter_lidar_points(lidar_pc, MAX_DISTANCE_METERS, MIN_DISTANCE_METERS)

    print("Segmenting ground using RANSAC...")
    ground_pc, non_ground_pc , plane_model = segment_ground_ransac(filtered_pc)

    print("Computing height from ground...")
    heights = compute_heights(lidar_pc, plane_model)

    print("height of road", heights)


    print(f"Ground points: {ground_pc.shape[0]}, Non-ground points: {non_ground_pc.shape[0]}")
    
    visualize_point_clouds(ground_pc, non_ground_pc)

if __name__ == "__main__":
    main()
