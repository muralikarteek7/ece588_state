import open3d as o3d
import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import cv2
from cv_bridge import CvBridge
from image_geometry import PinholeCameraModel
import os
import json
from datetime import datetime
import rasterio

bridge = CvBridge()



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

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.03, 
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

def process_oak(depth_path, camera_model):

    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)  # Load depth in float format
    depth = depth / np.max(depth) * MAX_DISTANCE_METERS  # Normalize (Assuming max depth corresponds to MAX_DISTANCE_METERS)

    # Apply distance filtering
    depth_filtered = np.where((depth >= MIN_DISTANCE_METERS) & (depth <= MAX_DISTANCE_METERS), depth, 0)

    # Convert depth map to point cloud
    points = []
    for v in range(depth.shape[0]):
        for u in range(depth.shape[1]):
            d = depth_filtered[v, u]
            if d > 0:  # Ignore zero-depth points
                x, y, z = camera_model.projectPixelTo3dRay((u, v))  # Get unit direction vector
                points.append((x * d, y * d, z * d))  # Scale by depth

    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points))

    return points



def filter_point_cloud(cam_pc, min_distance_meters, max_distance_meters):
    """
    Filters the point cloud based on distance range.

    Args:
        cam_pc (np.ndarray): Nx3 array containing (x, y, z) coordinates of the point cloud.
        min_distance_meters (float): Minimum distance threshold.
        max_distance_meters (float): Maximum distance threshold.

    Returns:
        np.ndarray: Filtered point cloud.
    """
    # Compute Euclidean distance of each point
    distances = np.linalg.norm(cam_pc, axis=1)
    
    # Apply distance filter
    mask = (distances >= min_distance_meters) & (distances <= max_distance_meters)
    
    return cam_pc[mask]



def process_ouster(points):
    ####################
    
    
    filtered_pc = filter_lidar_points(points, MAX_DISTANCE_METERS, MIN_DISTANCE_METERS)

    print("Segmenting ground using RANSAC...")
    ground_pc, non_ground_pc , plane_model = segment_ground_ransac(filtered_pc)

    print("Computing height from ground...")
    heights = compute_heights(points, plane_model)

    print("height of road", heights)


    print(f"Ground points: {ground_pc.shape[0]}, Non-ground points: {non_ground_pc.shape[0]}")
    
    


    return non_ground_pc



def depth_to_point_cloud(depth_img, camera_info):
    model = PinholeCameraModel()
    model.fromCameraInfo(camera_info)
    
    h, w = depth_img.shape
    points = []

    for v in range(h):
        for u in range(w):
            depth = depth_img[v, u] / 1000.0  # Convert to meters
            if depth > 0:
                x, y, z = model.projectPixelTo3dRay((u, v))
                points.append([x * depth, y * depth, depth])

    return np.array(points)

def visualize_point_cloud(points):
    """
    Visualizes the given point cloud using Open3D.

    Args:
        points (np.ndarray): Nx3 array of point cloud coordinates.
    """
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    pc.paint_uniform_color([0.1, 0.7, 0.9])  # Light blue color
    o3d.visualization.draw_geometries([pc])

def visualize_point_clouds(lidar_pc, cam_pc):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(lidar_pc)
    source.paint_uniform_color([1, 0, 0])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(cam_pc)
    target.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw_geometries([ target], window_name="Point Cloud Visualization")

def align_point_clouds(source_pc, target_pc):
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_pc)
    
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_pc)

    threshold = 0.05
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    print("ICP Transformation Matrix:\n", reg_p2p.transformation)
    return reg_p2p.transformation



def main():
    rospy.init_node('icp_lidar_camera_calibration')
    folder = "./scans"
    transformations = []
    
    for index in range(1,2):
        lidar_path = os.path.join(folder, f'lidar{index}.npz')
        depth_path = os.path.join(folder, f'depth{index}.tif')
        color_path = os.path.join(folder, f'color{index}.png')
        
        print("Waiting for camera info")
        camera_info_msg = rospy.wait_for_message("/oak/stereo/camera_info", CameraInfo)
        print("Camera info received")
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info_msg)
        

        color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        cam_pc = depth_to_point_cloud(depth_img, camera_info_msg)

        filtered_pc = filter_point_cloud(cam_pc, MIN_DISTANCE_METERS-0.5, MAX_DISTANCE_METERS + 0.5)

        data = np.load(lidar_path)
        lidar_pc = data['arr_0']

        
        visualize_point_cloud(cam_pc)

        cam_pc2 = process_oak(depth_path, camera_model)
        lidar_pc = process_ouster(lidar_pc)
        visualize_point_cloud(lidar_pc)
        visualize_point_cloud(cam_pc2)
        visualize_point_clouds(lidar_pc, cam_pc2)
        
        #visualize_point_clouds(lidar_pc, cam_pc)
        transformation = align_point_clouds(lidar_pc, filtered_pc )
        transformations.append(transformation)
    
    avg_transformation = np.mean(transformations, axis=0)
    print("Average Transformation Matrix:\n", avg_transformation)
    
    # Save to JSON file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"icp_transformation_ouster_2_oak_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": timestamp, "average_transformation": avg_transformation.tolist()}, f, indent=4)
    
    rospy.loginfo("ICP Calibration Done. Results saved in %s", output_path)
    return avg_transformation

if __name__ == "__main__":
    main()