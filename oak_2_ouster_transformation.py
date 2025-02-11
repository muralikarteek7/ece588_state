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


def process_oak(color_img_path, depth_img_path):
    color_img = cv2.imread(color_img_path)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    hsv = cv2.cvtColor(color_img, cv2.COLOR_RGB2HSV)

    lower_bound = np.array([98, 43, 0])
    upper_bound = np.array([118, 143, 94])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = max(contours, key=cv2.contourArea) if contours else None

    box_mask = np.zeros_like(mask, dtype=np.uint8)
    if largest_contour is not None:
        cv2.drawContours(box_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    with rasterio.open(depth_img_path) as src:
        depth_array = src.read(1)

    masked_depth = np.where(box_mask == 255, depth_array, 0)

    fx, fy = 684.8333, 684.6097
    cx, cy = 573.3711, 363.7009

    points = []
    colors = []
    h, w = depth_array.shape

    for y in range(h):
        for x in range(w):
            d = masked_depth[y, x]
            if d > 0:
                z = d * 0.1
                X = (x - cx) * z / fx
                Y = (y - cy) * z / fy
                points.append([X, Y, z])
                colors.append(color_img[y, x] / 255.0)
                
    non_ground_pcd = o3d.geometry.PointCloud()
    non_ground_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    non_ground_pcd.paint_uniform_color([0, 1, 0])  # Green for non-ground

    o3d.visualization.draw_geometries([points])

    return points

def process_ouster(points):
    ####################
    
    
    filtered_pc = filter_lidar_points(points, MAX_DISTANCE_METERS, MIN_DISTANCE_METERS)

    print("Segmenting ground using RANSAC...")
    ground_pc, non_ground_pc , plane_model = segment_ground_ransac(filtered_pc)

    print("Computing height from ground...")
    heights = compute_heights(lidar_pc, plane_model)

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
        

        color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        cam_pc = depth_to_point_cloud(depth_img, camera_info_msg)
        data = np.load(lidar_path)
        lidar_pc = data['arr_0']

        cam_pc = process_oak(color_path,depth_path )
        lidar_pc = process_ouster(lidar_pc)
        
        #visualize_point_clouds(lidar_pc, cam_pc)
        transformation = align_point_clouds(lidar_pc, cam_pc)
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