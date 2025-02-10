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

bridge = CvBridge()

def process_oak():
    ###################
    #TODO task1
    return points


def process_ouster():
    ####################
    #TODO task2
    return points



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
        
        print("Waiting for camera info")
        camera_info_msg = rospy.wait_for_message("/oak/stereo/camera_info", CameraInfo)
        print("Camera info received")
        
        depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        cam_pc = depth_to_point_cloud(depth_img, camera_info_msg)
        data = np.load(lidar_path)
        lidar_pc = data['arr_0']

        cam_pc = process_oak(cam_pc)
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