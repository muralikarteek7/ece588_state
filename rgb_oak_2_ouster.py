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

clicked_pixel = None


# Constants
MAX_DISTANCE_INCHES = 240  
MAX_DISTANCE_METERS = MAX_DISTANCE_INCHES * 0.0254  
MIN_DISTANCE_INCHES = 200  
MIN_DISTANCE_METERS = MIN_DISTANCE_INCHES * 0.0254  


image_points = np.array([
    [541, 513],
    [618, 511],
    [578, 511],
    [620, 548],
    [579, 547],
    [540, 548]
], dtype=np.float32)

# Corresponding 3D lidar points (X, Y, Z)
lidar_points = np.array([
    [5.78212643, 0.39177445, -1.50912416],
    [5.77956343, -0.19421133, -1.50576174],
    [5.78665829, 0.10744194, -1.50705492],
    [5.79314566, -0.20829456, -1.72898102],
    [5.77795982, 0.09370187, -1.72342598],
    [5.77857494, 0.39570308, -1.72751915]
], dtype=np.float32)


A = []
for i in range(len(image_points)):
    X, Y, Z = lidar_points[i]
    x, y = image_points[i]
    
    A.append([-X, -Y, -Z, -1,  0,  0,  0,  0, x * X, x * Y, x * Z, x])
    A.append([0,  0,  0,  0, -X, -Y, -Z, -1, y * X, y * Y, y * Z, y])

A = np.array(A)

# Solve Ap = 0 using SVD
U, S, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)  # Last row of Vt reshaped as 3x4 projection matrix

print("Estimated Camera Projection Matrix (P):\n", P)


def project_lidar_to_image(P, lidar_points):
    """ Projects 3D LiDAR points to 2D image coordinates using projection matrix P. """
    lidar_hom = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))  # Convert to homogeneous coordinates
    image_coords = (P @ lidar_hom.T).T  # Matrix multiplication

    # Normalize x and y
    image_coords[:, 0] /= image_coords[:, 2]  # x' = x / z
    image_coords[:, 1] /= image_coords[:, 2]  # y' = y / z

    return image_coords[:, :2].astype(int) 

def project_image_to_lidar(P, image):
    """ Projects image pixels to 3D lidar coordinates """
    h, w, _ = image.shape
    points_3d = []
    colors = []
    P_inv = np.linalg.pinv(P)
    
    for v in range(h):   # Iterate over image rows
        for u in range(w):  # Iterate over image columns
            pixel_hom = np.array([u, v, 1])  # Homogeneous coordinates
            world_coord = P_inv @ pixel_hom  # Transform to 3D
            world_coord /= world_coord[-1]  # Normalize
            points_3d.append(world_coord[:3])  # Keep (X, Y, Z)
            colors.append(image[v, u] / 255.0)  # Assign color

    return np.array(points_3d), np.array(colors)



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

def pick_points(lidar_pc):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pc)
    print("\n Click on a point in the Open3D window and press 'Q' to confirm selection.")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # Click on a point and press 'Q'
    vis.destroy_window()
    
    # Get selected point index
    picked_indexes = vis.get_picked_points()
    if picked_indexes:
        selected_points = np.asarray(pcd.points)[picked_indexes]
        print(f"Selected Points (X, Y, Z): {selected_points}")
    else:
        print("No point selected!")
        
def overlay_lidar_on_image(image, lidar_points, P, color=(0, 0, 255)):
    """
    Overlay LiDAR points onto an image using the projection matrix.
    
    Parameters:
        - image: The input image (numpy array).
        - lidar_points: Nx3 array of LiDAR points (X, Y, Z).
        - P: 3x4 camera projection matrix.
        - color: Color for the LiDAR points (default: red).
    """
    image_coords = project_lidar_to_image(P, lidar_points)

    # Draw points on image
    for (u, v) in image_coords:
        if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:  # Check bounds
            cv2.circle(image, (u, v), radius=3, color=color, thickness=-1)

    return image




def main():
    rospy.init_node('icp_lidar_camera_calibration')
    folder = "./scans"
    transformations = []
    
    for index in range(1,2):
        lidar_path = os.path.join(folder, f'lidar{index}.npz')
        #depth_path = os.path.join(folder, f'depth{index}.tif')
        color_path = os.path.join(folder, f'color{index}.png')
        mask_path = "segmentation_mask.png"
        print("Waiting for camera info")
        camera_info_msg = rospy.wait_for_message("/oak/rgb/camera_info", CameraInfo)
        print("Camera info received")
        camera_model = PinholeCameraModel()
        camera_model.fromCameraInfo(camera_info_msg)
        

        color_img = cv2.imread(color_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        #depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        #cam_pc = depth_to_point_cloud(color_img, camera_info_msg)

        #filtered_pc = filter_point_cloud(cam_pc, MIN_DISTANCE_METERS-0.5, MAX_DISTANCE_METERS + 0.5)

        data = np.load(lidar_path)
        lidar_pc = data['arr_0']
        lidar_pc = process_ouster(lidar_pc)
        output_image = overlay_lidar_on_image(color_img, lidar_pc, P)
        cv2.imshow("LiDAR Projection", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        #visualize_point_cloud(cam_pc)

        #cam_pc2 = process_oak(depth_path, camera_model)
        
        #pick_points(lidar_pc)

       
        #visualize_point_cloud(cam_pc2)
        #visualize_point_clouds(lidar_pc, cam_pc2)
        
        #visualize_point_clouds(lidar_pc, cam_pc)
        #transformation = align_point_clouds(lidar_pc, filtered_pc )
        #transformations.append(transformation)
    """
    avg_transformation = np.mean(transformations, axis=0)
    print("Average Transformation Matrix:\n", avg_transformation)
    
    # Save to JSON file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"icp_transformation_ouster_2_oak_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump({"timestamp": timestamp, "average_transformation": avg_transformation.tolist()}, f, indent=4)
    
    rospy.loginfo("ICP Calibration Done. Results saved in %s", output_path)
    return avg_transformation
    """

if __name__ == "__main__":
    main()