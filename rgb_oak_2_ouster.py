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
from scipy.spatial import ConvexHull, distance



bridge = CvBridge()

clicked_pixel = None


# Constants
MAX_DISTANCE_INCHES = 255  
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


def estimate_extrinsic(lidar_points, image_points, K):
    """
    Estimates the extrinsic matrix (R, t) from LiDAR 3D points to camera 2D points.

    :param lidar_points: Nx3 numpy array of 3D points in LiDAR frame.
    :param image_points: Nx2 numpy array of corresponding 2D points in the image.
    :param K: 3x3 numpy array, camera intrinsic matrix.
    :return: 4x4 numpy array, extrinsic matrix.
    """
    assert lidar_points.shape[0] == image_points.shape[0], "Number of points must match"

    # Solve PnP (Estimate rotation and translation)
    success, rvec, tvec = cv2.solvePnP(lidar_points, image_points, K, None)

    if not success:
        raise ValueError("PnP solution failed!")

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Construct 4x4 transformation matrix
    T_cam_lidar = np.eye(4)
    T_cam_lidar[:3, :3] = R
    T_cam_lidar[:3, 3] = tvec.flatten()

    return T_cam_lidar

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


import numpy as np
import open3d as o3d

def detect_plane_ransac(lidar_pc, distance_threshold=0.05, ransac_n=3, num_iterations=1000):
    """
    Detects a plane in the LiDAR point cloud using RANSAC.

    :param lidar_pc: Nx3 numpy array of LiDAR points.
    :param distance_threshold: Maximum distance for inliers.
    :param ransac_n: Number of points to sample for plane fitting.
    :param num_iterations: RANSAC iterations.
    :return: Plane coefficients (a, b, c, d), inlier points, outlier points, and 2D bounding box corners.
    """
    # Convert to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(lidar_pc)

    # RANSAC plane fitting
    plane_model, inliers = pcd.segment_plane(distance_threshold, ransac_n, num_iterations)
    a, b, c, d = plane_model

    # Separate inlier (plane) and outlier points
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    # Get 2D bounding box points
    bounding_box_2d_points = get_plane_2d_bounding_box(np.asarray(inlier_cloud.points), plane_model)

    return (a, b, c, d), inlier_cloud, outlier_cloud, bounding_box_2d_points


def get_plane_2d_bounding_box(inlier_points, plane_model):
    """
    Computes the 2D bounding box of the detected plane by projecting points onto the plane.

    :param inlier_points: Nx3 numpy array of points belonging to the plane.
    :param plane_model: Plane coefficients (a, b, c, d).
    :return: 4 corner points of the 2D bounding box projected onto the plane.
    """
    # Extract plane normal
    a, b, c, d = plane_model
    normal = np.array(plane_model[:3])
    
    # Find two orthogonal vectors on the plane
    if np.allclose(normal[:2], 0):  # Handle edge case where normal is aligned with Z-axis
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 0, 1])
    
    u /= np.linalg.norm(u)  # Normalize vector u
    v = np.cross(normal, u)  # Compute second orthogonal vector
    v /= np.linalg.norm(v)  # Normalize vector v

    # Project points onto the local 2D coordinate system (u-v basis)
    projected_points = np.dot(inlier_points, np.vstack([u, v]).T)

    # Compute min and max bounds in this 2D space
    min_bound = np.min(projected_points, axis=0)
    max_bound = np.max(projected_points, axis=0)

    # Define the corners of the bounding box in the 2D space
    corners_2d = np.array([
        [min_bound[0], min_bound[1]],
        [max_bound[0], min_bound[1]],
        [max_bound[0], max_bound[1]],
        [min_bound[0], max_bound[1]]
    ])

    # Convert corners back to 3D space on the plane
    corners_3d = []
    for corner in corners_2d:
        point_on_plane = corner[0] * u + corner[1] * v - d * normal / np.dot(normal, normal)
        corners_3d.append(point_on_plane)
    
    return np.array(corners_3d)


def visualize_plane(inlier_cloud, outlier_cloud, bounding_box_2d_points):
    """
    Visualizes the detected plane with its 2D bounding box.

    :param inlier_cloud: Open3D point cloud containing plane points.
    :param outlier_cloud: Open3D point cloud containing non-plane points.
    :param bounding_box_2d_points: 4 corner points of the 2D bounding box on the plane.
    """
    inlier_cloud.paint_uniform_color([1, 0, 0])  # Red for the plane
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])  # Gray for other points

    # Create bounding box visualization
    bounding_box_pcd = o3d.geometry.PointCloud()
    bounding_box_pcd.points = o3d.utility.Vector3dVector(bounding_box_2d_points)
    bounding_box_pcd.paint_uniform_color([0, 1, 0])  # Green for bounding box corners
    
    # Create a bounding box line set (connect corners)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0]   # Edges of the rectangle
    ]
    
    bounding_box_lines = o3d.geometry.LineSet()
    bounding_box_lines.points = o3d.utility.Vector3dVector(bounding_box_2d_points)
    bounding_box_lines.lines = o3d.utility.Vector2iVector(lines)
    
    bounding_box_lines.paint_uniform_color([0, 1, 0])  # Green for bounding box edges
    
    # Visualize
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud, bounding_box_pcd, bounding_box_lines])

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

        data = np.load(lidar_path)
        lidar_pc = data['arr_0']
        lidar_pc = process_ouster(lidar_pc)

        plane_eq, inlier_cloud, outlier_cloud, bounding_box_points = detect_plane_ransac(lidar_pc)
        visualize_plane(inlier_cloud, outlier_cloud, bounding_box_points)
        print(bounding_box_points)

        output_image = overlay_lidar_on_image(color_img, lidar_pc, P)
        cv2.imshow("LiDAR Projection", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        

if __name__ == "__main__":
    main()