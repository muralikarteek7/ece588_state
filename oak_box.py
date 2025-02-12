import os
import numpy as np
import cv2
import open3d as o3d
import rospy
from sensor_msgs.msg import CameraInfo
from image_geometry import PinholeCameraModel
import numpy as np
# Initialize ROS node
rospy.init_node("depth_to_pointcloud", anonymous=True)

# Fetch camera info from ROS
camera_info_msg = rospy.wait_for_message("/oak/stereo/camera_info", CameraInfo)
print("Camera info received")

# Initialize the Pinhole Camera Model
camera_model = PinholeCameraModel()
camera_model.fromCameraInfo(camera_info_msg)

# Load depth image
folder = "./scans"
depth_path = os.path.join(folder, "depth1.tif")

# Load depth image as is (without normalization)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

# Check if depth image loaded correctly
if depth is None or depth.size == 0:
    print("Error: Depth image is empty or not loaded correctly.")
    exit(1)

print(f"Depth image loaded: shape={depth.shape}, min={np.min(depth)}, max={np.max(depth)}")

# Normalize depth for visualization (optional)
#depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Show original depth image
cv2.imshow("Original Depth Image", depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Define the x and y ranges, and the pixel value range
x_range = (530, 630)
y_range = (490, 595)
depth_value_range = (29931, 31931)
deplist = []
# Filter depth image based on pixel coordinates and depth value range
filtered_depth = np.zeros_like(depth)
for u in range(y_range[0], y_range[1]):
    for v in range(x_range[0], x_range[1]):
        d = depth[u, v]
        deplist.append(d)
        print("hi d", d)
        if depth_value_range[0] <= d <= depth_value_range[1]:
            filtered_depth[u, v] = d
            print("hi")
median_depth = np.median(deplist)
print(f"Median of depth list: {median_depth}")
# Normalize filtered depth for visualization (optional)
#filtered_depth_vis = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Show filtered depth image
cv2.imshow("Filtered Depth Image", filtered_depth)

# Draw a bounding box around the filtered region
cv2.rectangle(depth, (x_range[0], y_range[0]), (x_range[1], y_range[1]), (0, 255, 0), 2)
cv2.putText(depth, "Filtered Region", (x_range[0], y_range[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the depth image with bounding box
cv2.imshow("Depth Image with Bounding Box", depth)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Convert filtered depth map to point cloud
points = []
scale_factor = 75.0  # Convert from mm to cm
valid_depth_values = []
for u in range(filtered_depth.shape[0]):
    for v in range(filtered_depth.shape[1]):
        d = filtered_depth[u, v]
        if d > 0:  # Ignore zero-depth points
            x, y, z = camera_model.projectPixelTo3dRay((v, u))  # Get unit direction vector
            points.append((x * d / scale_factor, y * d / scale_factor, z * d / scale_factor))  # Apply scale factor
            valid_depth_values.append(d)

# Compute the average depth
average_depth = np.mean(valid_depth_values)
print(f"Average Depth: {average_depth} mm")
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(np.array(points))

# Visualize point cloud
o3d.visualization.draw_geometries([pcd], window_name="Filtered Point Cloud")

# Wait for key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
